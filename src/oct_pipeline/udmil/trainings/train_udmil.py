

import os
import copy


import torch.nn.functional as F
import torch

from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.oct_pipeline.udmil.trainings.losses import udmil_loss
from src.oct_pipeline.udmil.models.udmil.udmil import get_ensemble_predictions
from sklearn.metrics import f1_score

from tqdm import tqdm




def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train == 'train':
        return create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )

    # eval transform
    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    size = int(args.input_size / crop_pct)
    t = [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(t)



def train_one_epoch(
    train_dataloader,
    model,
    device,
    optimizer,
    reliability_manager,
    checkpoints,
    total_instances,
):

    model.train()

    total_loss = 0.0

    for batch in train_dataloader:

        images = batch["images"].to(device)

        # --------------------------------------------------
        # 1. Initialisation des labels propagés
        # --------------------------------------------------

        instance_labels, initial_reliability = (
            reliability_manager.initialization_reliability(batch)
        )

        instance_labels = instance_labels.to(device).float()

        # --------------------------------------------------
        # 2. Mise à jour de Reliability R
        # --------------------------------------------------

        # On ne peux avoir 0 ou un seul checkpoint
        if len(checkpoints) >= 2:

            p_M = get_ensemble_predictions(
                model=model,
                checkpoints=checkpoints,
                images=images,
                device=device,
            )

            reliability = reliability_manager.update_reliability(
                batch=batch,
                p_M=p_M,
            )

        else:

            # Au début, toutes les instances sont fiables
            reliability = initial_reliability.to(device)

        reliability = reliability.to(device).float()

        # --------------------------------------------------
        # 3. Forward avec le modèle courant
        # --------------------------------------------------

        logits = model(
            images,
            return_features=False
        ).reshape(-1)

        # print("logits", logits.shape)
        # print("logits", instance_labels.shape)
        # print("reliability", reliability.shape)

        # image [4, 3, 224, 224]
        # model(images).shape [4, 1]
        # logits.shape [4]

        # --------------------------------------------------
        # 4. Loss UD-MIL
        # --------------------------------------------------

        loss_dict = udmil_loss(
            logits=logits,
            instance_labels=instance_labels,
            reliability=reliability,
            classifier_weights=[model.classifier.weight],
            total_instances=total_instances,
        )

        loss = loss_dict["loss"]

        # --------------------------------------------------
        # 5. Optimisation
        # --------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        train_dataloader.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / len(train_dataloader)



def val_one_epoch(
    val_dataloader,
    model,
    device,
    threshold=0.5,
):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in val_dataloader:

            images = batch["images"].to(device)

            bag_labels = batch["bag_labels"].to(device)
            bag_ids = batch["bag_ids"].to(device)

            logits = model(
                images,
                return_features=False
            ).reshape(-1)

            probabilities = torch.sigmoid(logits)

            # Agrégation des prédictions par volume
            n_bags = len(bag_labels)

            bag_probabilities = torch.zeros(
                n_bags,
                device=device
            )

            bag_probabilities.scatter_reduce_(
                0,
                bag_ids,
                probabilities,
                reduce="amax",
                include_self=False
            )

            predictions = (
                bag_probabilities >= threshold
            ).int()

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                bag_labels.cpu().tolist()
            )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    return f1

def udmil_training_all(
    train_loader,
    val_loader,
    device,
    epochs,
    early_stopping,
    model,
    reliability_managers,
    total_instances,
    learning_rate,
    checkpoint_dir,
):

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"udmil_best.pth"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    best_f1 = float("-inf")
    patience_counter = 0

    epoch_checkpoints = []
    checkpoints = []
    val_scores = []

    for epoch in range(epochs):

        # -------------------------
        # Training
        # -------------------------

        train_bar = tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{epochs}"
        )

        train_loss = train_one_epoch(
            train_dataloader=train_bar,
            model=model,
            device=device,
            optimizer=optimizer,
            reliability_manager=reliability_managers,
            checkpoints=checkpoints,
            total_instances=total_instances,
        )

        # -------------------------
        # Validation
        # -------------------------

        val_bar = tqdm(
            val_loader,
            desc=f"Val {epoch + 1}/{epochs}"
        )

        val_f1 = val_one_epoch(
            val_dataloader=val_bar,
            model=model,
            device=device,
        )

        val_scores.append(val_f1)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Train loss: {train_loss:.4f} "
            f"Val F1: {val_f1:.4f}"
        )

        # -------------------------
        # Best checkpoint
        # -------------------------

        if val_f1 > best_f1:

            best_f1 = val_f1
            patience_counter = 0

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

        else:
            patience_counter += 1

        # --------------------------------------
        # ADD CHECKPOINT TO ENSEMBLE
        # --------------------------------------

        checkpoint = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

        epoch_checkpoints.append(checkpoint)

        # Garder uniquement les M (5) derniers checkpoints
        # checkpoints = checkpoints[-ensemble_size:]

        if (epoch + 1) % 12 == 0:

            recent_scores = val_scores[-12:]
            recent_checkpoints = epoch_checkpoints[-12:]

            # Récupérer l'indice de val_score maximum
            best_epoch = max(range(12), key=lambda i: recent_scores[i])

            start = best_epoch - 2
            end = best_epoch + 3

            checkpoints = recent_checkpoints[start:end]
            best_checkpoints = recent_checkpoints[best_epoch]

            print(
                f"Best epoch: {best_epoch + 1} | "
                f"Selected checkpoints: "
                f"{start + 1} to {end}"
            )

            #--------------------
            # Fine tuning
            # -------------------

            model.load_state_dict(best_checkpoints)

            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze final connected layer
            for param in model.fc.parameters():
                param.requires_grad = True

            # Recreate optimizer with trainable parameters only
            optimizer = torch.optim.Adam(
                filter(
                    lambda p: p.requires_grad,
                    model.parameters()
                ),
                lr=learning_rate
            )

            # Clear checkpoints for next block
            epoch_checkpoints = []
            val_scores = []
            patience_counter = 0

        # -------------------------
        # Early stopping
        # -------------------------

        if patience_counter >= early_stopping:

            print(
                f"Early stopping"
                f"at epoch {epoch + 1}"
            )

            break

        # -------------------------
        # Load best checkpoint
        # -------------------------

        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        )

    best_weights = copy.deepcopy(model.state_dict())
    best_f1_scores = best_f1

    print(
        f"Best F1 : {best_f1:.4f}"
    )

    return best_weights, best_f1_scores



    

        
        