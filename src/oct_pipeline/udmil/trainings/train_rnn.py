from sklearn.metrics import f1_score, accuracy_score

def train_one_epoch(
    train_dataloader,
    model,
    device,
    optimizer,
    criterion,
):

    model.train()

    total_loss = 0.0

    for batch in train_dataloader:

        images = batch["images"].to(device)

        bag_ids = batch["bag_ids"].to(device)
        scan_indices = batch["scan_indices"].to(device)
        bag_labels = batch["bag_labels"].to(device)

        # UDMIL + RNN
        logits = model(
            images=images,
            bag_ids=bag_ids,
            scan_indices=scan_indices,
            num_bags=len(bag_labels),
        )

        # Bag-level loss
        loss = criterion(
            logits,
            bag_labels.long()
        )

        # Optimization
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
):

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        for batch in val_dataloader:

            images = batch["images"].to(device)

            bag_ids = batch["bag_ids"].to(device)
            scan_indices = batch["scan_indices"].to(device)

            bag_labels = batch["bag_labels"].to(device)

            # CNN + RNN
            logits = model(
                images=images,
                bag_ids=bag_ids,
                scan_indices=scan_indices,
                num_bags=len(bag_labels),
            )

            loss = criterion(
                logits,
                bag_labels.long()
            )

            total_loss += loss.item()

            # Predicted bag labels
            predictions = torch.argmax(
                logits,
                dim=1
            )

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

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    val_loss = total_loss / len(val_dataloader)

    return val_loss, f1, accuracy



import os
import copy
import torch
import torch.nn as nn

from tqdm import tqdm


def RNN_trainer_all(
    train_loaders,
    val_loaders,
    device,
    epochs,
    early_stopping,
    model,
    learning_rate=1e-4,
    checkpoint_dir="model_weights",
):

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"RNN_best.pth"
    )

    # --------------------------------
    # 1. Initialize CNN + RNN
    # --------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    criterion = nn.CrossEntropyLoss()

    best_f1 = float("-inf")
    patience_counter = 0

    # --------------------------------
    # 2. Training loop
    # --------------------------------

    for epoch in range(epochs):

        train_bar = tqdm(
            train_loaders,
            desc=f"Train {epoch + 1}/{epochs}"
        )

        train_loss = train_one_epoch(
            train_dataloader=train_bar,
            model=model,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
        )

        # --------------------------------
        # 3. Validation
        # --------------------------------

        val_bar = tqdm(
            val_loaders,
            desc=f" Val {epoch + 1}/{epochs}"
        )

        val_loss, val_f1, val_accuracy = val_one_epoch(
            val_dataloader=val_bar,
            model=model,
            device=device,
        )

        print(
            f"| Epoch {epoch + 1}/{epochs} "
            f"| Train loss: {train_loss:.4f} "
            f"| Val loss: {val_loss:.4f} "
            f"| Val F1: {val_f1:.4f} "
            f"| Val Acc: {val_accuracy:.4f}"
        )

        # --------------------------------
        # 4. Best checkpoint
        # --------------------------------

        if val_f1 > best_f1:

            best_f1 = val_f1
            patience_counter = 0

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

        else:

            patience_counter += 1

        # --------------------------------
        # 5. Early stopping
        # --------------------------------

        if patience_counter >= early_stopping:

            print(
                f"Early stopping at epoch {epoch + 1}"
            )

            break

    # --------------------------------
    # 6. Load best model
    # --------------------------------

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )

    best_weights = model
    best_f1_scores = best_f1

    print(
        f"Best F1 : {best_f1:.4f}"
    )

    return best_weights, best_f1_scores

