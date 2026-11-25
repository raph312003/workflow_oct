import os
import torch
from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score
from src.oct_pipeline.loading import show_oct, show_oct_cam


def udmil_testing_all(
    test_loader,
    cam,
    device,
    model,
    checkpoint_dir,
    best_weights_udmil,
    threshold=0.5
):

    os.makedirs(checkpoint_dir, exist_ok=True)

    model.load_state_dict(best_weights_udmil)
    model.eval()

    all_predictions = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for batch in test_loader:
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

            # Affichage
            print()
            for idx in range(bag_labels.shape[0]):
                print("idx", idx)
                # Sélectionner les B-scans du bag idx
                indice = bag_ids == idx
        
                # Récupérer les images correspondantes
                cube = images[indice]
        
                ps = batch["pixel_spacing"][idx]
                id_p = batch["patient_ids"][idx]
                id_v = batch["volume_ids"][idx]

                show_oct_cam(cube, cam, checkpoint_dir, id_p, id_v, ps)

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )
        
    return f1, accuracy


    

        
        