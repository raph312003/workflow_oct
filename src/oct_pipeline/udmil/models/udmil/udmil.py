
import numpy as np


import torch


class ReliabilityManager:

    def __init__(self, tau=0.5):
        self.tau = tau

    def initialization_reliability(self, batch):

        bag_labels = batch["bag_labels"]
        bag_ids = batch["bag_ids"]

        instance_labels = bag_labels[bag_ids].long()

        reliability = torch.ones(
            len(instance_labels),
            dtype=torch.float32
        )

        return instance_labels, reliability

    @torch.no_grad()
    def update_reliability(self, batch, p_M):

        bag_labels = batch["bag_labels"]
        bag_ids = batch["bag_ids"]

        mu = p_M.mean(dim=0)

        sigma = p_M.std(
            dim=0,
            unbiased=False
        )

        reliability = torch.ones(
            len(mu),
            device=mu.device,
            dtype=torch.float32
        )

        for bag_idx in range(len(bag_labels)):

            # Bags négatifs : toutes les instances restent fiables
            if bag_labels[bag_idx] == 0:
                continue

            mask = bag_ids == bag_idx

            indices = torch.where(mask)[0]

            mu_t = mu[indices]
            sigma_t = sigma[indices]

            gamma_t = sigma_t.mean()

            reliable_mask = (
                (mu_t > self.tau)
                &
                (sigma_t < gamma_t)
            )

            reliability[indices] = reliable_mask.float()

            # Au moins une instance doit être sélectionnée
            if reliable_mask.sum() == 0:

                reliability[indices] = 0

                best_idx = torch.argmax(mu_t)

                reliability[indices[best_idx]] = 1

        return reliability



import torch


@torch.no_grad()
def get_ensemble_predictions(
    model,
    checkpoints,
    images,
    device,
):

    model.eval()

    all_predictions = []

    current_state = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
    }

    for checkpoint in checkpoints:

        model.load_state_dict(checkpoint)

        logits = model(images).reshape(-1)

        probabilities = torch.sigmoid(logits)

        all_predictions.append(probabilities)

    # [M, N]
    p_M = torch.stack(all_predictions, dim=0)

    # Restaurer les poids actuels du modèle
    model.load_state_dict(current_state)

    model.train()

    return p_M




        


            


    

        