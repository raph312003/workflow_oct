import torch
import torch.nn.functional as F


import torch
import torch.nn.functional as F


def udmil_loss(
    logits,
    instance_labels,
    reliability,
    classifier_weights,
    total_instances,
):

    logits = logits.reshape(-1)
    instance_labels = instance_labels.float().reshape(-1)
    reliability = reliability.float().reshape(-1)

    # BCE individuelle de chaque B-scan
    bce = F.binary_cross_entropy_with_logits(
        logits,
        instance_labels,
        reduction="none",
    )

    # BCE pondérée par la fiabilité
    classification_loss = (
        (reliability * bce).sum()
        / total_instances
    )

    # Regularisation L2 du classifieur
    l2_regularization = 0.5 * sum(
        param.pow(2).sum()
        for param in classifier_weights
    )

    # Loss finale
    total_loss = (
        classification_loss
        + l2_regularization
    )

    return {
        "loss": total_loss,
        "classification_loss": classification_loss,
        "l2_regularization": l2_regularization,
    }