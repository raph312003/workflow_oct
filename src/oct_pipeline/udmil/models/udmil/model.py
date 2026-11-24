
from torchvision import models
import torch.nn as nn
import torch


def RESnet50(**kwargs):

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # Récupérer la première convolution
    old_conv = model.conv1

    # Créer une convolution à un seul canal
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    # Initialiser avec les poids pré-entraînés
    with torch.no_grad():
        new_conv.weight.copy_(
            old_conv.weight.mean(dim=1, keepdim=True)
        )

    model.conv1 = new_conv

    # Supprimer le classifieur ImageNet
    model.fc = nn.Identity()

    return model


class InstanceClassifier(nn.Module):

    def __init__(self, backbone, feature_dim=516):
        super().__init__()

        self.backbone = backbone

        # Réduire la dim de sortie
        self.projection = nn.Linear(2048, feature_dim)
        self.classifier = nn.Linear(
            feature_dim,
            1
        )

    def forward_features(self, images):

        features = self.backbone(images)
        features = self.projection(features)

        return features

    def forward(self, images, return_features=False):

        features = self.forward_features(images)

        logits = self.classifier(features)

        if return_features:
            return logits, features

        return logits



import torch
import torch.nn as nn

from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
)


class RNN_classifier(nn.Module):

    def __init__(
        self,
        feature_dim=512,
        hidden_dim=256,
        fc_hidden_dim=128,
        num_classes=2,
        dropout=0.5,
    ):
        super().__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Dropout after LSTM
        self.dropout_lstm = nn.Dropout(dropout)

        # First FC layer
        self.fc1 = nn.Linear(
            hidden_dim,
            fc_hidden_dim
        )

        self.relu = nn.ReLU()

        # Dropout after first FC
        self.dropout_fc = nn.Dropout(dropout)

        # Final FC layer
        self.fc2 = nn.Linear(
            fc_hidden_dim,
            num_classes
        )

    def update_instance(self, instance_features, hidden_state=None):
        """
        Process one or several consecutive B-scans.

        instance_features: [B, 1, feature_dim]
        hidden_state: (hidden, cell)
        """

        output, hidden_state = self.lstm(
            instance_features,
            hidden_state
        )

        return output, hidden_state

    def forward(
        self,
        instance_features,
        bag_ids,
        scan_indices,
        num_bags=None,
    ):
        """
        instance_features: [N, feature_dim]
        bag_ids:           [N]
        scan_indices:      [N]
        """

        if num_bags is None:
            num_bags = int(bag_ids.max().item()) + 1

        # --------------------------------
        # 1. Reconstruct each OCT volume
        # --------------------------------

        sequences = []

        for bag_idx in range(num_bags):

            # Select B-scans belonging to this volume
            mask = bag_ids == bag_idx

            volume_features = instance_features[mask]
            volume_indices = scan_indices[mask]

            # Sort B-scans according to their scan index
            sorted_indices = torch.argsort(volume_indices)

            volume_features = volume_features[sorted_indices]

            sequences.append(volume_features)

        # --------------------------------
        # 2. Handle variable sequence length
        # --------------------------------

        lengths = torch.tensor(
            [len(seq) for seq in sequences],
            dtype=torch.long
        )

        padded_sequences = pad_sequence(
            sequences,
            batch_first=True,
        )

        # Shape: [num_bags, max_scans, feature_dim]

        packed_sequences = pack_padded_sequence(
            padded_sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # --------------------------------
        # 3. LSTM
        # --------------------------------

        _, (hidden, cell) = self.lstm(
            packed_sequences
        )

        # Last hidden state
        # Shape: [num_bags, hidden_dim]

        bag_features = hidden[-1]

        bag_features = self.dropout_lstm(bag_features)

        # --------------------------------
        # 4. Fully connected layers
        # --------------------------------

        x = self.fc1(bag_features)
        x = self.relu(x)

        x = self.dropout_fc(x)

        logits = self.fc2(x)

        return logits

    def predict_proba(
        self,
        instance_features,
        bag_ids,
        scan_indices,
        num_bags=None,
    ):

        logits = self.forward(
            instance_features,
            bag_ids,
            scan_indices,
            num_bags,
        )

        return torch.softmax(logits, dim=1)

class OCT_RNN_Model(nn.Module):

    def __init__(self, udmil, rnn):
        super().__init__()

        self.udmil = udmil
        self.rnn = rnn

    def forward(
        self,
        images,
        bag_ids,
        scan_indices,
        num_bags=None,
    ):

        # CNN feature extraction
        # [N, 3, 224, 224] -> [N, feature_dim]
        _, instance_features = self.udmil(
            images,
            return_features=True
        )

        # RNN bag-level classification
        logits = self.rnn(
            instance_features=instance_features,
            bag_ids=bag_ids,
            scan_indices=scan_indices,
            num_bags=num_bags,
        )

        return logits





