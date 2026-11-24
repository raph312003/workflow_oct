from src.oct_pipeline.udmil.data.load_data import OCTDataset, collate_bags
from src.oct_pipeline.loading import intranet_connect

from src.oct_pipeline.udmil.models.udmil.model import InstanceClassifier, RESnet50, OCT_RNN_Model, RNN_classifier
from src.oct_pipeline.udmil.trainings.train_rnn import RNN_trainer_all
from src.oct_pipeline.udmil.trainings.train_udmil import udmil_training_all
from src.oct_pipeline.udmil.models.udmil.udmil import ReliabilityManager

import yaml
import argparse
import pandas as pd
from torch.utils.data import DataLoader

# import matplotlib.pyplot as plt
# import numpy as np
# import os

# from src.oct_pipeline.loading import show_oct

import torch
from torch.utils.data import DataLoader, Subset
import yaml

def main():
    # args = setup_argparse()
    parser = argparse.ArgumentParser(description="Extract dataset in csv file")
    parser.add_argument("--config", default="configs/udmil/train.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    

    client = intranet_connect(config["username"], config["password"], config["url"])
    #transform = build_transform(is_train, args)

    inv = pd.read_csv(config["dataset"])

    patient_data_opt = inv[inv["split"] == "train"]#.reset(drop=True)
    patient_data_opt = inv[inv["split"] == "val"]#.reset(drop=True)

    dataset_train = OCTDataset(
        df=patient_data_opt,
        client=client,
        size = (768, 486)
    )

    dataset_val = OCTDataset(
        df=patient_data_opt,
        client=client,
        size = (768, 486)
    )

    dataset_train_test = Subset(
        dataset_train,
        range(min(2, len(dataset_train)))
    )

    dataset_val_test = Subset(
        dataset_val,
        range(min(1, len(dataset_val)))
    )

    train_loader = DataLoader(
        dataset_train_test,
        config["batch_size"],
        shuffle=True,
        collate_fn=collate_bags
    )

    val_loader = DataLoader(
        dataset_val_test,
        config["batch_size"],
        shuffle=False,
        collate_fn=collate_bags
    )

    # train_dataiter = iter(train_loader)
    # scans_train = next(train_dataiter)
    # print(type(scans_train))
    # print(scans_train["images"].shape)
    # print(scans_train["bag_labels"].shape)
    # print(scans_train["bag_ids"].shape)
    # print(scans_train["scan_indices"].shape)
    # print(len(scans_train["patient_ids"]))
    # print(len(scans_train["volume_ids"]))
    # print(len(scans_train["pixel_spacing"]))
    

    # val_dataiter = iter(val_loader)
    # scans_val = next(val_dataiter)
    # print(type(scans_val))
    # print(scans_val["images"].shape)
    # print(scans_val["bag_labels"].shape)
    # print(scans_val["bag_ids"].shape)
    # print(scans_val["scan_indices"].shape)
    # print(len(scans_val["patient_ids"]))
    # print(len(scans_val["volume_ids"]))
    # print(len(scans_val["pixel_spacing"]))

    # # ---- show bscans ----
    # for idx in range(scans_val["bag_labels"].shape[0]):
    #     # Sélectionner les B-scans du bag idx
    #     indice = scans_val["bag_ids"] == idx

    #     # Récupérer les images correspondantes
    #     cube = scans_val["images"][indice]

    #     ps = scans_val["pixel_spacing"][idx]
    #     id_p = scans_val["patient_ids"][idx]
    #     id_v = scans_val["volume_ids"][idx]
    #     print("id", id)

    #     print(cube.shape)
    #     show_oct(cube[:,0,:,:].numpy(), "images", id_p, id_v, ps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    udmil = config["model_udmil"]
    rnn = config["model_rnn"]

    # ----------
    # udml
    # ----------

    # Manager propre au modèle récuperer objet
    reliability_managers = ReliabilityManager(udmil["tau"])

    # Nombre total d'instances propre au dataset
    total_instances = sum(
        len(sample["images"])
        for sample in val_loader.dataset
    )

    backbone = RESnet50()

    model_udmil = InstanceClassifier(
        backbone=backbone,
        feature_dim=udmil["feature_dim"]
    ).to(device)

    # training udmil
    best_weights_udmil, best_f1_scores_udmil = udmil_training_all(
        train_loader,
        val_loader,
        device,
        udmil["epochs"],
        udmil["early_stopping"],
        model_udmil,
        reliability_managers,
        total_instances,
        udmil["lr"],
        udmil["checkpoint_dir"],
    )
    
    # ---------
    # RNN
    # ---------

    # Model RNN
    model_rnn = RNN_classifier(
        feature_dim=rnn["feature_dim"],
        hidden_dim=rnn["hidden_dim"],
        fc_hidden_dim=rnn["fc_hidden_dim"],
        num_classes=rnn["num_classes"],
        dropout=rnn["dropout"],
    )

    model_udmil.load_state_dict(best_weights_udmil)

    # Combiner les modeles pour récupérer les features de 
    combine_model = OCT_RNN_Model(
        udmil=model_udmil,
        rnn=model_rnn
    ).to(device)
    
    # RNN_trainer_all
    best_weights_rnn, best_f1_scores_rnn = RNN_trainer_all(
        train_loader,
        val_loader,
        device,
        rnn["epochs"],
        rnn["early_stopping"],
        combine_model,
        learning_rate=rnn["lr"],
        checkpoint_dir=rnn["model_weights"],
    )
    print("----Results-----")

    print("best_f1_scores_udmil", best_f1_scores_udmil)
    print("best_f1_scores_rnn", best_f1_scores_rnn)

if __name__ == "__main__":
    main()