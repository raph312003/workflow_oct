from src.oct_pipeline.udmil.data.load_data import OCTDataset, collate_bags
from src.oct_pipeline.loading import intranet_connect
from src.oct_pipeline.param import parameters_info

from src.oct_pipeline.udmil.models.udmil.model import InstanceClassifier, RESnet50
from src.oct_pipeline.udmil.trainings.test_udmil import udmil_testing_all

import yaml
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

class BinaryClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if self.category == 1:
            return model_output.squeeze()

        return -model_output.squeeze()

# Réseau produit un logit en sortie
# y=1 sigma(logit)
# y=0 -sigma(logit)

def main():
    # args = setup_argparse()
    parser = argparse.ArgumentParser(description="Extract dataset in csv file")
    parser.add_argument("--config", default="configs/udmil/evaluation.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    

    client = intranet_connect(config["username"], config["password"], config["url"])
    #transform = build_transform(is_train, args)

    inv = pd.read_csv(config["dataset"])

    patient_data_opt_test = inv[inv["split"] == "test"]#.reset(drop=True)

    dataset_test = OCTDataset(
        df=patient_data_opt_test,
        client=client,
        size = config["size"]
    )

    subset_test = Subset(
        dataset_test,
        range(min(1, len(dataset_test)))
    )

    test_loader = DataLoader(
        subset_test,
        config["batch_size"],
        shuffle=True,
        collate_fn=collate_bags
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = RESnet50()
    
    model_udmil = InstanceClassifier(
        backbone=backbone,
        feature_dim=config["feature_dim"]
    ).to(device)
    
    best_weights_udmil = torch.load(
        config["checkpoint_udmil"],
        map_location=device,
        weights_only=True,
    )

    model_udmil.load_state_dict(best_weights_udmil)
    # parameters_info(model_udmil)

    target_layers = model_udmil.backbone.layer4[-1]
    
    cam = GradCAM(
        model = model_udmil,
        target_layers = [target_layers]
    ) # choose class index

    f1_score, accuracy = udmil_testing_all(
        test_loader,
        cam,
        device,
        model_udmil,
        config["display_dir"],
        best_weights_udmil,
        threshold=0.5
    )

    print("f1_score", f1_score)
    print("accuracy", accuracy)

if __name__ == "__main__":
    main()