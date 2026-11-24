import argparse
from webdav4.client import Client
from io import BytesIO
import pydicom
import pandas as pd
import yaml

from src.oct_pipeline.loading import load_opm, load_op, load_opt, intranet_connect


def setup_argparse() -> argparse.Namespace:
    """Setup the command line arguments"""
    parser = argparse.ArgumentParser(
        description=(
            "Read a HRA DICOM dataset and extract and the pixel "
            "images"
        )
    )

    parser.add_argument(
        "--patient_id",
        default=6712,
        help="Path to HRA DICOM dataset",
        type=int
    )

    return parser.parse_args()

def main():
    parser = argparse.ArgumentParser(description="Show patients")
    parser.add_argument("--config", default="configs/loading_array_images.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    client = intranet_connect(config["username"], config["password"], config["url"])

    inv = pd.read_csv("extraction_HRA/inventaire.csv")

    # Sélectionner les DICOM d'un patient
    patient_id = config["patient_id"]

    patient_data = inv[inv["patient"] == patient_id]

    # Sélectionner les images OP
    op_data = patient_data[patient_data["modality"] == "OP"]

    # Sélectionner les images OT
    # ot_data = patient_data[patient_data["modality"] == "OT"]

    # Sélectionner les images OPT
    opt_data = patient_data[patient_data["modality"] == "OPT"]

    # Charger une image
    path_op = op_data.iloc[0]["path"]
    # path_ot = ot_data.iloc[0]["path"]
    path_opt = opt_data.iloc[0]["path"]

    ds_op, array = load_op(client, path_op)
    # ds_ot = load_opm(client, path_ot)
    ds_opt, cube = load_opt(client, path_opt)

    print(array.shape)
    # print(img_ot.shape)
    print(cube.shape)
    

if __name__ == "__main__":
    main()