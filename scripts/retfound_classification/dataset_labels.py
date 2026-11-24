
import os
import pandas as pd
import argparse

# manifest = pd.read_csv("manifest.csv")

DATA_PATH = "/data/original"
OUTPUT_PATH = "/data/retfound_dataset"

def setup_argparse() -> argparse.Namespace:
    """Setup the command line arguments"""
    parser = argparse.ArgumentParser(
        description=(
            "Read a HRA DICOM dataset and extract and the pixel "
            "images"
        )
    )

    parser.add_argument(
        "--manifests_files",
        default="manifests/OCT_HRA/OCT_HRA.csv",
        help="Path to OCT HRA",
        type=str
    )

    parser.add_argument(
        "--labels_files",
        default="labels.csv",
        help="Creation of a label csv files patient, laterality, label",
        type=str
    )

    parser.add_argument(
        "--output_dir",
        default="scripts/retfound_classification",
        help="Path to OCT HRA",
        type=str
    )

    return parser.parse_args()

def main():
    args =  setup_argparse()
    labels = pd.read_csv(args.labels_files)
    manifests = pd.read_csv(args.manifests_files)
   

    # Informations utiles sur les images
    manifests = manifests.loc[
        manifests["modality"] == "OPT",
        ["path", "patient", "laterality", "split"]
    ]

    # Informations utiles sur les labels
    labels = labels[["patient", "laterality", "label"]]

    # Fusion
    df = manifests.merge(
        labels,
        on=["patient", "laterality"],
        how="inner",
        validate="one_to_one"
    )

    OUTPUT_DIR = args.output_dir

    df.to_csv(
        os.path.join(OUTPUT_DIR, "dataset.csv"),
        index=False 
    )

    df.to_json(
        os.path.join(
            OUTPUT_DIR,
            f"dataset.jsonl"
        ),
        orient="records",
        lines=True,
        force_ascii=False
    )

    # return df

if __name__ == "__main__":
    main()