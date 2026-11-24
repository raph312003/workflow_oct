import argparse
from webdav4.client import Client
from io import BytesIO
import pydicom
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.widgets import Button

from src.oct_pipeline.manifest import create_manifests

def setup_argparse() -> argparse.Namespace:
    """Setup the command line arguments"""
    parser = argparse.ArgumentParser(
        description=(
            "Read a HRA DICOM dataset and extract and the pixel "
            "images"
        )
    )

    return parser.parse_args()

def main():
    args = setup_argparse()

    inv = pd.read_csv("datasets/extraction_HRA/inventaire.csv")

    manifest, patient_splits = create_manifests(inv, output_dir="manifests/OCT_HRA")

    # 5. Vérifications
    
    print("\nRépartition des patients :")
    print(patient_splits["split"].value_counts())

    print("\nRépartition des acquisitions :")
    print(manifest["split"].value_counts())

    print("\nNombre de patients par split :")
    print(manifest.groupby("split")["patient"].nunique())

    print("\nNombre d'acquisitions par modalité :")
    print(pd.crosstab(manifest["split"], manifest["modality"]))


if __name__ == "__main__":
    main()