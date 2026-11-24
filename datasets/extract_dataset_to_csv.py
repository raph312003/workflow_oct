import argparse
from webdav4.client import Client
import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from pydicom.errors import InvalidDicomError
import yaml

from src.oct_pipeline.loading import walk_sr, intranet_connect

def parcourir_webdav(client, dossier):
    for entree in client.ls(dossier, detail=True):

        chemin = entree["name"]
         
        resource_type = entree.get("type")

        if resource_type == "directory":
            yield from parcourir_webdav(client, chemin)

        elif resource_type == "file":
            yield chemin

def extract_metadata_webdav(client, dossier):

    rows = []

    for chemin in parcourir_webdav(client, dossier):

        try:
            # Lecture en mémoire depuis WebDAV
            with client.open(chemin, "rb") as fichier:
                # print("fichier", fichier)
                contenu = fichier.read()

            # Lecture des métadonnées uniquement
            ds = pydicom.dcmread(
                BytesIO(contenu),
                stop_before_pixels=True
            )

            print(chemin, ds.get("Modality"))

            rows.append({
                "path": chemin,
                "modality": ds.get("Modality"),
                "sop_class": ds.SOPClassUID.name,
                "tsyntax": ds.file_meta.TransferSyntaxUID.name,
                "rows": ds.get("Rows"),
                "cols": ds.get("Columns"),
                "frames": ds.get("NumberOfFrames", 1),
                "bits_alloc": ds.get("BitsAllocated"),
                "bits_stored": ds.get("BitsStored"),
                "photometric": ds.get("PhotometricInterpretation"),
                "wc": ds.get("WindowCenter"),
                "ww": ds.get("WindowWidth"),
                "slope": ds.get("RescaleSlope"),
                "intercept": ds.get("RescaleIntercept"),
                "laterality": ds.get(
                    "ImageLaterality",
                    ds.get("Laterality")
                ),
                "date": ds.get("StudyDate"),
                "patient": ds.get("PatientID"),
            })

        except InvalidDicomError:
            print(f"Fichier non DICOM : {chemin}")

        except Exception as e:
            print(f"Erreur sur {chemin} : {e}")

    return pd.DataFrame(rows)

def main():
    # args = setup_argparse()
    parser = argparse.ArgumentParser(description="Extract dataset in csv file")
    parser.add_argument("--config", default="configs/extract_dataset_to_csv.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    

    client = intranet_connect(config["username"], config["password"], config["url"])

    sr_frames = []

    print("1) Extraction des métadonnées")

    inv = extract_metadata_webdav(
        client,
        config["dicom_path_disp"]
    )

    print("\n2) Résumé")

    print(f"\nNombre de fichiers DICOM : {len(inv)}")

    print("3) sr_frames")
    
    for p in inv.loc[inv.modality == "SR", "path"]:
        ds =    (client, p)
        df = pd.DataFrame(walk_sr(ds))
        df["path"] = os.path.basename(p)
        df["patient"] = ds.get("PatientID")
        df["laterality"] = ds.get("ImageLaterality", ds.get("Laterality"))
        df["date"] = ds.get("StudyDate")
        sr_frames.append(df)

    sr = pd.concat(sr_frames, ignore_index=True) if sr_frames else pd.DataFrame()


    OUT = "datasets/extraction_HRA"
    os.makedirs(OUT, exist_ok=True)

    inv.to_csv(f"{OUT}/inventaire.csv", index=False)
    if not sr.empty:
        sr.to_csv(f"{OUT}/mesures_sr.csv", index=False)

    print(sorted(os.listdir(OUT)))

if __name__ == "__main__":
    main()