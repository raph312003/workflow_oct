import argparse
from webdav4.client import Client
from io import BytesIO
import pydicom
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.widgets import Button
import yaml

from src.oct_pipeline.loading import load_opm, load_op, load_opt, intranet_connect

def autoscale(a, lo_p=1.0, hi_p=99.5):
    """Fenêtrage par percentiles -> [0,1]. Robuste quand WC/WW sont absents."""
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, (lo_p, hi_p))
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max() or 1)
    return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)

def pixel_spacing(ds):
    """Espacement (ligne, colonne) en mm ; None si absent."""
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        if tag in ds:
            return [float(v) for v in ds[tag].value]
    sh = ds.get("SharedFunctionalGroupsSequence")
    if sh:
        try:
            return [float(v) for v in sh[0].PixelMeasuresSequence[0].PixelSpacing]
        except Exception:
            pass
    return None

def show_bscans(ds, cube, idx=None):
    n = cube.shape[0]
    idx = idx or [0, n // 4, n // 2, 3 * n // 4, n - 1]
    idx = sorted(set(i for i in idx if 0 <= i < n))
    ps = pixel_spacing(ds)
    asp = (ps[0] / ps[1]) if ps else "auto"
    fig, axes = plt.subplots(len(idx), 1, figsize=(4 * len(idx), 4))
    for ax, i in zip(np.atleast_1d(axes), idx):
        ax.imshow(autoscale(cube[i]), cmap="gray", aspect=asp)
        ax.set_title(f"B-scan {i}/{n-1}"); ax.axis("off")
    plt.tight_layout(); plt.show()

def en_face(cube, z0=None, z1=None):
    """Projection d'intensité moyenne sur l'axe axial -> (n_bscans, largeur)."""
    sub = cube[:, z0:z1, :]
    return sub.mean(axis=1)


def save_patient_images(client, inv, patient_ids, laterality,
                        output_dir="datasets/show_patient"):

    os.makedirs(output_dir, exist_ok=True)

    for patient_id in patient_ids:

        patient_data = inv[
            (inv["patient"] == patient_id) &
            (inv["laterality"] == laterality)
        ]

        print(f"\nPatient {patient_id} - {laterality}")

        # Dossier du patient
        patient_dir = os.path.join(
            output_dir, f"patient_{patient_id}"
        )
        os.makedirs(patient_dir, exist_ok=True)

        # OP
        for p in patient_data.loc[
            patient_data.modality == "OP", "path"
        ]:

            ds, img = load_op(client, p)

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img, cmap="gray")
            ax.set_title(
                f"OP - {ds.get('ImageLaterality', '?')} "
                f"- {os.path.basename(p)}"
            )
            ax.axis("off")

            fig.savefig(
                os.path.join(patient_dir, "OP.png"),
                dpi=200,
                bbox_inches="tight"
            )
            plt.close(fig)

            break  # Première acquisition OP

        # OPT
        for p in patient_data.loc[
            patient_data.modality == "OPT", "path"
        ]:

            ds, cube = load_opt(client, p)

            # B-scans
            n = cube.shape[0]
            idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
            idx = sorted(set(idx))

            ps = pixel_spacing(ds)
            asp = (ps[0] / ps[1]) if ps else "auto"

            fig, axes = plt.subplots(
                len(idx), 1,
                figsize=(12, 4 * len(idx))
            )

            for ax, i in zip(np.atleast_1d(axes), idx):
                ax.imshow(
                    autoscale(cube[i]),
                    cmap="gray",
                    aspect=asp
                )
                ax.set_title(f"B-scan {i}/{n-1}")
                ax.axis("off")

            fig.suptitle(
                f"OPT - {ds.get('ImageLaterality', '?')} "
                f"- {cube.shape} "
                f"range [{cube.min():.0f}, {cube.max():.0f}] "
                f"spacing={pixel_spacing(ds)} "
            )

            fig.savefig(
                os.path.join(patient_dir, "OPT.png"),
                dpi=200,
                bbox_inches="tight"
            )
            plt.close(fig)

            # Projection en-face
            proj = en_face(cube)

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(
                autoscale(proj),
                cmap="gray",
                aspect="auto"
            )
            ax.set_title(
                f"OPT En-face - {ds.get('ImageLaterality', '?')}"
            )
            ax.axis("off")

            fig.savefig(
                os.path.join(patient_dir, "OPT_en-face.png"),
                dpi=200,
                bbox_inches="tight"
            )
            plt.close(fig)

            break  # Première acquisition OPT

        # OPM
        for p in patient_data.loc[
            patient_data.modality == "OPM", "path"
        ]:

            ds, tmap, unit = load_opm(client, p)

            valid = tmap[tmap > 0]

            if valid.size == 0:
                print(f"OPM sans pixels valides : {p}")
                continue

            ps = pixel_spacing(ds)
            asp = (ps[0] / ps[1]) if ps else 1.0

            masked = np.ma.masked_where(tmap <= 0, tmap)

            fig, ax = plt.subplots(figsize=(10, 10))

            im = ax.imshow(
                masked,
                cmap="jet",
                aspect=asp,
                vmin=np.percentile(valid, 2),
                vmax=np.percentile(valid, 98)
            )

            fig.colorbar(im, ax=ax, label=unit)

            ax.set_title(
                f"OPM - {ds.get('ImageLaterality', '?')}"
            )
            ax.axis("off")

            fig.savefig(
                os.path.join(patient_dir, "OPM.png"),
                dpi=200,
                bbox_inches="tight"
            )
            plt.close(fig)

            break  # Première acquisition OPM

        print(f"Images sauvegardées dans {patient_dir}")


def main():
    parser = argparse.ArgumentParser(description="Show patients")
    parser.add_argument("--config", default="configs/show_patient.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    client = intranet_connect(config["username"], config["password"], config["url"])

    inv = pd.read_csv("datasets/extraction_HRA/inventaire.csv")

    # Sélectionner les DICOM d'un patient
    patient_ids = config["patient_id"]
    laterality =  config["laterality"]

    save_patient_images(
        client=client,
        inv=inv,
        patient_ids=patient_ids,
        laterality = laterality
    )

if __name__ == "__main__":
    main()