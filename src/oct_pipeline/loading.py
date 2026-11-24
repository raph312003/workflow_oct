import argparse
from webdav4.client import Client
from io import BytesIO
import pydicom
import pandas as pd
from pydicom.errors import InvalidDicomError
import numpy as np
import matplotlib.pyplot as plt
import os

def read_dicom_webdav(client, path, **kwargs):
    with client.open(path, "rb") as fichier:
        return pydicom.dcmread(BytesIO(fichier.read()), **kwargs)

def to_float(ds, arr):
    """Applique Rescale slope/intercept si présents."""
    slope = float(ds.get("RescaleSlope", 1) or 1)
    inter = float(ds.get("RescaleIntercept", 0) or 0)
    return arr.astype(np.float32) * slope + inter

def load_ot(client, path):
    ds = read_dicom_webdav(client, path)

def load_opm(client, path):
    """Retourne (ds, carte en µm). Gère RealWorldValueMapping si présent."""
    ds = read_dicom_webdav(client, path)
    arr = to_float(ds, ds.pixel_array)
    unit = "unité brute"
    rw = ds.get("RealWorldValueMappingSequence")
    if rw:
        item = rw[0]
        s = float(item.get("RealWorldValueSlope", 1))
        i = float(item.get("RealWorldValueIntercept", 0))
        arr = arr * s + i
        try:
            unit = item.MeasurementUnitsCodeSequence[0].CodeMeaning
        except Exception:
            unit = "µm"
    return ds, arr, unit

def load_opt(client, path):
    ds = read_dicom_webdav(client, path)
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    return ds, to_float(ds, arr)

def load_op(client, path):
    ds = read_dicom_webdav(client, path)
    arr = ds.pixel_array
    if ds.get("PhotometricInterpretation", "").startswith("YBR"):
        from pydicom.pixel_data_handlers.util import convert_color_space
        arr = convert_color_space(arr, ds.PhotometricInterpretation, "RGB")
    return ds, arr

def walk_sr(item, path=""):
    """Parcourt récursivement ContentSequence et rend les feuilles mesurables."""
    out = []
    for c in item.get("ContentSequence", []):
        name = ""
        if "ConceptNameCodeSequence" in c:
            name = c.ConceptNameCodeSequence[0].get("CodeMeaning", "")
        here = f"{path} / {name}".strip(" /")
        vt = c.get("ValueType")
        if vt == "NUM":
            try:
                m = c.MeasuredValueSequence[0]
                unit = m.MeasurementUnitsCodeSequence[0].get("CodeValue", "")
                out.append({"concept": here, "value": float(m.NumericValue), "unit": unit})
            except Exception:
                pass
        elif vt == "TEXT":
            out.append({"concept": here, "value": c.get("TextValue"), "unit": ""})
        elif vt == "CODE":
            out.append({"concept": here,
                        "value": c.ConceptCodeSequence[0].get("CodeMeaning"), "unit": ""})
        out += walk_sr(c, here)
    return out

def intranet_connect(username, password, url):
    USERNAME = username
    PASSWORD = password

    client = Client(
        url,
        auth=(USERNAME, PASSWORD)
    )
    return client

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

def show_oct(cube, patient_dir, id_p, id_v, ps):
    
    n = cube.shape[0]
    idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    idx = sorted(set(idx))

    # ps = pixel_spacing(ds)
    asp = (ps[0] / ps[1]) if ps else "auto"

    fig, axes = plt.subplots(
        len(idx), 1,
        figsize=(12, 4 * len(idx))
    )

    for ax, i in zip(np.atleast_1d(axes), idx):
        ax.imshow(
            cube[i], # autoscale(cube[i])
            cmap="gray",
            aspect=asp
        )
        ax.set_title(f"B-scan {i}/{n-1}")
        ax.axis("off")

    fig.suptitle(
        # f"OPT - {ds.get('ImageLaterality', '?')} "
        f"- {cube.shape} "
        f"range [{cube.min():.0f}, {cube.max():.0f}] "
        f"spacing={ps} "
    )

    fig.savefig(
        os.path.join(patient_dir, f"OPT_{id_p}_{id_v}_bis.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)