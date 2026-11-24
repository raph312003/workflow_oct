import torch
import os
import pandas as pd
import numpy as np
import torch

from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, Subset
import pydicom
from torchvision import transforms

from src.oct_pipeline.loading import load_opt, read_dicom_webdav, to_float, pixel_spacing, autoscale

class OCTDataset(Dataset):

    def __init__(self, df, client, size, transform=None):
        self.df = df.reset_index(drop=True)
        self.client = client
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = row["path"]
        bag_label = int(row["label"])

        # Lecture du volume OCT
        ds = read_dicom_webdav(self.client, path)
        arr = ds.pixel_array

        mid_slice = arr.shape[0]//2
        s_min = mid_slice-5
        s_max = mid_slice+5

        arr = arr[s_min:s_max,...]
        # print("arr_shape", arr.shape)

        # pixel spacing
        ps = pixel_spacing(ds)

        # Gestion du cas où le DICOM contient un seul B-scan
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        # Conversion des intensités
        arr = to_float(ds, arr)
        arr = np.nan_to_num(arr)

        images = []

        for scan in arr:

            #Normalisation individuelle
            scan_min = scan.min()
            scan_max = scan.max()

            if scan_max > scan_min:
                scan = (scan - scan_min) / (scan_max - scan_min)
            else:
                scan = np.zeros_like(scan)

            scan = (scan * 255).astype(np.uint8)
            # scan = autoscale(scan)

            scan = np.ascontiguousarray(scan, dtype=np.uint8)

            image = Image.fromarray(scan, mode="L")
            # image = image.convert("RGB")

            image = image.resize(self.size, Image.Resampling.BILINEAR)

            if self.transform is not None:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)

            images.append(image)

        # [N, C, H, W]
        images = torch.stack(images)

        return {
            "images": images,
            "bag_label": bag_label,
            "patient_id": row["patient"],
            "volume_id": idx,
            "scan_indices": torch.arange(len(images)),
            "ps": ps
        }



def collate_bags(batch):

    all_images = []
    all_bag_labels = []
    all_bag_ids = []
    all_scan_indices = []

    for bag_idx, sample in enumerate(batch):

        images = sample["images"]
        n_scans = len(images)

        all_images.append(images)

        all_bag_labels.append(sample["bag_label"])

        all_bag_ids.append(
            torch.full((n_scans,), bag_idx, dtype=torch.long)
        )

        all_scan_indices.append(sample["scan_indices"])


    return {
        # exemple :
        # A	3 1
        # B	2 0
        # C	4 1
        # mettre tous les bscans ensemble au sein du même batch
        "images": torch.cat(all_images, dim=0), # [9, 3, 224, 224] Les 9 B-scans sont concaténés dans un seul tenseur pour 3 volumes
        "bag_labels": torch.tensor(all_bag_labels), # tensor([1, 0, 1]) bag labels
        "bag_ids": torch.cat(all_bag_ids), # tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        "scan_indices": torch.cat(all_scan_indices), # tensor([0, 1, 2, 0, 1, 0, 1, 2, 3]) scan indice
        "patient_ids": [sample["patient_id"] for sample in batch], # ["12332", "133133", "13313"]
        "volume_ids": [sample["volume_id"] for sample in batch], # # ["12332", "133133", "13313"]
        "pixel_spacing": [sample["ps"] for sample in batch]
    }