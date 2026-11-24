import torch
import os
import pandas as pd
import numpy as np
import torch

from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, Subset
import pydicom

from torch.utils.data import Subset
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from webdav4.client import Client
from src.oct_pipeline.loading import load_opt, read_dicom_webdav, to_float



# Multiple instance learning
# Un patient = in bag -> label pour chaque bag
# chaque B-scan est une instance -> Modèle apprend à exploiter les b-scans les plus informatifs
# Pipeline :

# 1) Volume OCT d'un patient
# 2) Retfound encoder (Extraction des features de chaque b-scans)
# 3) 100 embeddings * 1024
# 4) Aggregation / Attention / Pooling
# 5) Classification head -> pathologie patient

class OCTDataset(Dataset):

    def __init__(self, df, client, transform=None):
        self.df = df.reset_index(drop=True)
        self.client = client
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = row["path"]
        label = int(row["label"])

        # Lecture du DICOM via WebDAV
        ds = read_dicom_webdav(self.client, path)

        # Extraction des pixels
        arr = ds.pixel_array

        # Gestion des dimensions
        if arr.ndim == 3:
            slice = arr.shape[0]//2
            arr = arr[slice]  # Sélection de la première frame > 1

        # Conversion en image PIL
        arr = to_float(ds, arr)

        # Conversion vers une image compatible avec les transforms ImageNet
        arr = np.asarray(arr)

        # Normalisation et conversion en uint8
        if arr.dtype != np.uint8:
            arr = np.nan_to_num(arr)

            arr_min = arr.min()
            arr_max = arr.max()

            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min)
            else:
                arr = np.zeros_like(arr)

            arr = (arr * 255).astype(np.uint8)

        image = Image.fromarray(arr).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train == 'train':
        return create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )

    # eval transform
    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    size = int(args.input_size / crop_pct)
    t = [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(t)

def build_dataset(is_train, args):

    USERNAME = "6fef28b0-6242-1041-9be8-290d06645fc2"
    PASSWORD = "Ecorans3105@"

    url = "https://www.intranet.disp-lab.fr/remote.php/dav/files/6fef28b0-6242-1041-9be8-290d06645fc2/"

    client = Client(
        url,
        auth=(USERNAME, PASSWORD)
    )

    transform = build_transform(is_train, args)

    inv = pd.read_csv(args.data_path)

    patient_data_opt = inv[inv["split"] == is_train].reset(drop=True)

    dataset = OCTDataset(
        df=patient_data_opt,
        client=client,
        transform=transform
    )

    if is_train == 'train':
        ratio = float(getattr(args, "dataratio", 1.0))
        seed = int(getattr(args, "seed", 0))
        stratified = bool(getattr(args, "stratified", False))

        if 0.0 < ratio < 1.0:
            if stratified:
                idx = dataset.df["label"].tolist()
            else:
                # simple uniform subsample with torch.Generator for reproducibility
                g = torch.Generator().manual_seed(seed)
                n = len(dataset)
                k = max(1, int(n * ratio))
                idx = torch.randperm(n, generator=g)[:k].tolist()
            dataset = Subset(dataset, idx)

    return dataset