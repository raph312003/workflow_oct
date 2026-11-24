import os
import pandas as pd
from sklearn.model_selection import train_test_split


def create_manifests(inv, output_dir="manifests/OCT_HRA"):

    os.makedirs(output_dir, exist_ok=True)

    # 1. Split au niveau patient

    patients = inv["patient"].dropna().unique()

    train_patients, test_patients = train_test_split(
        patients,
        test_size=0.15,
        random_state=42
    )

    train_patients, val_patients = train_test_split(
        train_patients,
        test_size=0.15 / 0.85,
        random_state=42
    )

    # Association patient -> split
    patient_splits = pd.DataFrame({
        "patient": list(train_patients)
                    + list(val_patients)
                    + list(test_patients),

        "split": (
            ["train"] * len(train_patients)
            + ["val"] * len(val_patients)
            + ["test"] * len(test_patients)
        )
    })

    # 2. Ajouter le split à chaque acquisition
    manifest = inv.merge(
        patient_splits,
        on="patient",
        how="left",
        validate="many_to_one"
    )

    # 3. Sauvegarder les CSV

    manifest.to_csv(
        os.path.join(output_dir, "OCT_HRA.csv"),
        index=False
    )

    patient_splits.to_csv(
        os.path.join(output_dir, "patient_splits.csv"),
        index=False
    )

    # 4. Sauvegarder les JSONL

    for split in ["train", "val", "test"]:

        subset = manifest[manifest["split"] == split]

        subset.to_json(
            os.path.join(
                output_dir,
                f"OCT_HRA_{split}.jsonl"
            ),
            orient="records",
            lines=True,
            force_ascii=False
        )

    return manifest, patient_splits