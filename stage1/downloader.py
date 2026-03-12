import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Traffic sign"],
    max_samples=1500,  # do eksperymentów
)

dataset_val = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=["Traffic sign"],
    max_samples=200,  # do eksperymentów
)