from torch.utils.data import DataLoader
from data.visdrone_dataset import VisDroneDataset

from torchvision import transforms
import torch


def detection_collate(batch):
    collated = {}

    collated["image"] = torch.stack([item["image"] for item in batch], dim=0)
    collated["boxes"] = [item["boxes"] for item in batch]
    collated["labels"] = [item["labels"] for item in batch]
    collated["visibilities"] = [item["visibilities"] for item in batch]
    collated["truncations"] = [item["truncations"] for item in batch]
    collated["occlusions"] = [item["occlusions"] for item in batch]
    collated["image_id"] = [item["image_id"] for item in batch]
    collated["img_name"] = [item["img_name"] for item in batch]
    collated["orig_size"] = [item["orig_size"] for item in batch]

    return collated


def get_dataset(
    root_dir: str,
    split: str,
    transform=None,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
):
    """
    Get the dataset based on the split.
    :param root_dir: Root directory of the dataset.
    :param split: Split of the dataset (train, val, test).
    :return: Dataloader object.
    """
    transform = transformation(split) if transform is None else transform
    dataset = VisDroneDataset(root_dir=root_dir, split=split, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate,
    )

    return dataloader


def transformation(split: str):
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((800, 1333)),  # fixed size (height, width)
                transforms.ToTensor(),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((800, 1333)),  # fixed size (height, width)
            transforms.ToTensor(),
        ]
    )
