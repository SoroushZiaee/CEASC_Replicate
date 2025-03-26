import torch
from PIL import Image

from torch.utils.data import Dataset
import os
import numpy as np


class VisDroneDataset(Dataset):
    def __init__(self, root_dir, split: str = "train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.categories = {
            1: "pedestrian",
            2: "person",
            3: "bicycle",
            4: "car",
            5: "van",
            6: "truck",
            7: "tricycle",
            8: "awning-tricycle",
            9: "bus",
            10: "motor",
        }

        if self.split == "train":
            self.img_dir = os.path.join(
                self.root_dir, "VisDrone2019-DET-train", "images"
            )
            self.anno_dir = os.path.join(
                self.root_dir, "VisDrone2019-DET-train", "annotations"
            )

        elif self.split == "val":
            self.img_dir = os.path.join(self.root_dir, "VisDrone2019-DET-val", "images")
            self.anno_dir = os.path.join(
                self.root_dir, "VisDrone2019-DET-val", "annotations"
            )

        elif self.split == "test":
            self.img_dir = os.path.join(
                self.root_dir, "VisDrone2019-DET-test-challenge", "images"
            )
            self.anno_dir = os.path.join(
                self.root_dir, "VisDrone2019-DET-test-challenge", "annotations"
            )

        else:
            raise ValueError(f"Invalid split: {split}")

        self.imgs = [f for f in os.listdir(self.img_dir) if f.endswith(".jpg")]
        self.imgs.sort()

    def __len__(self):
        return len(self.imgs)

    def parse_annotation(self, anno_path):
        boxes = []
        labels = []
        visibilities = []
        truncations = []
        occlusions = []

        # Read annotation file
        with open(anno_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    print(f"ignore this line: {line}")
                    continue

                x, y, width, height, visibility, category_id, truncation, occlusion = (
                    map(int, parts[:8])
                )

                # Skip objects with category_id of 0 or 11 (ignored regions)
                if category_id == 0 or category_id == 11:
                    print(f"category: {category_id}")
                    continue

                # Skip objects that are invisible or uncertain (optional)
                # if visibility == 0 or visibility == 2:
                #     continue

                boxes.append(
                    [x, y, x + width, y + height]
                )  # Convert to [x1, y1, x2, y2] format
                labels.append(category_id)
                visibilities.append(visibility)
                truncations.append(truncation)
                occlusions.append(occlusion)

        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        visibilities = np.array(visibilities, dtype=np.int64)
        truncations = np.array(truncations, dtype=np.int64)
        occlusions = np.array(occlusions, dtype=np.int64)

        return {
            "boxes": boxes,
            "labels": labels,
            "visibilities": visibilities,
            "truncations": truncations,
            "occlusions": occlusions,
        }

    def __getitem__(self, idx):
        """
        Get data sample by index

        Args:
            idx (int): Index

        Returns:
            Dictionary containing image and annotation information
        """
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size

        # Load annotations if available
        if self.anno_dir is not None:
            anno_name = img_name.replace(".jpg", ".txt").replace(".png", ".txt")
            anno_path = os.path.join(self.anno_dir, anno_name)

            if os.path.exists(anno_path):
                targets = self.parse_annotation(anno_path)
            else:
                # Empty targets if annotation file doesn't exist
                targets = {
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "labels": np.zeros((0,), dtype=np.int64),
                    "visibilities": np.zeros((0,), dtype=np.int64),
                    "truncations": np.zeros((0,), dtype=np.int64),
                    "occlusions": np.zeros((0,), dtype=np.int64),
                }
        else:
            # For test set, create empty targets
            targets = {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
                "visibilities": np.zeros((0,), dtype=np.int64),
                "truncations": np.zeros((0,), dtype=np.int64),
                "occlusions": np.zeros((0,), dtype=np.int64),
            }

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert targets to tensors
        boxes = torch.as_tensor(targets["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(targets["labels"], dtype=torch.int64)
        visibilities = torch.as_tensor(targets["visibilities"], dtype=torch.int64)
        truncations = torch.as_tensor(targets["truncations"], dtype=torch.int64)
        occlusions = torch.as_tensor(targets["occlusions"], dtype=torch.int64)

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "visibilities": visibilities,
            "truncations": truncations,
            "occlusions": occlusions,
            "image_id": torch.tensor([idx]),
            "img_name": img_name,
            "orig_size": torch.as_tensor([orig_height, orig_width]),
        }

    def visualize_item(self, idx, figsize=(10, 10)):
        """
        Visualize an image with its annotations

        Args:
            idx (int): Index of the item to visualize
            figsize (tuple): Figure size
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.colors import to_rgba

        # Get the sample
        sample = self[idx]

        # Convert tensor to numpy for visualization
        if isinstance(sample["image"], torch.Tensor):
            # If image is a tensor (transformed), convert back to numpy
            img = sample["image"].permute(1, 2, 0).numpy()
            # Unnormalize if normalized
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
        else:
            # If image is PIL, convert to numpy
            img = np.array(sample["image"]) / 255.0

        # Get the boxes and labels
        boxes = sample["boxes"].numpy()
        labels = sample["labels"].numpy()

        # Create figure and axis
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(img)

        # Define colors for different categories (you can customize these)
        colors = [
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "orange",
            "cyan",
            "magenta",
            "brown",
            "pink",
        ]

        # Plot bounding boxes
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # Get color based on category
            color = colors[(label - 1) % len(colors)]

            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height, linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

            # Add label text
            category_name = self.categories.get(label, f"Unknown ({label})")
            ax.text(
                x1,
                y1 - 5,
                category_name,
                color="white",
                fontsize=8,
                bbox=dict(facecolor=color, alpha=0.7, pad=1),
            )

        plt.title(f"Image: {sample['img_name']} - {len(boxes)} objects")
        plt.axis("off")
        plt.tight_layout()

        # save instead of show
        plt.savefig("test.png")
        plt.close()
