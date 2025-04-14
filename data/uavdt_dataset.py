from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch

import os 


class UAVDTDataset(Dataset):
    def __init__(self, root_dir, split: str = "train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.categories = {
            0: "car",
            1: "truck",
            2: "bus" 
        }

        meta_path = os.path.join(self.root_dir, "UAV-benchmark-M", f"{split}set_meta.csv")

        with open(meta_path, 'r') as meta_file:
            self.imgs = meta_file.readlines() # get the list of imgs from the csv file 
        meta_file.close()

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
    
    def __len__(self):
        return len(self.imgs)
    
    def parse_annotation(self, movie_name, img_idx):
        boxes = []
        labels = []

        # Get the annotations from the toolkit gt files -- this function was largely inherited from UAVDT2COCO.py logic
        anno_path = os.path.join(self.root_dir, "UAV-benchmark-MOTD_v1.0", "GT", f"{movie_name}_gt_whole.txt")
        with open(anno_path, "r") as anno_file:
            annos = anno_file.readlines()
            our_img_annos = [a for a in annos if int(a.split(',')[0]) == img_idx] # get all the annotation lines that correspond to our image
            for anno in our_img_annos:
                items = [int(i) for i in anno.split(',')] # get all the individual annotations
                [frame_index, target_id, x, y, width, height, out_of_view, occlusion,
                 object_category] = items
                boxes.append([x, y, x + width, y + height]) # convert to [x1, y1, x2, y2] format
                labels.append(object_category - 1) # the categories are coded as one less than the true numerical value in the dataset

        anno_file.close()

        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        return {
            "boxes": boxes,
            "labels": labels}



    def __getitem__(self, idx):
        """
        Get data sample by index

        Args:
            idx (int): Index

        Returns:
            Dictionary containing image and annotation information
        """
        img_path = self.imgs[idx].strip()

        # Load the image
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size

        # Load annotations if available - if test set them to 0
        if self.split == "test":
            targets = {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
            }
            # figure out if we are also trying to predict other things from this dataset -- look through the paper again 
        else:
            movie_name = img_path.split('/')[6]
            img_idx = int(img_path.split('/')[-1][3:9])
            targets = self.parse_annotation(movie_name,img_idx)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert targets to tensors
        boxes = torch.as_tensor(targets["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(targets["labels"], dtype=torch.int64)

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "img_name": img_path.split('/')[-2:],
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
            color = colors[label]

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

