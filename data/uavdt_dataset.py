from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

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

        with open(os.path.join(self.root_dir, "UAV-benchmark-M-train", f"{split}set_meta.csv"), 'r') as file:
            self.imgs = file.readlines() # get the list of imgs from the csv file 
        file.close()

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
    
    def __len__(self):
        return len(self.imgs)
    
    def parse_annotation(self):
        boxes = []
        labels = []

        # Get the annotations from the toolkit gt files 


    def __getitem__(self, idx):
        """
        Get data sample by index

        Args:
            idx (int): Index

        Returns:
            Dictionary containing image and annotation information
        """
        img_path = self.imgs[idx].split('\\')[0] # get the path without the \n character

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
            targets = self.parse_annotation()