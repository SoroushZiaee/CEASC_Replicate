from torch.utils.data import Dataset
from torchvision import transforms

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

        if split == "train":
            self.img_dir = os.path.join(self.root_dir, 'UAV-benchmark-M-train')
            self.anno_path = os.path.join(self.root_dir, 'UAV-benchmark-M-train', 'train_annos.json') # make sure these paths work
        
        else:
            self.img_dir = os.path.join(self.root_dir, 'UAV-benchmark-M-test')
            self.anno_path = os.path.join(self.root_dir, 'UAV-benchmark-M-test', 'test_annos.json')

        self.imgs = [f for f in os.listdir(self.img_dir) if f.endswith(".jpg")]
        self.imgs.sort()

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
    
    def __len__(self):
        return len(self.imgs)
    
    def parse_annotation(self, anno_path):
        # start here once the annotations have been made