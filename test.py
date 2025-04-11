# import os

# root_dir = "/home/eyakub/scratch/CEASC_replicate"

# with open(os.path.join(root_dir, "UAV-benchmark-M", f"valset_meta.csv"), 'r') as file:
#     imgs = file.readlines()

# file.close()

# image_path = imgs[5].split('\\')[0]


# print(image_path)
# print(image_path.split('/'))
# print(int(image_path.split('/')[-1][:-1][3:9]))

import torch
from data import uavdt_dataset

ds = uavdt_dataset.UAVDTDataset("/home/eyakub/scratch/CEASC_replicate", "train")

print("loaded!")

