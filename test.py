# import os

# root_dir = "/home/eyakub/scratch/CEASC_replicate"

# with open(os.path.join(root_dir, "UAV-benchmark-M", f"valset_meta.csv"), 'r') as file:
#     imgs = file.readlines()

# file.close()

# splits = imgs[0].split('\\')

# image_path = splits[0]


# print(image_path)
# print(image_path.split('/')[-1][3:9])
# print(image_path.split('/'))
# print(int(image_path.split('/')[-1][:-1][3:9]))

# import torch
# from data import uavdt_dataset

# ds = uavdt_dataset.UAVDTDataset("/home/eyakub/scratch/CEASC_replicate", "train")

# print(ds.__len__())

# out_dict = ds[1]
# print(out_dict["boxes"])
# print(out_dict["labels"])
# print(f"number of boxes: {len(out_dict['boxes'])} | number of labels: {len(out_dict['labels'])}")

# ds.visualize_item(1)

# print("loaded!")

# import torch
# from mmengine.config import Config
# from mmdet.registry import MODELS

# print(f"perfect")


# import torch
# from utils.losses import Lnorm, Lamm

# l_amm = Lamm()

# print("Lamm base classes:", Lamm.__bases__)
# print("l_amm type:", type(l_amm))
# print("isinstance l_amm of nn.Module:", isinstance(l_amm, torch.nn.Module))
# print("dir(l_amm):", dir(l_amm))

import torch 
t = torch.rand(10,5)
print(torch.argmax(t,dim=1).shape)

