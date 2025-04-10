import os

root_dir = "/home/eyakub/scratch/CEASC_replicate"

with open(os.path.join(root_dir, "UAV-benchmark-M", f"valset_meta.csv"), 'r') as file:
    imgs = file.readlines()

print(imgs[0].split('\\')[0])
