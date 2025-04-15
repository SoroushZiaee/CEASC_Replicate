# script for making a meta file for UAVDT dataset 

# this meta file will provide the paths to each image for indexing by the dataloader 

import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description=
                                 "build UAVDT meta files containing paths to all images in  train, val, and test sets")

parser.add_argument("--root_dir", help="the directory where your UAV-benchmark-M movie files are located")
args = parser.parse_args()

# define the testset based on insights from https://github.com/PuAnysh/UFPMP-Det, the recommended repository for
# downloading the dataset

testset = ['M0203', 'M0205', 'M0208', 'M0209', 'M0403', 'M0601', 'M0602', 'M0606',
            'M0701', 'M0801', 'M0802', 'M1001',
            'M1004', 'M1007', 'M1009', 'M1101', 'M1301', 'M1302', 'M1303', 'M1401']

# define a separate val set that is 10% of the subset -- this is our own val set as the paper and dataset authors did not provide 
# any information on a val set
val_size = int(len(testset)*0.1)
np.random.seed(42) # seed the random number generator to make sure that we get same val set for all who run this code
val_idx = np.random.choice(len(testset),size=val_size,replace=False)
valset = [testset[i] for i in val_idx]

# define the trainset based on what is not in the testset
trainset = []
root_dir = args.root_dir
movies = os.listdir(root_dir) 
movies.sort()
for movie in movies:
    if movie != "GT" and movie not in testset and movie not in valset:
        trainset.append(movie)

set_dict = {"valset": valset,
            "testset": testset,
            "trainset": trainset}

# set_dict = {"valset": valset,} # for testing with only valset

# make the meta files for each set
for split, movie_list in set_dict.items():
    csv_path = os.path.join(root_dir,f"{split}_meta.csv")
    with open(csv_path,'w') as meta_file:
        for movie in movie_list:
            frame_list = os.listdir(os.path.join(root_dir,movie))
            frame_list.sort()
            for frame in frame_list:
                meta_file.write(f"{os.path.join(root_dir,movie,frame)}\n")
            print(f"{movie} done!")
    print(f"{split} done!")
    meta_file.close()



