import os
import sys

# add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.visdrone_dataset import VisDroneDataset


def test_visdrone_dataset():
    # Test VisDroneDataset
    dataset = VisDroneDataset(
        root_dir="/home/soroush1/scratch/eecs_project",
        split="train",
    )

    # Test dataset
    print("len(dataset):", len(dataset))
    print("dataset[0]:", dataset[0])

    dataset.visualize_item(0)

    val_dataset = VisDroneDataset(
        root_dir="/home/soroush1/scratch/eecs_project",
        split="val",
    )
    print("len(val_dataset):", len(val_dataset))
    print("val_dataset[0]:", val_dataset[0])

    val_dataset.visualize_item(0)


def main():
    test_visdrone_dataset()


if __name__ == "__main__":
    main()
