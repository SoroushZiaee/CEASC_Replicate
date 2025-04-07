import sys
import os

# add parent directory, it should add parent of parent
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
from torch import nn, optim

from torchvision import transforms
from tqdm import tqdm

from models import Res18FPNCEASC  # Adjust as needed
from utils.dataset import get_dataset
from utils.losses import Lnorm, Lamm  # Adjust as needed


def safe_shape(x):
    if isinstance(x, torch.Tensor):
        return x.shape
    elif isinstance(x, (list, tuple)):
        return [safe_shape(e) for e in x]
    return type(x)


def train(config):
    # Unpack config
    root_dir = config["root_dir"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    num_epochs = config["num_epochs"]
    learning_rate = config["lr"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and loader
    dataloader = get_dataset(
        root_dir=root_dir,
        split="train",
        transform=None,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Model
    model = Res18FPNCEASC(config_path=config["config_path"], num_classes=10)
    model.to(device)
    model.train()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

    # Losses
    l_norm = Lnorm()
    l_amm = Lamm()

    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            images = batch["image"].to(device)
            targets = {
                "boxes": batch["boxes"],
                "labels": batch["labels"],
                "image_id": batch["image_id"],
                "orig_size": batch["orig_size"],
            }
            print("\n🔍 Inspecting `targets` structure:")
            for i in range(len(targets["boxes"])):
                print(f"--- Sample {i} ---")
                print(f"Image ID:         {targets['image_id'][i]}")
                print(f"Original Size:    {targets['orig_size'][i]}")
                print(f"Boxes shape:      {targets['boxes'][i].shape}")   # [N_i, 4]
                print(f"Labels shape:     {targets['labels'][i].shape}") # [N_i]
                print(f"Boxes:            {targets['boxes'][i]}")
                print(f"Labels:           {targets['labels'][i]}")

            # Forward pass
            outputs = model(images, stage="train")
            (
                cls_outs,
                reg_outs,
                cls_soft_mask_outs,
                reg_soft_mask_outs,
                sparse_cls_feats_outs,
                sparse_reg_feats_outs,
                dense_cls_feats_outs,
                dense_reg_feats_outs,
                feats,
            ) = outputs

            print("\n🔍 Output shapes from model:")
            for i in range(len(cls_outs)):
                print(f"--- FPN Level {i} ---")
                print(f"cls_outs[{i}]:              {safe_shape(cls_outs[i])}")
                print(f"reg_outs[{i}]:              {safe_shape(reg_outs[i])}")
                print(
                    f"cls_soft_mask_outs[{i}]:    {safe_shape(cls_soft_mask_outs[i])}"
                )
                print(
                    f"reg_soft_mask_outs[{i}]:    {safe_shape(reg_soft_mask_outs[i])}"
                )
                print(
                    f"sparse_cls_feats[{i}]:      {safe_shape(sparse_cls_feats_outs[i])}"
                )
                print(
                    f"sparse_reg_feats[{i}]:      {safe_shape(sparse_reg_feats_outs[i])}"
                )
                print(
                    f"dense_cls_feats[{i}]:       {safe_shape(dense_cls_feats_outs[i])}"
                )
                print(
                    f"dense_reg_feats[{i}]:       {safe_shape(dense_reg_feats_outs[i])}"
                )
                print(f"feats[{i}]:                 {safe_shape(feats[i])}")

            loss_norm = l_norm(
                sparse_cls_feats_outs, cls_soft_mask_outs, dense_cls_feats_outs
            )

            loss_amm = l_amm(
                targets["boxes"], reg_soft_mask_outs
            ) # used the soft masks in this version, might be incorrect 

            print(f"Loss Norm: {loss_norm.item()}")
            print(f"Loss AMM: {loss_amm.item()}")

            # Calculate loss — you must define this to fit CEASC
            # loss = ceasc_loss(outputs, targets)  # Custom function required
            # loss = 0.0
            # # make the loss a torch tensor
            # loss = torch.tensor(loss, dtype=torch.float32).to(device)

            # optimizer.zero_grad()
            # # loss.backward()
            # optimizer.step()

            # # total_loss += loss.item()
            # progress_bar.set_postfix(loss=loss.item())
            break

        print(f"Epoch {epoch + 1}: Total Loss = {total_loss:.4f}")

    # Save model
    # torch.save(model.state_dict(), "weights/ceasc_final.pth")


def evaluate():
    pass


def main():
    mode = "train"  # Change to "eval" or "test" as needed

    config = {
        "root_dir": "/home/soroush1/scratch/eecs_project",
        "batch_size": 4,
        "num_workers": 4,
        "num_epochs": 1,
        "lr": 1e-3,
        "config_path": "configs/resnet18_fpn_feature_extractor.py",
    }

    if mode == "train":
        train(config)
    elif mode == "eval":
        evaluate(config)  # Optional to implement
    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    print(f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

    main()
