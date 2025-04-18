# the script for training GFL V1-CEASC on UAVDT

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

import sys
import os

# add parent directory, it should add parent of parent
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from tqdm import tqdm

from models import Res18FPNCEASC  # Adjust as needed
from utils.uavdt_dataloader import get_dataset
from utils.losses import Lnorm, Lamm, DetectionLoss  # Adjust as needed

def safe_shape(x):
    if isinstance(x, torch.Tensor):
        return x.shape
    elif isinstance(x, (list, tuple)):
        return [safe_shape(e) for e in x]
    return type(x)

# get the setup 
mode = "train"  # Change to "eval" or "test" as needed

config = {
    "root_dir": "/home/eyakub/scratch/CEASC_replicate",
    "batch_size": 4,
    "num_workers": 4,
    "num_epochs": 6,
    "lr": 1e-1,
    "alpha": 1,
    "beta": 10,
    "config_path": "../configs/resnet18_fpn_feature_extractor.py",
}

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True) # for debugging 

    writer = SummaryWriter()

    # Unpack config
    root_dir = config["root_dir"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    num_epochs = config["num_epochs"]
    learning_rate = config["lr"]
    alpha = config["alpha"]
    beta = config["beta"]
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
    model = Res18FPNCEASC(config_path=config["config_path"], num_classes=3)
    model.to(device)
    model.train()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) 

    def lr_lambda(it):
        '''
        function to control the learning rate - 10 x decrease at epochs 5 and 6
        ''' 
        if it == 4 or it ==5:
            return 0.1 # decrease the lr by a factor of 10 at epochs 12 and 15

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Losses
    l_amm = Lamm()
    l_norm = Lnorm()
    l_det = DetectionLoss(num_classes=3)

    global_it = 0

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

            optimizer.zero_grad()

            with autocast():
    
                # Forward pass
                outputs = model(images, stage="train")
                (
                    cls_outs,
                    reg_outs,
                    soft_mask_outs,
                    sparse_cls_feats_outs,
                    sparse_reg_feats_outs,
                    dense_cls_feats_outs,
                    dense_reg_feats_outs,
                    feats,
                    anchors,
                ) = outputs

                loss_amm = l_amm(
                    soft_mask_outs, targets["boxes"], im_dimx=1024, im_dimy=540
                    ) 
                
                loss_norm = l_norm(
                    sparse_cls_feats_outs, soft_mask_outs, dense_cls_feats_outs
                    )
                
                loss_det = l_det(
                    cls_outs, reg_outs, anchors, targets
                    )

                if global_it % 1000 == 0:
                    writer.add_scalar("AMM Loss/train", loss_amm.item(), global_it)
                    writer.add_scalar("Norm Loss/train", loss_norm.item(), global_it)
                    writer.add_scalar("Det Loss/train", loss_det["total_loss"].item(), global_it)

                # sum the losses 
                loss_overall = loss_det["total_loss"] + alpha*loss_norm + beta*loss_amm

            scaler.scale(loss_overall).backward()
    
            scaler.step(optimizer)

            scaler.update()

            global_it += 1

            del loss_amm, loss_norm, loss_det, loss_overall, outputs

            torch.cuda.empty_cache()
        
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(),f'{root_dir}/uavdt-mmdet-epoch-{epoch}.pth')

        scheduler.step()

    writer.close()