import math
import numpy as np
import pandas as pd
import sys
import torch
from tqdm import tqdm, notebook



def train_one_epoch(model, optimizer, loader, device, epoch, experimentNum):
    model.to(device)
    model.train()

    all_losses = []
    all_losses_dict = []
    for images, targets in tqdm(loader, desc=f"Epoch {epoch}"):
        images = list(image.to(device) for image in images)
        targets = [{k:v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k:v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, training stopped.") # training stop if loss = inf
            print(loss_dict)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
    all_losses_dict = pd.DataFrame(all_losses_dict)
    print("Epoch {}, lr:{:.6f}, loss:{:.6f}, loss_classifier:{:.6f}, loss_box:{:.6f}, loss_rpn_box:{:.6f}, loss_object:{:.6f}".format(
        epoch, 
        optimizer.param_groups[0]['lr'], 
        np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))

    # checkpoints
    if epoch % 25 == 0:
        name = f"experiments/exp_{experimentNum}_{epoch}_{np.mean(all_losses):.4f}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'results' : all_losses_dict
        }, name)