import argparse
import os
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
import wandb
from tqdm import tqdm

from dataloader.PushDataset_mae import PushDatasetMAE

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from transformers import ViTImageProcessor
from transformers import ViTMAEForPreTraining
from transformers import ViTFeatureExtractor
import torch
import numpy as np
import matplotlib.pyplot as plt

save_dir = "mae_checkpoint_6"
feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base")
imagenet_mean = np.array(feature_extractor.image_mean)
imagenet_std = np.array(feature_extractor.image_std)

def show_image(image, title='', ax=None):
    assert image.shape[2] == 3
    # ax.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    ax.imshow(image)
    ax.set_title(title, fontsize=16)
    ax.axis('off')

def visualize(pixel_values, model, epoch_num, save_dir="transformer-mae"):
    model.eval()
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)
    
    outputs = model(pixel_values)
    y = model.unpatchify(outputs.logits)
    y = torch.einsum('nchw->nhwc', y)
    
    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 * 3)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask)
    
    x = torch.einsum('nchw->nhwc', pixel_values)
    im_masked = x * (1 - mask)
    im_paste = x * (1 - mask) + y * mask

    x = x.detach().cpu()
    im_masked = im_masked.detach().cpu()
    y = y.detach().cpu()
    im_paste = im_paste.detach().cpu()

    fig, axes = plt.subplots(1, 4, figsize=(24, 24))
    show_image(x[0], "original", ax=axes[0])
    show_image(im_masked[0], "masked", ax=axes[1])
    show_image(y[0], "reconstruction", ax=axes[2])
    show_image(im_paste[0], "reconstruction + visible", ax=axes[3])

    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'epoch{epoch_num}.png'), bbox_inches='tight')
    plt.close(fig)
    
params = {
    'root_dir': "",
    'train_dir': "",
    'val_dir': "",
    'data_size': 1.0,
    'epochs': 100,
    'batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 0.05,
    'patch_size': 16,
    'img_size': 224,
    'model': 'vit_base_patch16_224',
    'mask_ratio': 0.75,
    'decoder_dim': 512,
    'decoder_depth': 6,
    'device': 'cuda:2',
    'save_dir': './checkpoints',
    'project': 'mae-pretraining',
    'run_name': 'mae-run',
    'num_workers': 8,
    'train_layer_num': 7
}
device = params["device"]

def make_dataloader(params, batch_size, num_workers, mode="train"):
    dataset = PushDatasetMAE(mode, params)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=="train"), num_workers=num_workers, drop_last=True)
    return loader

wandb.init(project=params["project"], name=params["run_name"], config=params)

train_loader = make_dataloader(params, params["batch_size"], params["num_workers"], mode="train")
val_loader = make_dataloader(params, params["batch_size"], params["num_workers"], mode="val")

processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")

model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
model.to(device) 

#############################################################################################################################
# freeze all
for name, param in model.named_parameters():
    param.requires_grad = False
# encoder layer 0,1 train
max_train_layers = min(params['train_layer_num'], len(model.vit.encoder.layer))
for idx in range(0, max_train_layers): # ~ 6 layer
    for name, param in model.vit.encoder.layer[idx].named_parameters():
        param.requires_grad = True
for name, param in model.named_parameters():
    print(name, param.requires_grad)
#############################################################################################################################

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_epochs = params["epochs"]
visualize_interval = 5

best_val_loss = float('inf')
for epoch in tqdm(range(num_epochs), desc="Epoch"):
    # Train -------------------------------------------------------------------------
    model.train()
    total_loss = 0.0
    
    for _, imgs_t, _, _, _ in tqdm(train_loader, desc="Training"):
        imgs_t = torch.stack([x.squeeze(0) for x in imgs_t], dim=0).to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=imgs_t)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
    
    # Validation -------------------------------------------------------------------------
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            _, imgs_t, _, _, _ = batch
            imgs_t = imgs_t.squeeze(1).to(device)
            outputs = model(pixel_values=imgs_t)
            loss = outputs.loss
            total_loss += loss.item()
            count += 1
    val_loss = total_loss / max(1, count)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained(f"./{save_dir}/transformer-mae-model")
        processor.save_pretrained(f"./{save_dir}/transformer-mae-processor")
        print(f"Saved best model at epoch {epoch} with val_loss={val_loss:.4f}")

    # viz each 5 epoch
    if (epoch + 1) % visualize_interval == 0:
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            sample_imgs = sample_batch[1]
            
            if isinstance(sample_imgs, list):
                sample_imgs = torch.stack([x.squeeze(0) for x in sample_imgs], dim=0)
            else:
                sample_imgs = sample_imgs.squeeze(1) if sample_imgs.ndim == 5 else sample_imgs

            sample = sample_imgs[:1].to(device)

            visualize(sample, model, epoch_num=epoch)
            
    wandb.log({"train_loss": avg_loss, "val_loss": val_loss, "epoch": epoch})
    
    model.save_pretrained(f"./{save_dir}/transformer-mae-model-{epoch}")
    processor.save_pretrained(f"./{save_dir}/transformer-mae-processor-{epoch}")

wandb.finish()
