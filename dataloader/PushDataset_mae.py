from __future__ import print_function, division
from torch.utils.data import Dataset
import glob
import numpy as np
from utils import jsonreader
from torchvision import transforms
from PIL import Image, ImageFile
import torch
from scipy.spatial.transform import Rotation as R
import random

class PushDatasetMAE(Dataset):
  """Push images dataset (Playful)"""
  def __init__(self, dataset, params, transform=None):
    if dataset == "train":
      root_dir = params['train_dir']
    elif dataset == "val":
      root_dir = params['val_dir']
    elif dataset == "test":
      root_dir = params['test_dir']

    self.data_size = params['data_size']
    self.dataset = dataset
    folders = sorted(glob.glob(root_dir + "/*"), reverse=True)

    self.transform = transforms.Compose([
          transforms.Resize(240),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
        ])

    total_imgs = 0
    total_runs = len(folders)

    self.train = []
    counter = 0
    used_runs = 0
    target_runs = round(total_runs * self.data_size)
    for run in folders:
      counter += 1
      if self.dataset == "train":
        if used_runs > target_runs:
          continue

      used_runs += 1

      the_old_imgs, translations, _ = jsonreader.get_vals(run)
      imgs = the_old_imgs
      if len(the_old_imgs) == 0 or len(translations) == 0:
        continue

      scale_factor = np.max(np.abs(translations))
      if scale_factor >= 2.5 or scale_factor < 0.05:
        print("Max norm(action) is not normal for run " + run + ": ", scale_factor)
      scaled_trans = translations / scale_factor

      num_imgs = len(imgs)
      num_valid = num_imgs
      total_imgs += num_valid

      for i in range(num_valid):
        img_t = imgs[i]
        self.train.append(([img_t],scaled_trans[i], run, 0))

    print("num used runs:", used_runs)
    print("num collected images: ", total_imgs)


  def __len__(self):
    return len(self.train)

  def __getitem__(self, idx):
    img_names_t, actions, run, mirrored = self.train[idx]
    img_t = img_names_t[0]

    try:
      imgs_t = torch.unsqueeze(self.transform(Image.open(img_t)), 0)
    except:
      img = Image.new("RGB", (224, 224), (0, 0, 0))
      imgs_t = torch.unsqueeze(self.transform(img), 0)

    # returning 6 values now
    return img_names_t, imgs_t, actions, run, []