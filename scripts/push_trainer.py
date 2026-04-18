import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch import optim
import torch.nn.functional as F
import dataloader.PushDataset as PushDataset
import utils.logger as logger
import utils.plot as plotting
from utils.earlystop import EarlyStop
from utils.utils import convert_6drep_to_mat, convert_3dmat_to_6d_rep


class PushTrainer:
  """
  Trainer class
  """

  def __init__(self, model, params):
    torch.manual_seed(0)
    np.random.seed(0)

    self.params = params
    self.device = torch.device('cuda:' + str(params['which_gpu']) if torch.cuda.is_available() else "cpu")
    self.model = model
    self.logger = logger.Logger(self.params['logdir'])
    self.batch_size = params['batch_size']
    self.epochs = params['epochs']
    self.rot = params['rot']
    self.model_name = params['model']
    self.history = params['history']
    self.mult = params['mult']
    self.l1 = params['l1']
    self.l2 = params['l2']
    self.l3 = params['l3']
    self.rad = params['rad']
    self.net = params['net_type']
    self.mse_loss = nn.MSELoss()
    self.scaled_mse_loss = nn.MSELoss(reduce=False)
    self.l1_loss = nn.L1Loss()
    self.dir_weight = 1 # 이것 잠시 조정 (원래는 1)
    self.mse_weight = 1 # CHANGE THIS

    self.trans_loss_type = params.get('trans_loss_type', 'mse')
    self.trans_weight = params.get('trans_weight', 1)

    self.train_set = PushDataset.PushDataset("train", params)
    self.num_train = len(self.train_set)
    self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                                                    sampler=None,
                                                    batch_sampler=None, num_workers=4,
                                                    pin_memory=True, drop_last=False, timeout=0,
                                                    worker_init_fn=None)

    self.val_set = PushDataset.PushDataset("val", params)
    self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=True, drop_last=False, timeout=0,
                                                  worker_init_fn=None)
    self.num_val = len(self.val_set)

    self.test_set = PushDataset.PushDataset("test", params)
    self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                                                   num_workers=1,
                                                   pin_memory=False, drop_last=False, timeout=0,
                                                   worker_init_fn=None)
    self.num_test = len(self.test_set)

    wandb.init(project="trashbot")
    wandb.save("*.pt")
    params['num_train_runs'] = self.num_train
    params['num_val_runs'] = self.num_val
    params['num_test_runs'] = self.num_test
    wandb.config.update(params)

    self.optimizer = optim.Adam(model.parameters(), lr=self.params['lr'])

  def train(self, logdir=None):
    wandb.watch(self.model)

    self.num_train_batches = np.ceil(self.num_train / self.batch_size)
    early_stopping = EarlyStop(patience=150)
    num_epochs = self.epochs
    self.model.to(self.device)
    for epoch in range(num_epochs):

      self.wandb_dict = {}
      ######### TRAIN ===============================================================================================
      self.model.train()
      self.run_loop(self.train_loader, "train", epoch, logdir, None)

      ######### VALIDATE ============================================================================================
      self.model.eval()
      with torch.no_grad():
        early_stop = self.run_loop(self.val_loader, "val", epoch, logdir, early_stopping)

      ######### TEST ============================================================================================
      self.model.eval()
      with torch.no_grad():
        self.run_loop(self.test_loader, "test", epoch, logdir, None)

      wandb.log(self.wandb_dict)

      if early_stop:
        print("Early stopping at epoch " + str(epoch))

  def run_loop(self, data_loader, current, epoch, logdir, early_stopping):

    epoch_error = []
    epoch_loss = []

    pos_loss = []
    pos_error = []

    ang_loss = []
    ang_error = []

    dir_loss = []

    for i_batch, (batch_img_names, batch_imgs, batch_actions, runs, aug) in enumerate(data_loader):
      # Debug: print shapes/dtypes/devices for the first batch to help trace CUDA errors
      if i_batch == 0:
        try:
          print(f"[DEBUG] batch_imgs: shape={getattr(batch_imgs,'shape',None)}, dtype={getattr(batch_imgs,'dtype',None)}, device={getattr(batch_imgs,'device',None)}")
          print(f"[DEBUG] batch_actions: shape={getattr(batch_actions,'shape',None)}, dtype={getattr(batch_actions,'dtype',None)}, device={getattr(batch_actions,'device',None)}")
        except Exception:
          pass
      batch_imgs = batch_imgs.to(self.device)
      # make sure batch_actions is moved to device (reassign .to result)
      batch_actions = batch_actions.to(self.device)

      # Debug: after moving to device
      if i_batch == 0:
        try:
          print(f"[DEBUG after .to()] batch_imgs device={getattr(batch_imgs,'device',None)}, batch_actions device={getattr(batch_actions,'device',None)}")
        except Exception:
          pass

      real_positions, _, real_angles_raw = self.get_label_act_pos(batch_actions)

      real_angles_raw = real_angles_raw.to(self.device)
      real_positions = real_positions.to(self.device)

      pred_pos = self.model.forward(batch_imgs)

      # Debug: inspect prediction tensor for shape/device/grad
      if i_batch == 0:
        try:
          print(f"[DEBUG] pred_pos: shape={getattr(pred_pos,'shape',None)}, dtype={getattr(pred_pos,'dtype',None)}, device={getattr(pred_pos,'device',None)}, requires_grad={getattr(pred_pos,'requires_grad',None)}")
        except Exception:
          pass

      total_loss, total_error, p_loss, d_loss = self.calculate_loss_error(
        pred_pos,
        real_positions,
      )
      if current == "train":
        self.optimizer.zero_grad()
        # 3. backward propagation
        total_loss.backward()

        # 4. weight optimization
        self.optimizer.step()

      epoch_loss.append(total_loss.item())
      epoch_error.append(total_error.item())

      pos_loss.append(p_loss.item())

      dir_loss.append(d_loss.item())


      if i_batch % 20 == 0 and current == "train":
        print('Epoch [{}/{}, Step [{}/{}], Loss: {:.4f}'
              .format(epoch, self.epochs, i_batch + 1, self.num_train_batches, total_loss.item()))

      # self.log_sometimes(epoch, i_batch, batch_img_names, batch_imgs, real_positions, real_angles_raw, pred_pos,
      #                    logdir, current)

    print("Epoch", epoch, current + " Error", np.mean(epoch_error))
    print("Epoch", epoch, current + " Loss", np.mean(epoch_loss))

    self.wandb_dict.update({
      current.lower().capitalize() + " Error": np.mean(epoch_error),
      current.lower().capitalize() + " Loss": np.mean(epoch_loss),
      current.lower().capitalize() + " Translation Loss": np.mean(pos_loss),
      current.lower().capitalize() + " Direction Loss": np.mean(dir_loss),
    })

    self.logger.log_scalar(np.mean(epoch_error), current + " Error", epoch)
    self.logger.log_scalar(np.mean(epoch_loss), current + " Loss", epoch)
    self.logger.log_scalar(np.mean(pos_loss), current + " Translation Loss", epoch)

    if current == "train":
      if epoch % 25 == 0 or epoch >= self.epochs - 1:
        self.save_model_all(logdir, self.model_name, epoch)

    elif current == "val":
      stop, save = early_stopping(np.mean(epoch_loss), self.model, epoch)
      if save:
        self.save_model_all(logdir, self.model_name, epoch, earlystop=True)
      return stop

  def save_model_all(self, save_dir, model_name, epoch, earlystop=False):
    """
    :param model:  nn model
    :param save_dir: save model direction
    :param model_name:  model name
    :param epoch:  epoch
    :return:  None
    """
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, model_name)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    if earlystop:
      save_path = '{}_earlystop.pt'.format(save_prefix)
    print("Saving model to {}".format(save_path))
    output = open(save_path, mode="wb")

    checkpoint = {
      'epoch': epoch,
      'state_dict': self.model.state_dict(),
      'optimizer': self.optimizer.state_dict()
    }

    torch.save(checkpoint, output)
    torch.save(checkpoint, os.path.join(wandb.run.dir, 'model.pt'))
    output.close()

  def plot(self, batch_img_names, batch_imgs, batch_pos, predicted_pos, epoch, batch, logdir,
           current, log=True):
    for j in range(len(batch_img_names)):
      img1_name = batch_img_names[j]
      real_action_pos = batch_pos[j]
      predicted_action_pos = predicted_pos[j]

      if self.history == 1:
        img1 = batch_imgs[j]

        plott = plotting.plot_single(
          img1.detach().cpu().squeeze(0).permute(1, 2, 0).numpy(),
          real_action_pos.detach().cpu(),
          predicted_action_pos=predicted_action_pos.detach().cpu(),
          img_name=img1_name, min_z=-1, max_z=1)

      folders = img1_name.split("/")
      img_loc = folders[-3] + "_" + folders[-1].split(".")[0]

      if current == "val":
        plott.savefig(logdir + '/valimages/' + img_loc)
        if log:
          self.logger.log_figure(plott.figure(1), "Predicted vs Actual Val", batch * len(batch_img_names) + j, epoch)
      elif current == "test":
        plott.savefig(logdir + '/testimages/' + img_loc)
      else:
        plott.savefig(logdir + '/trainimages/' + img_loc)
        if log:
          self.logger.log_figure(plott.figure(1), "Predicted vs Actual Train", batch * len(batch_img_names) + j, epoch)

      plott.close()

  def log_sometimes(self, epoch, i_batch, batch_img_names, batch_imgs, real_positions, real_angles_raw, pred_pos,
                    logdir, current):
    if epoch == -1:
      self.plot(batch_img_names, batch_imgs, real_positions, pred_pos, epoch, i_batch,
                logdir, current,
                log=False)
    if epoch % 20 == 0:
      if i_batch % 100 == 0:
        self.plot(batch_img_names, batch_imgs, real_positions, pred_pos, epoch, i_batch,
                  logdir, current, log=True)
      else:
        self.plot(batch_img_names, batch_imgs, real_positions, pred_pos, epoch, i_batch,
                  logdir, current, log=False)
    elif epoch == self.epochs - 1:
      self.plot(batch_img_names, batch_imgs, real_positions, pred_pos, epoch, i_batch,
                logdir, current,
                log=True)

  def get_label_act_pos(self, batch_actions):
    """

    :param batch_actions:
    :return:  real positions: tensor of size (batch, 3)
              real_angles: list of tensors of size 6 (batch, 6)
              raw_angles: tensor of size (batch, 3, 3)
    """

    # 1st row of every batch is positions (x,y,z), rest is angle
    real_positions = batch_actions[:, 0]
    real_angles_raw = batch_actions[:, 1:]
    real_angles = real_angles_raw

    if "6d" in self.rot:
      real_angles = convert_3dmat_to_6d_rep(real_angles_raw, self.device)
    return real_positions, real_angles, real_angles_raw

  def calculate_loss_error(self, pred_pos, real_positions):
      """
      total_loss = dir_weight * direction_loss + mse_weight * mse_loss
      where MSE is computed on translation vectors only.
      Returns: total_loss, total_error (== mse_loss), mse_loss, direction_loss
      """

      if isinstance(pred_pos, (list, tuple)):
          p_tensor = torch.stack(pred_pos, dim=0).reshape(len(pred_pos), -1).to(self.device)
      else:
          p_tensor = pred_pos.reshape(pred_pos.size(0), -1).to(self.device)

      if isinstance(real_positions, torch.Tensor):
          label_pos = real_positions.reshape(real_positions.size(0), -1).float().to(self.device)
      else:
          rows = []
          for l in real_positions:
              rows.append(l.tolist() if hasattr(l, "tolist") else l)
          label_pos = torch.as_tensor(rows, dtype=torch.float32, device=self.device).reshape(len(real_positions), -1)

      if self.trans_loss_type == 'l1':
          mse_loss = self.l1_loss(p_tensor, label_pos)
      else:
          mse_loss = self.mse_loss(p_tensor, label_pos)

      # Direction loss
      def direction_loss(pred, real):
          # only look at 0th and last column (x and z)
          real_xz = torch.stack((real[:, 0], real[:, -1]), dim=1).to(self.device)
          pred_xz = torch.stack((pred[:, 0], pred[:, -1]), dim=1).to(self.device)
          top = torch.sum(real_xz * pred_xz, dim=1)
          bottom = torch.norm(real_xz, dim=1) * torch.norm(pred_xz, dim=1)
          eps = 1e-7
          cos = torch.clamp(top / (bottom + eps), min=-1 + eps, max=1 - eps)
          return torch.mean(torch.acos(cos))

      d_loss = direction_loss(p_tensor, label_pos).to(self.device)

      dir_w = getattr(self, "dir_weight", 1.0)
      mse_w = getattr(self, "mse_weight", 1.0)
      #total_loss = dir_w * d_loss + mse_w * mse_loss
      total_loss = self.dir_weight * d_loss + self.trans_weight * mse_loss
      
      #print(dir_w, mse_w)
 
      total_error = mse_loss

      return total_loss, total_error, mse_loss, d_loss

class PushTrainerL1:
  """
  Trainer class
  """

  def __init__(self, model, params):
    torch.manual_seed(0)
    np.random.seed(0)

    self.params = params
    self.device = torch.device('cuda:' + str(params['which_gpu']) if torch.cuda.is_available() else "cpu")
    self.model = model
    self.logger = logger.Logger(self.params['logdir'])
    self.batch_size = params['batch_size']
    self.epochs = params['epochs']
    self.rot = params['rot']
    self.model_name = params['model']
    self.history = params['history']
    self.mult = params['mult']
    self.l1 = params['l1']
    self.l2 = params['l2']
    self.l3 = params['l3']
    self.rad = params['rad']
    self.net = params['net_type']
    self.mse_loss = nn.MSELoss()
    self.scaled_mse_loss = nn.MSELoss(reduce=False)
    self.l1_loss = nn.L1Loss()
    self.dir_weight = 1
    self.mse_weight = 1 # CHANGE THIS

    self.train_set = PushDataset.PushDataset("train", params)
    self.num_train = len(self.train_set)
    self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                                                    sampler=None,
                                                    batch_sampler=None, num_workers=4,
                                                    pin_memory=True, drop_last=False, timeout=0,
                                                    worker_init_fn=None)

    self.val_set = PushDataset.PushDataset("val", params)
    self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=True, drop_last=False, timeout=0,
                                                  worker_init_fn=None)
    self.num_val = len(self.val_set)

    self.test_set = PushDataset.PushDataset("test", params)
    self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                                                   num_workers=1,
                                                   pin_memory=False, drop_last=False, timeout=0,
                                                   worker_init_fn=None)
    self.num_test = len(self.test_set)

    wandb.init(project="trashbot")
    wandb.save("*.pt")
    params['num_train_runs'] = self.num_train
    params['num_val_runs'] = self.num_val
    params['num_test_runs'] = self.num_test
    wandb.config.update(params)

    self.optimizer = optim.Adam(model.parameters(), lr=self.params['lr'])

  def train(self, logdir=None):
    wandb.watch(self.model)

    self.num_train_batches = np.ceil(self.num_train / self.batch_size)
    early_stopping = EarlyStop(patience=150)
    num_epochs = self.epochs
    self.model.to(self.device)
    for epoch in tqdm(range(num_epochs), desc="Epoch"):

      self.wandb_dict = {}
      ######### TRAIN ===============================================================================================
      self.model.train()
      self.run_loop(self.train_loader, "train", epoch, logdir, None)

      ######### VALIDATE ============================================================================================
      self.model.eval()
      with torch.no_grad():
        early_stop = self.run_loop(self.val_loader, "val", epoch, logdir, early_stopping)

      ######### TEST ============================================================================================
      self.model.eval()
      with torch.no_grad():
        self.run_loop(self.test_loader, "test", epoch, logdir, None)

      wandb.log(self.wandb_dict)

      if early_stop:
        print("Early stopping at epoch " + str(epoch))

  def run_loop(self, data_loader, current, epoch, logdir, early_stopping):

    epoch_error = []
    epoch_loss = []

    pos_loss = []
    pos_error = []

    ang_loss = []
    ang_error = []

    dir_loss = []

    for i_batch, (batch_img_names, batch_imgs, batch_actions, runs, aug) in enumerate(data_loader):
      # Debug: print shapes/dtypes/devices for the first batch to help trace CUDA errors
      if i_batch == 0:
        try:
          print(f"[DEBUG] batch_imgs: shape={getattr(batch_imgs,'shape',None)}, dtype={getattr(batch_imgs,'dtype',None)}, device={getattr(batch_imgs,'device',None)}")
          print(f"[DEBUG] batch_actions: shape={getattr(batch_actions,'shape',None)}, dtype={getattr(batch_actions,'dtype',None)}, device={getattr(batch_actions,'device',None)}")
        except Exception:
          pass
      batch_imgs = batch_imgs.to(self.device)
      # make sure batch_actions is moved to device (reassign .to result)
      batch_actions = batch_actions.to(self.device)

      # Debug: after moving to device
      if i_batch == 0:
        try:
          print(f"[DEBUG after .to()] batch_imgs device={getattr(batch_imgs,'device',None)}, batch_actions device={getattr(batch_actions,'device',None)}")
        except Exception:
          pass

      real_positions, _, real_angles_raw = self.get_label_act_pos(batch_actions)

      real_angles_raw = real_angles_raw.to(self.device)
      real_positions = real_positions.to(self.device)

      pred_pos = self.model.forward(batch_imgs)

      # Debug: inspect prediction tensor for shape/device/grad
      if i_batch == 0:
        try:
          print(f"[DEBUG] pred_pos: shape={getattr(pred_pos,'shape',None)}, dtype={getattr(pred_pos,'dtype',None)}, device={getattr(pred_pos,'device',None)}, requires_grad={getattr(pred_pos,'requires_grad',None)}")
        except Exception:
          pass

      total_loss, total_error, p_loss, d_loss = self.calculate_loss_error(
        pred_pos,
        real_positions,
      )
      if current == "train":
        self.optimizer.zero_grad()
        # 3. backward propagation
        total_loss.backward()

        # 4. weight optimization
        self.optimizer.step()

      epoch_loss.append(total_loss.item())
      epoch_error.append(total_error.item())

      pos_loss.append(p_loss.item())

      dir_loss.append(d_loss.item())


      if i_batch % 20 == 0 and current == "train":
        print('Epoch [{}/{}, Step [{}/{}], Loss: {:.4f}'
              .format(epoch, self.epochs, i_batch + 1, self.num_train_batches, total_loss.item()))

      self.log_sometimes(epoch, i_batch, batch_img_names, batch_imgs, real_positions, real_angles_raw, pred_pos,
                         logdir, current)

    print("Epoch", epoch, current + " Error", np.mean(epoch_error))
    print("Epoch", epoch, current + " Loss", np.mean(epoch_loss))

    self.wandb_dict.update({
      current.lower().capitalize() + " Error": np.mean(epoch_error),
      current.lower().capitalize() + " Loss": np.mean(epoch_loss),
      current.lower().capitalize() + " Translation Loss": np.mean(pos_loss),
      current.lower().capitalize() + " Direction Loss": np.mean(dir_loss),
    })

    self.logger.log_scalar(np.mean(epoch_error), current + " Error", epoch)
    self.logger.log_scalar(np.mean(epoch_loss), current + " Loss", epoch)
    self.logger.log_scalar(np.mean(pos_loss), current + " Translation Loss", epoch)

    if current == "train":
      if epoch % 25 == 0 or epoch >= self.epochs - 1:
        self.save_model_all(logdir, self.model_name, epoch)

    elif current == "val":
      stop, save = early_stopping(np.mean(epoch_loss), self.model, epoch)
      if save:
        self.save_model_all(logdir, self.model_name, epoch, earlystop=True)
      return stop

  def save_model_all(self, save_dir, model_name, epoch, earlystop=False):
    """
    :param model:  nn model
    :param save_dir: save model direction
    :param model_name:  model name
    :param epoch:  epoch
    :return:  None
    """
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, model_name)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    if earlystop:
      save_path = '{}_earlystop.pt'.format(save_prefix)
    print("Saving model to {}".format(save_path))
    output = open(save_path, mode="wb")

    checkpoint = {
      'epoch': epoch,
      'state_dict': self.model.state_dict(),
      'optimizer': self.optimizer.state_dict()
    }

    torch.save(checkpoint, output)
    torch.save(checkpoint, os.path.join(wandb.run.dir, 'model.pt'))
    output.close()

  def plot(self, batch_img_names, batch_imgs, batch_pos, predicted_pos, epoch, batch, logdir,
           current, log=True):
    for j in range(len(batch_img_names)):
      img1_name = batch_img_names[j]
      real_action_pos = batch_pos[j]
      predicted_action_pos = predicted_pos[j]

      if self.history == 1:
        img1 = batch_imgs[j]

        plott = plotting.plot_single(
          img1.detach().cpu().squeeze(0).permute(1, 2, 0).numpy(),
          real_action_pos.detach().cpu(),
          predicted_action_pos=predicted_action_pos.detach().cpu(),
          img_name=img1_name, min_z=-1, max_z=1)

      folders = img1_name.split("/")
      img_loc = folders[-3] + "_" + folders[-1].split(".")[0]

      if current == "val":
        plott.savefig(logdir + '/valimages/' + img_loc)
        if log:
          self.logger.log_figure(plott.figure(1), "Predicted vs Actual Val", batch * len(batch_img_names) + j, epoch)
      elif current == "test":
        plott.savefig(logdir + '/testimages/' + img_loc)
      else:
        plott.savefig(logdir + '/trainimages/' + img_loc)
        if log:
          self.logger.log_figure(plott.figure(1), "Predicted vs Actual Train", batch * len(batch_img_names) + j, epoch)

      plott.close()

  def log_sometimes(self, epoch, i_batch, batch_img_names, batch_imgs, real_positions, real_angles_raw, pred_pos,
                    logdir, current):
    if epoch == -1:
      self.plot(batch_img_names, batch_imgs, real_positions, pred_pos, epoch, i_batch,
                logdir, current,
                log=False)
    if epoch % 20 == 0:
      if i_batch % 100 == 0:
        self.plot(batch_img_names, batch_imgs, real_positions, pred_pos, epoch, i_batch,
                  logdir, current, log=True)
      else:
        self.plot(batch_img_names, batch_imgs, real_positions, pred_pos, epoch, i_batch,
                  logdir, current, log=False)
    elif epoch == self.epochs - 1:
      self.plot(batch_img_names, batch_imgs, real_positions, pred_pos, epoch, i_batch,
                logdir, current,
                log=True)

  def get_label_act_pos(self, batch_actions):
    """

    :param batch_actions:
    :return:  real positions: tensor of size (batch, 3)
              real_angles: list of tensors of size 6 (batch, 6)
              raw_angles: tensor of size (batch, 3, 3)
    """

    # 1st row of every batch is positions (x,y,z), rest is angle
    real_positions = batch_actions[:, 0]
    real_angles_raw = batch_actions[:, 1:]
    real_angles = real_angles_raw

    if "6d" in self.rot:
      real_angles = convert_3dmat_to_6d_rep(real_angles_raw, self.device)
    return real_positions, real_angles, real_angles_raw

  def calculate_loss_error(self, pred_pos, real_positions):
      """
      total_loss = dir_weight * direction_loss + mse_weight * mse_loss
      where MSE is computed on translation vectors only.
      Returns: total_loss, total_error (== mse_loss), mse_loss, direction_loss
      """

      if isinstance(pred_pos, (list, tuple)):
          p_tensor = torch.stack(pred_pos, dim=0).reshape(len(pred_pos), -1).to(self.device)
      else:
          p_tensor = pred_pos.reshape(pred_pos.size(0), -1).to(self.device)

      if isinstance(real_positions, torch.Tensor):
          label_pos = real_positions.reshape(real_positions.size(0), -1).float().to(self.device)
      else:
          rows = []
          for l in real_positions:
              rows.append(l.tolist() if hasattr(l, "tolist") else l)
          label_pos = torch.as_tensor(rows, dtype=torch.float32, device=self.device).reshape(len(real_positions), -1)



      mse_loss = self.mse_loss(p_tensor, label_pos)
      l1_loss = self.l1_loss(p_tensor, label_pos)

      # Direction loss
      def direction_loss(pred, real):
          # only look at 0th and last column (x and z)
          real_xz = torch.stack((real[:, 0], real[:, -1]), dim=1).to(self.device)
          pred_xz = torch.stack((pred[:, 0], pred[:, -1]), dim=1).to(self.device)
          top = torch.sum(real_xz * pred_xz, dim=1)
          bottom = torch.norm(real_xz, dim=1) * torch.norm(pred_xz, dim=1)
          eps = 1e-7
          cos = torch.clamp(top / (bottom + eps), min=-1 + eps, max=1 - eps)
          return torch.mean(torch.acos(cos))

      d_loss = direction_loss(p_tensor, label_pos).to(self.device)

      dir_w = getattr(self, "dir_weight", 1.0)
      mse_w = getattr(self, "mse_weight", 1.0)
      #total_loss = dir_w * d_loss + mse_w * mse_loss
      # total_loss = self.dir_weight * d_loss + self.trans_weight * mse_loss
      total_loss = self.dir_weight * d_loss + self.mse_weight * mse_loss
      
      #print(dir_w, mse_w)
 
      total_error = mse_loss

      return total_loss, total_error, mse_loss, d_loss