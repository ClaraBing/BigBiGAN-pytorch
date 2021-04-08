#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
from collections import namedtuple
import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils
from torch.autograd import Variable
import torchvision.models as models

import numpy as np
import matplotlib
matplotlib.use('Agg')
import importlib
import cv2
# from scipy.ndimage.filters import gaussian_filter1d
import random
import pdb

from resnet50 import resnet50
from resnet50_pytorch import resnet50 as resnet50_torch

from utils_vis import *

torch.cuda.set_device(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.FloatTensor

# DATASET = 'cifar10'  # cifar10, imagenet
IMG_SIZE = 256
CIFAR_MEAN=torch.tensor([0.4913997551666284, 0.48215855929893703, 0.4465309133731618]).type(dtype)
CIFAR_STD=torch.tensor([0.24703225141799082, 0.24348516474564, 0.26158783926049628]).type(dtype)
IMG_MEAN = torch.tensor([0.5, 0.5, 0.5])
IMG_STD = torch.tensor([0.5, 0.5, 0.5])


def get_encoder(fckpt=''):
  if fckpt and os.path.exists(fckpt):
    E = resnet50()
    loaded_sd = torch.load(fckpt)
    try:
      E.load_state_dict(loaded_sd)
    except:
      curr_params = E.state_dict()
      curr_keys = list(curr_params.keys())

      updated_params = {}
      for k,v in loaded_sd.items():
        if k in curr_keys and loaded_sd[k].shape==curr_params[k].shape:
          updated_params[k] = v
        else:
          print('Failed to load:', k)
      curr_params.update(updated_params)
      E.load_state_dict(curr_params)
  else:
    # load the state_dict given by PyTorch modelzoo
    E = resnet50_torch(pretrained=True)
  return E.to(device)


if __name__ == '__main__':
  scale = 1

  # fckpt = 'pretrained_weights/resnet50_v2_pretrained_cnt.pt'
  fckpt = 'pretrained_weights/resnet50_v2_pretrained_cnt_reordered.pt'
  fckpt = ''
  E = get_encoder(fckpt)

  hidden_channels = {
    0: 256,
    1: 512,
    2: 1024,
    3: 2048,
    -1: 2048,
  }
  center_locs = {
    0: 3,
    1: 1,
    2: 0, 
    3: 0,
    -1: 0, 
  }
  uptos = {
    -1:-1
  }
  E = get_encoder(fckpt)

  img_size = [3, IMG_SIZE, IMG_SIZE]
  coord_idx = 0
  if not fckpt or not os.path.exists(fckpt):
    subfolder = 'pytorch_pretrained'
  else:
    subfolder = 'pretrained'
  img_token = ''

  for ret_layer in [3]:
    bdd_pixels = 1
    if ret_layer in uptos:
      upto = uptos[ret_layer]
    else:
      upto = -1

    l2_reg = 1e-2
    num_iterations = 600
    for lr in [0.0003, 0.001, 0.003]:
      img_token = '{}std_lr{}_reg{}_'.format(scale, lr, l2_reg)
      # for coord_idx in range(hidden_channels[ret_layer]):
      for coord_idx in range(0, 30):
        yosinski(coord_idx, E, subfolder=subfolder, ret_layer=ret_layer,
          learning_rate=lr, l2_reg=l2_reg, num_iterations=num_iterations,
          center_loc = center_locs[ret_layer],
          upto=upto, img_token=img_token, scale=scale, bdd_pixels=bdd_pixels)
