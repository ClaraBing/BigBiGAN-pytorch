import os
import torch
import numpy as np
from math import sqrt, ceil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2

import pdb

def show_net_weights(fckpt, fimg, fnpy='', ckpt_key='backbone.features.0.weight',
                     blur=0, single_channel=0):
  ckpt = torch.load(fckpt)
  W1 = ckpt[ckpt_key]
  W1 = W1.cpu().numpy()
  W1 = W1.transpose(0,2,3,1)

  grid = visualize_grid(W1, padding=1, blur=blur, single_channel=single_channel).astype('uint8')
  cv2.imwrite(fimg, grid)

scale = 1
scale_trim = 1

def visualize_grid(Xs, ubound=255.0, padding=1,
                                 blur=0, single_channel=0):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.
    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    - blur: whether to blur(smooth) the image.
    - single_channel: whether to visualize each channel individually.
    """
    (N, H, W, C) = Xs.shape
    if single_channel:
      Xs = Xs.transpose(0, 3, 1, 2).reshape(-1, H, W)
      N = N * C
      C = 1
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    if single_channel:
      grid = np.zeros((grid_height, grid_width))
    else:
      grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    low, high = np.min(Xs), np.max(Xs)
    trim = scale_trim*np.std(Xs)
    mean = np.mean(Xs)
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                if blur:
                  img = cv2.blur(img,(blur, blur))
                img -= mean
                img = np.minimum(img, trim)
                img = np.maximum(img, -trim)
                img = ubound * (img + trim) * scale
                grid[y0:y1, x0:x1] = img
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid

if __name__ == '__main__':
  blur = 0
  single_channel = 0
 
  if 1:
    # pretrained
    ker_size = 7
    fckpt = 'pretrained_weights/resnet50_v2_pretrained.pt'
    ckpt_key = 'conv1.weight'
    fimg = 'images/bigbigan_pretrained_conv1_k{}_scale{}_trim{}.png'.format(ker_size, scale, scale_trim)

  show_net_weights(fckpt, fimg, ckpt_key=ckpt_key, blur=blur, single_channel=single_channel)



