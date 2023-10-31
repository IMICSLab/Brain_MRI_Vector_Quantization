# this script is for image gradient loss


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

import sys
sys.path.append("/home/szhou/pLGG")

import os


class img_grad(nn.Module):
    def __init__(self, device, b_size) -> None:
        super(img_grad, self).__init__()
        
        self.device = device
        self.x_gradient_filter = torch.Tensor(
            [
                #[[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                # [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                # [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
            ]
        ).to(self.device)
        self.x_gradient_filter = self.x_gradient_filter.unsqueeze(0)#.repeat(b_size,1,1,1)

        self.y_gradient_filter = torch.Tensor(
            [
                #[[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                # [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                # [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                [[1., 2., 1.],[0, 0, 0],[-1., -2., -1.]],
            ]
        ).to(self.device)
        self.y_gradient_filter = self.y_gradient_filter.unsqueeze(0)#.repeat(b_size,1,1,1)

        self.weight_x = nn.Parameter(data=self.x_gradient_filter, requires_grad=False).to(device)
        self.weight_y = nn.Parameter(data=self.y_gradient_filter, requires_grad=False).to(device)
    
    def forward(self, true_img, rec_img):
        '''
        input should be 2d images with shape [batch, 1, h, w]
        batch should be 3
        '''
        grad_x_true = F.conv2d(true_img, self.weight_x, padding=1)
        grad_x_rec = F.conv2d(rec_img, self.weight_x, padding=1)
        grad_y_true = F.conv2d(true_img, self.weight_y, padding=1)
        grad_y_rec = F.conv2d(rec_img, self.weight_y, padding=1)

        x_diff = ((torch.abs(grad_x_true - grad_x_rec) ** 2)).mean()
        y_diff = ((torch.abs(grad_y_true - grad_y_rec) ** 2)).mean()

        return x_diff + y_diff



def img_grad_loss_3d(true_img, rec_img, device):
    '''
    compute img grad loss in 3d
    true_img and rec_img: should have size [batch, 1, 128, 128, 128], channel first
    '''

    _batch_size = true_img.shape[0]
    img_grad_loss = img_grad(device, _batch_size)

    total = 0

    for i in range(true_img.shape[2]):
        temp = 0
        # now get 3d loss
        img_trueX = true_img[:,:,i,:,:]
        img_recX = rec_img[:,:,i,:,:]
        img_grad_lossX = img_grad_loss(img_trueX, img_recX)
        temp += img_grad_lossX

        img_trueY = true_img[:,:,:,i,:]
        img_recY = rec_img[:,:,:,i,:]
        img_grad_lossY = img_grad_loss(img_trueY, img_recY)
        temp += img_grad_lossY

        img_trueZ = true_img[:,:,:,:,i]
        img_recZ = rec_img[:,:,:,:,i]
        img_grad_lossZ = img_grad_loss(img_trueZ, img_recZ)
        temp += img_grad_lossZ

        #temp /= 3
        total += temp
    
    return total / true_img.shape[2]