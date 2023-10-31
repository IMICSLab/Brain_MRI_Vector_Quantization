# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:54:00 2022

@author: Ernest Namdar (ernest.namdar@utoronto.ca)

Modify on 2022-10-28 by Simon, for the use in image generation using GAN model
"""
# importing the required libraries#############################################
# >>>>Seeding function libraries
import numpy as np
import torch
import random
# >>>>other required libs
import pickle
from torch.utils.data import Dataset


# Seeding######################################################################
def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

random_seed(0, True)
###############################################################################


class iMICSDataset(Dataset):
    def __init__(self, rois, pIDs, inds, shuffle):
        self.len = len(inds)
        #self.Imgs = [Imgs[i] for i in inds]
        self.Rois = [rois[i] for i in inds]
        #self.labels = [labels[i] for i in inds] # do not need label for now
        self.pIDs = [pIDs[i] for i in inds]

        if shuffle is True:
            p = np.random.permutation(self.len)
            #self.Imgs = [self.Imgs[i] for i in p]
            self.Rois = [self.Rois[i] for i in p]
            #self.labels = [self.labels[i] for i in p]
            self.pIDs = [self.pIDs[i] for i in p]
        #self.Imgs = np.stack(self.Imgs, axis=0)
        self.Rois = np.stack(self.Rois, axis=0)

    def __len__(self):
        return (self.len)

    def __getitem__(self, idx):
        #img = self.Imgs[idx, :, :, :]
        roi = self.Rois[idx, :, :, :]
        #lbl = self.labels[idx]
        # pid = self.pIDs[idx]
        # roi = img*msk
        sample = {'ROI': roi} # for image generation, we will only use Img and Mask / or roi
        return sample
    

class iMICSDataset128(Dataset):
    def __init__(self, rois, inds, shuffle):
        self.len = len(inds)
        #self.Imgs = [Imgs[i] for i in inds]
        self.Rois = [rois[i] for i in inds]
        #self.labels = [labels[i] for i in inds] # do not need label for now
        #self.pIDs = [pIDs[i] for i in inds]

        if shuffle is True:
            p = np.random.permutation(self.len)
            #self.Imgs = [self.Imgs[i] for i in p]
            self.Rois = [self.Rois[i] for i in p]
            #self.labels = [self.labels[i] for i in p]
            #self.pIDs = [self.pIDs[i] for i in p]
        #self.Imgs = np.stack(self.Imgs, axis=0)
        self.Rois = torch.stack(self.Rois, axis=0)

    def __len__(self):
        return (self.len)

    def __getitem__(self, idx):
        #img = self.Imgs[idx, :, :, :]
        roi = self.Rois[idx, :, :, :]
        #lbl = self.labels[idx]
        # pid = self.pIDs[idx]
        # roi = img*msk
        sample = {'ROI': roi} # for image generation, we will only use Img and Mask / or roi
        return sample
    # def positive_ratio(self):
    #     count = 0
    #     for lb in self.labels:
    #         if lb==1:
    #             count += 1
    #     return count/len(self)
    # def negative_ratio(self):
    #     return 1-self.positive_ratio()

class iMICSDataset128_new(Dataset):
    def __init__(self, rois, inds, shuffle, labels=None, need_lab = False):
        self.len = len(inds)
        self.need_lab = need_lab
        #self.Imgs = [Imgs[i] for i in inds]
        self.Rois = [rois[i] for i in inds]
        if self.need_lab:
            self.labels = [labels[i] for i in inds] # label for cond transformer
        #self.pIDs = [pIDs[i] for i in inds]

        if shuffle is True:
            p = np.random.permutation(self.len)
            #self.Imgs = [self.Imgs[i] for i in p]
            self.Rois = [self.Rois[i] for i in p]
            if self.need_lab:
                self.labels = [self.labels[i] for i in p]
            #self.pIDs = [self.pIDs[i] for i in p]
        #self.Imgs = np.stack(self.Imgs, axis=0)
        self.Rois = torch.stack(self.Rois, axis=0)

    def __len__(self):
        return (self.len)

    def __getitem__(self, idx):
        #img = self.Imgs[idx, :, :, :]
        roi = self.Rois[idx, :, :, :]
        if self.need_lab:
            lbl = torch.Tensor([self.labels[idx]])
        # pid = self.pIDs[idx]
        # roi = img*msk
        if self.need_lab:
            sample = {'ROI': roi, "label": lbl}
        else:
            sample = {'ROI': roi} # for image generation, we will only use Img and Mask / or roi
        return sample
# def load_object(filename):
#   # opening a file in read, binary form
#   file = open(filename, 'rb')

#   ob = pickle.load(file)

#   # close the file
#   file.close()
#   return ob


# if __name__ == '__main__':
#     Msks = load_object("../data/binMasks.p")
#     Imgs = load_object("../data/binFLAIR_Imgs.p")
#     # labels = load_object("../data/binlabels.p")
#     pIDs = load_object("../data/binpIDs.p")

#     inds = [3,4,5,7,10]
#     dset = iMICSDataset(Imgs, Msks, pIDs, inds, shuffle=True)
