import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
from torch.utils.data import Dataset
import random
import sys
sys.path.append("../")

from utils.imicsDataset_cust import iMICSDataset128, iMICSDataset128_new
from utils.find_large_bbox import pad_image
from utils.center_crop import *


# def random_seed(seed_value, use_cuda):
#     np.random.seed(seed_value) # cpu vars
#     torch.manual_seed(seed_value) # cpu  vars
#     random.seed(seed_value) # Python
#     if use_cuda:
#         torch.cuda.manual_seed(seed_value)
#         torch.cuda.manual_seed_all(seed_value) # gpu vars
#         torch.backends.cudnn.deterministic = True  #needed
#         torch.backends.cudnn.benchmark = False


def load_object(filename):
  # opening a file in read, binary form
    file = open(filename, 'rb')

    ob = pickle.load(file)

    # close the file
    file.close()
    return ob


def get_target_pids(file, target, hospital):
    '''
    file: meta data file
    target: target subtype: BRAF fusion, mutation, NF1, ...
    exclude: the exlude patients in the meta data file
    hospital: "SK" for SickKids or "Stanford" for Stanford children
    '''
    
    exclude = [9, 12, 23, 37, 58, 74, 78, 85, 121, 122, 130, 131, 138, 140, 
    150, 171, 176, 182, 204, 213, 221, 224, 234, 245, 246, 274, 306, 311, 
    312, 330, 333, 347, 349, 352, 354, 359, 364, 377]

    meta_data = pd.read_excel(file, sheet_name=hospital)
    meta_data = meta_data.iloc[0:400,:]
    target_pids = np.array(meta_data.loc[~meta_data[target].isnull()]["code"]).astype(int)
    target_valid_pids = [i for i in target_pids if i not in exclude]# filter out if the index is excluded
    return target_valid_pids


def get_bad_index(target_pids, all_pids):
    '''
    return index(pids) in the target subtype but not in the processed data
    target_pids: list of pids for the target subtype
    all_pids: all processed pids (pIDs.p file)
    '''
    valid_idx = []
    bad_inx = []
    for i in target_pids:
        try: # filter out the data is not been processed
            temp = all_pids.index(i)
            valid_idx.append(temp)
        except ValueError:
            bad_inx.append(i)
    
    return valid_idx, bad_inx


# def mask_start_end(img):
#     '''
#     return the start ind of mask and end ind, in order to get the reduced slices
#     '''
#     start = -1
#     end = -1
#     for i in range(img.shape[-1]):
#         if torch.any(img[:,:,i]):
#             start = i
#             break
    
#     for j in range(img.shape[-1]-1, -1, -1):
#         if torch.any(img[:,:,j]):
#             end = j
#             break
    
#     return start, end

# def slice_pad(start, end, target):
#     slices = end - start + 1
#     left = target - slices
#     pref = suf = 0
#     if left % 2 == 0:
#         pref = suf = left // 2
#     else:
#         pref = left // 2
#         suf = left // 2 + 1
    
#     return pref, suf

def normalize(im):
    # normalize to 0,1
    mins = [im[idx].min() for idx in range(len(im))]
    maxes = [im[idx].max() for idx in range(len(im))]

    for idx in range(len(im)):
        min_val = mins[idx]
        max_val = maxes[idx]

        if min_val == max_val:
            im[idx] = torch.zeros(im[idx].shape)
        else:
            im[idx] = (im[idx] - min_val)/(max_val - min_val)

def normalize2(im):
    im = (im-torch.min(im))/(torch.max(im)-torch.min(im)) if torch.max(im) != torch.min(im) else im # to 0,1
    im = 2 * im - 1 # to -1,1
    return im

def normalize_sk(im):
    new = (2 * im) - 1
    return new

def get_data_loader(Imgs, Msks, pIDs, idx, shuffle, norm = True):
    
    # which class? based on the idx of the pids
    if norm:
        for i in range(len(Imgs)):
            normalize(Imgs[i])
    target_dataset = iMICSDataset(Imgs, Msks, pIDs, idx, shuffle=shuffle)
    return target_dataset

# def get_data_loader2(rois, pIDs, idx, shuffle, norm = True):
    
#     # which class? based on the idx of the pids
#     if norm:
#         for i in range(len(rois)):
#             normalize(rois[i])
#     target_dataset = iMICSDataset(rois, pIDs, idx, shuffle=shuffle)
#     return target_dataset


def get_data_loader_128(rois, idx, shuffle, norm = True):
    # load center cropped 128*128*128 data
    # which class? based on the idx of the pids
    # here, rois is in torch file
    if norm:
        for i in range(rois.shape[0]):
            normalize(rois[i,:,:,:])
    target_dataset = iMICSDataset128(rois, idx, shuffle=shuffle)
    return target_dataset

def get_data_loader_128_cond(rois, labels, idx, shuffle, norm = True):
    # load center cropped 128*128*128 data
    # which class? based on the idx of the pids
    # here, rois is in torch file
    if norm:
        for i in range(rois.shape[0]):
            normalize(rois[i,:,:,:])
    target_dataset = iMICSDataset128_new(rois, idx, shuffle, labels, need_lab = True)
    return target_dataset

def get_data_batch(data_batch, device):
    
    if torch.cuda.is_available():
        inputs = Variable(data_batch["Img"].float().to(device))
        masks = Variable(data_batch["Msk"].float().to(device))
    else:
        inputs = Variable(data_batch["Img"].float())
        masks = Variable(data_batch["Msk"].float())
    
    assert inputs.shape == masks.shape, "something wrong with load the data, two tensor sizes do not match"
    
    if inputs.shape[-1] == 155: # channel is in the last dim, switch to the front
        inputs = inputs.permute((0, 3, 1, 2)).contiguous()
        masks = masks.permute((0, 3, 1, 2)).contiguous()

    inputs = inputs.unsqueeze(1)
    masks = masks.unsqueeze(1)

    img_mask_pair = torch.cat((inputs, masks), dim = 1)

    return img_mask_pair # should return [batch, 2, 155, 240, 240]

def get_data_batch_imgOnly(data_batch, device):
    
    if torch.cuda.is_available():
        inputs = Variable(data_batch["Img"].float().to(device))
        masks = Variable(data_batch["Msk"].float().to(device))
    else:
        inputs = Variable(data_batch["Img"].float())
        masks = Variable(data_batch["Msk"].float())
    
    assert inputs.shape == masks.shape, "something wrong with load the data, two tensor sizes do not match"
    
    if inputs.shape[-1] == 155: # channel is in the last dim, switch to the front
        inputs = inputs.permute((0, 3, 1, 2)).contiguous()
        #masks = masks.permute((0, 3, 1, 2)).contiguous()

    inputs = inputs.unsqueeze(1)
    #masks = masks.unsqueeze(1)

    #img_mask_pair = torch.cat((inputs, masks), dim = 1)

    return inputs # should return [batch, 2, 155, 240, 240]


def get_data_batch_ROI(data_batch, device):
    
    if torch.cuda.is_available():
        inputs = Variable(data_batch["Img"].float().to(device))
        masks = Variable(data_batch["Msk"].float().to(device))
    else:
        inputs = Variable(data_batch["Img"].float())
        masks = Variable(data_batch["Msk"].float())
    
    assert inputs.shape == masks.shape, "something wrong with load the data, two tensor sizes do not match"

    # get roi
    roi_whole = inputs * masks
    roi_01 = (roi_whole-torch.min(roi_whole))/(torch.max(roi_whole)-torch.min(roi_whole)) if torch.max(roi_whole) != torch.min(roi_whole) else roi_whole
    rois = (2 * roi_01) - 1 # [-1, 1]

    if rois.shape[-1] == 155: # channel is in the last dim, switch to the front
        rois = rois.permute((0, 3, 1, 2)).contiguous()
    
    rois = rois.unsqueeze(1) # [batch, 1, 155, 240, 240]

    return rois

def get_data_batch_ROI_small(data_batch, rmin, rmax, cmin, cmax, target, device):

    if torch.cuda.is_available():
        inputs = Variable(data_batch["Img"].float().to(device))
        masks = Variable(data_batch["Msk"].float().to(device))
    else:
        inputs = Variable(data_batch["Img"].float())
        masks = Variable(data_batch["Msk"].float())
    
    assert inputs.shape == masks.shape, "something wrong with load the data, two tensor sizes do not match"

    # get roi and crop mask
    
    roi_whole = inputs * masks
    roi = roi_whole[:, rmin:rmax+1, cmin:cmax,:] # this cmin:cmax only used for BraTS19 data

    # pad image
    rois = torch.zeros(inputs.shape[0], target, target, inputs.shape[-1], device=device)
    for i in range(roi.shape[0]):
        for j in range(roi[i].shape[-1]):
            rois[i,:,:,j] = pad_image(roi[i,:,:,j], target)

    # normalize to -1,1, batch by batch
    for i in range(rois.shape[0]):
        rois[i,:,:,:] = normalize2(rois[i,:,:,:])
    #print("rois max min pixel: ", torch.max(rois), torch.min(rois))
    #roi_01 = (roi_whole-torch.min(roi_whole))/(torch.max(roi_whole)-torch.min(roi_whole)) if torch.max(roi_whole) != torch.min(roi_whole) else roi_whole
    #rois = (2 * roi_01) - 1 # [-1, 1]

    if rois.shape[-1] == 155: # channel is in the last dim, switch to the front
        rois = rois.permute((0, 3, 1, 2)).contiguous()
    
    rois = rois.unsqueeze(1) # [batch, 1, 155, 176, 176]

    return rois, masks


def get_data_batch_ROI_slice_small(data_batch, rmin, rmax, cmin, cmax, target, slices, device):

    if torch.cuda.is_available():
        inputs = Variable(data_batch["ROI"].float().to(device))
        #masks = Variable(data_batch["Msk"].float().to(device))
    else:
        inputs = Variable(data_batch["ROI"].float())
        #masks = Variable(data_batch["Msk"].float())
    
    #assert inputs.shape == masks.shape, "something wrong with load the data, two tensor sizes do not match"

    # get roi and crop mask
    roi_whole = inputs #inputs * masks
    roi = roi_whole[:, rmin:rmax+1, cmin:cmax,:] # this cmin:cmax only used for BraTS19 data

    # pad image h,w
    rois = torch.zeros(inputs.shape[0], target, target, inputs.shape[-1], device=device)
    for i in range(roi.shape[0]):
        for j in range(roi[i].shape[-1]):
            rois[i,:,:,j] = pad_image(roi[i,:,:,j], target)

    # now, process slices
    rois_new = torch.zeros(inputs.shape[0], target, target, slices, device=device) # 176, 176, slices
    for i in range(rois.shape[0]):
        mask_start, mask_end = mask_start_end(rois[i,:,:,:])
        temp = rois[i,:,:,mask_start:mask_end+1]
        #print(mask_start, mask_end, temp.shape)
        pad_pref, pad_suf = slice_pad(mask_start, mask_end, slices)
        #print(pad_pref, pad_suf)
        temp = torch.cat((torch.zeros(target, target, pad_pref, device=device), temp), dim = -1)
        temp = torch.cat((temp, torch.zeros(target, target, pad_suf, device=device)), dim = -1)
        rois_new[i,:,:,:] = temp
    
    # normalize to -1,1, batch by batch
    for i in range(rois_new.shape[0]):
        rois_new[i,:,:,:] = normalize2(rois_new[i,:,:,:])
    #print("rois max min pixel: ", torch.max(rois), torch.min(rois))
    #roi_01 = (roi_whole-torch.min(roi_whole))/(torch.max(roi_whole)-torch.min(roi_whole)) if torch.max(roi_whole) != torch.min(roi_whole) else roi_whole
    #rois = (2 * roi_01) - 1 # [-1, 1]

    #if rois_new.shape[-1] == 155: # channel is in the last dim, switch to the front
    rois_new = rois_new.permute((0, 3, 1, 2)).contiguous()
    assert rois_new.shape[1:] == (slices, target, target), "roi_small, reduce slices go wrong"
    
    rois_new = rois_new.unsqueeze(1) # [batch, 1, 155, 176, 176]

    return rois_new


# def get_data_batch_ROI_128(data_batch, device):

#     if torch.cuda.is_available():
#         inputs = Variable(data_batch["ROI"].float().to(device))
#     else:
#         inputs = Variable(data_batch["ROI"].float())
    
#     #assert inputs.shape == masks.shape, "something wrong with load the data, two tensor sizes do not match"

#     # get roi and crop mask
#     roi_whole = inputs #inputs * masks
    
#     # normalize to -1,1, batch by batch
#     for i in range(roi_whole.shape[0]):
#         roi_whole[i,:,:,:] = normalize2(roi_whole[i,:,:,:])
#     #print("rois max min pixel: ", torch.max(rois), torch.min(rois))
#     #roi_01 = (roi_whole-torch.min(roi_whole))/(torch.max(roi_whole)-torch.min(roi_whole)) if torch.max(roi_whole) != torch.min(roi_whole) else roi_whole
#     #rois = (2 * roi_01) - 1 # [-1, 1]

#     #if rois_new.shape[-1] == 155: # channel is in the last dim, switch to the front
#     roi_whole = roi_whole.permute((0, 3, 1, 2)).contiguous()
#     #assert rois_new.shape[1:] == (slices, target, target), "roi_small, reduce slices go wrong"
    
#     roi_whole = roi_whole.unsqueeze(1) # [batch, 1, 128, 128, 128]

#     return roi_whole

def get_data_batch_ROI_128(data_batch, device, need_lab = False):

    if torch.cuda.is_available():
        inputs = Variable(data_batch["ROI"].float().to(device))
        if need_lab:
            label = Variable(data_batch["label"].to(device))
    else:
        inputs = Variable(data_batch["ROI"].float())
        if need_lab:
            label = Variable(data_batch["label"])
    
    #assert inputs.shape == masks.shape, "something wrong with load the data, two tensor sizes do not match"

    # get roi and crop mask
    roi_whole = inputs #inputs * masks
    
    # normalize to -1,1, batch by batch
    for i in range(roi_whole.shape[0]):
        roi_whole[i,:,:,:] = normalize2(roi_whole[i,:,:,:])
    #print("rois max min pixel: ", torch.max(rois), torch.min(rois))
    #roi_01 = (roi_whole-torch.min(roi_whole))/(torch.max(roi_whole)-torch.min(roi_whole)) if torch.max(roi_whole) != torch.min(roi_whole) else roi_whole
    #rois = (2 * roi_01) - 1 # [-1, 1]

    #if rois_new.shape[-1] == 155: # channel is in the last dim, switch to the front
    roi_whole = roi_whole.permute((0, 3, 1, 2)).contiguous()
    #assert rois_new.shape[1:] == (slices, target, target), "roi_small, reduce slices go wrong"
    
    roi_whole = roi_whole.unsqueeze(1) # [batch, 1, 128, 128, 128]

    if need_lab:
        return roi_whole, label
    return roi_whole


def get_data_batch_ROI_128_pl(data_batch, need_lab = False):

    if torch.cuda.is_available():
        inputs = Variable(data_batch["ROI"].float())
        if need_lab:
            label = Variable(data_batch["label"])
    else:
        inputs = Variable(data_batch["ROI"].float())
        if need_lab:
            label = Variable(data_batch["label"])
    
    #assert inputs.shape == masks.shape, "something wrong with load the data, two tensor sizes do not match"

    # get roi and crop mask
    roi_whole = inputs #inputs * masks
    
    # normalize to -1,1, batch by batch
    for i in range(roi_whole.shape[0]):
        roi_whole[i,:,:,:] = normalize2(roi_whole[i,:,:,:])
    #print("rois max min pixel: ", torch.max(rois), torch.min(rois))
    #roi_01 = (roi_whole-torch.min(roi_whole))/(torch.max(roi_whole)-torch.min(roi_whole)) if torch.max(roi_whole) != torch.min(roi_whole) else roi_whole
    #rois = (2 * roi_01) - 1 # [-1, 1]

    #if rois_new.shape[-1] == 155: # channel is in the last dim, switch to the front
    roi_whole = roi_whole.permute((0, 3, 1, 2)).contiguous()
    #assert rois_new.shape[1:] == (slices, target, target), "roi_small, reduce slices go wrong"
    
    roi_whole = roi_whole.unsqueeze(1) # [batch, 1, 128, 128, 128]

    if need_lab:
        return roi_whole, label
    return roi_whole