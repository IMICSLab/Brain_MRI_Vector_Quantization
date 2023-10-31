# find the largest bounding box of segmentation mask per patients

import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/szhou/pLGG")

from dcgan.model_v6 import *

def find_bbox(img, img_ind, s_ind):
    # greedy to find the largest bounding box of seg masks
    global ROW_MAX, ROW_MIN, COL_MAX, COL_MIN
    for i in range(img.shape[0]): # scanning rows
        row = img[i,:]
        if not np.all(row == 0.):
            # if i < ROW_MIN:
            #     ROW_MIN = i
            #     print("image ind, min row number at which slice, ", img_ind, i, s_ind)
            # if i > ROW_MAX:
            #     ROW_MAX = i
            #     print("image ind, max row number at which slice, ", img_ind, i, s_ind)
            ROW_MIN = min(ROW_MIN, i)
            ROW_MAX = max(ROW_MAX, i)

    for j in range(img.shape[1]):
        col = img[:,j]
        if not np.all(col == 0.):
            # if j < COL_MIN:
            #     COL_MIN = j
            #     print("image ind, min col number at which slice, ", img_ind, j, s_ind)
            # if j > COL_MAX:
            #     COL_MAX = j
            #     print("image ind, max col number at which slice, ", img_ind, j, s_ind)
            COL_MIN = min(COL_MIN, j)
            COL_MAX = max(COL_MAX, j)

def get_desired_size(rmin, rmax, cmin, cmax):

    width = rmax - rmin + 1
    height = cmax - cmin + 1
    target = 0
    if width <= 64 and height <= 64:
        target = 64
    else:
        target = 176

    return target

def pad_image(img, target):
    row_dif = target - img.shape[0]
    if row_dif % 2 == 0:
        one_pad = row_dif // 2
        row_pad1 = row_pad2 = one_pad

    else:
        one_pad = row_dif // 2
        row_pad1 = one_pad
        row_pad2 = one_pad+1
    
    col_dif = target - img.shape[1]
    if col_dif % 2 == 0:
        one_pad = col_dif // 2
        col_pad1 = col_pad2 = one_pad
    else:
        one_pad = col_dif // 2
        col_pad1 = one_pad
        col_pad2 = one_pad+1

    img_pad = nn.functional.pad(img, (col_pad1, col_pad2, row_pad1, row_pad2))
    return img_pad

def load_object(filename):
  # opening a file in read, binary form
    file = open(filename, 'rb')

    ob = pickle.load(file)

    # close the file
    file.close()
    return ob

if __name__ == "__main__":
    data_dir = "./Simon/data/BraTS/Masks.p"
    mask = load_object(data_dir)
    mask_np = np.array(mask)
    del mask
    print(mask_np.shape)
    # record min max coordinates of bounding box
    ROW_MAX = -1
    ROW_MIN = int(1e9)
    COL_MAX = -1
    COL_MIN = int(1e9)

    for im in range(mask_np.shape[0]):
        temp_im = mask_np[im,:,:,:]
        for s in range(temp_im.shape[-1]):
            find_bbox(temp_im[:,:,s], im, s)
    
    print(ROW_MIN, ROW_MAX, COL_MIN, COL_MAX)
    width = ROW_MAX - ROW_MIN + 1
    height = COL_MAX - COL_MIN + 1
    print("width & height: ", width, height)

    target = 0
    if width <= 64 and height <= 64:
        target = 64
    else:
        target = 176

    temp_img1 = torch.from_numpy(mask_np[20,:,:,76])
    temp_img2 = torch.from_numpy(mask_np[5,:,:,55])

    temp_img1 = temp_img1[ROW_MIN:ROW_MAX+1, COL_MIN:COL_MAX]
    print("cropped temp1", temp_img1.shape)
    temp_img2 = temp_img2[ROW_MIN:ROW_MAX+1, COL_MIN:COL_MAX]
    print("cropped temp2", temp_img2.shape)

    # if target < height or target < width:
    #     temp_img1 = temp_img1[:,0:-1]
    #     temp_img2 = temp_img2[:,0:-1]
    img1_pad = pad_image(temp_img1, target)
    img2_pad = pad_image(temp_img2, target)
    print(img1_pad.shape, img2_pad.shape)
    # plt.imshow(mask_np[20,:,:,76], "gray")
    # plt.savefig("./extreme_case_col.jpg")
    # plt.imshow(mask_np[5,:,:,55], "gray")
    # plt.savefig("./extreme_case_row.jpg")
    # plt.imshow(img1_pad, "gray")
    # plt.savefig("./extreme_case_col_pad.jpg")
    # plt.imshow(img2_pad, "gray")
    # plt.savefig("./extreme_case_row_pad.jpg")
    ####
    print("now test the GAN model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z_dim = 128
    netG = Generator3D(z_dim, 176).to(device)
    netD = Discriminator3D().to(device)
    noise = torch.randn(1, z_dim, 1, 1, 1, device=device)

    gen_o = netG(noise)
    dis_o = netD(gen_o.detach())