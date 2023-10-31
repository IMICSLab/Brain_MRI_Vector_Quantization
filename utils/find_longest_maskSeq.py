# find the longest segmentation mask sequence per patients

import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
sys.path.append("/home/szhou/pLGG")

from dcgan.model_v6 import *

def find_maskSeq(img):
    num = 0
    # get the longest continous mask seq
    for i in range(img.shape[-1]):
        if np.any(img[:,:,i]):
            num += 1
    return num


def load_object(filename):
  # opening a file in read, binary form
    file = open(filename, 'rb')

    ob = pickle.load(file)

    # close the file
    file.close()
    return ob

def get_vis(file, target_dir):
    
    img = file #torch.load(file).numpy()
    print(np.max(img), np.min(img))
    fig = plt.figure(figsize = (8,8))
    plt.axis("off")

    ims = [[plt.imshow(img[:,:,i], cmap="gray", animated = True)] for i in range(img.shape[-1])]
    ani = animation.ArtistAnimation(fig, ims, interval = 1000, repeat_delay = 1000, blit = True)
    with open(os.path.join(target_dir, "mask68.html"), "w") as w:
        w.write(ani.to_jshtml())

if __name__ == "__main__":
    msk_dir = "./Simon/data/BraTS/Masks.p"
    data_dir = "./Simon/data/BraTS/imgs.p"
    mask = load_object(msk_dir)
    img = load_object(data_dir)
    mask_np = np.array(mask)
    img_np = np.array(img)
    del mask, img
    print(mask_np.shape)
    # record min max coordinates of bounding box
    anw = -1

    rois = mask_np * img_np
    for im in range(rois.shape[0]):
        temp_im = rois[im,:,:,:]
        cur_num = find_maskSeq(temp_im)
        print("{}th patient, mask seq: {}".format(im, cur_num))
        anw = max(anw, cur_num)
        if im == 68:
            get_vis(temp_im, "/home/szhou/pLGG/utils")
            print("{}th patient saved".format(im))
    
    print("longest mask seq of all patients: {}".format(anw))
    # width = ROW_MAX - ROW_MIN + 1
    # height = COL_MAX - COL_MIN + 1
    # print("width & height: ", width, height)

    # target = 0
    # if width <= 64 and height <= 64:
    #     target = 64
    # else:
    #     target = 176

    # temp_img1 = torch.from_numpy(mask_np[20,:,:,76])
    # temp_img2 = torch.from_numpy(mask_np[5,:,:,55])

    # temp_img1 = temp_img1[ROW_MIN:ROW_MAX+1, COL_MIN:COL_MAX]
    # print("cropped temp1", temp_img1.shape)
    # temp_img2 = temp_img2[ROW_MIN:ROW_MAX+1, COL_MIN:COL_MAX]
    # print("cropped temp2", temp_img2.shape)

    # # if target < height or target < width:
    # #     temp_img1 = temp_img1[:,0:-1]
    # #     temp_img2 = temp_img2[:,0:-1]
    # img1_pad = pad_image(temp_img1, target)
    # img2_pad = pad_image(temp_img2, target)
    # print(img1_pad.shape, img2_pad.shape)
    # # plt.imshow(mask_np[20,:,:,76], "gray")
    # # plt.savefig("./extreme_case_col.jpg")
    # # plt.imshow(mask_np[5,:,:,55], "gray")
    # # plt.savefig("./extreme_case_row.jpg")
    # # plt.imshow(img1_pad, "gray")
    # # plt.savefig("./extreme_case_col_pad.jpg")
    # # plt.imshow(img2_pad, "gray")
    # # plt.savefig("./extreme_case_row_pad.jpg")
    # ####
    # print("now test the GAN model")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # z_dim = 128
    # netG = Generator3D(z_dim, 176).to(device)
    # netD = Discriminator3D().to(device)
    # noise = torch.randn(1, z_dim, 1, 1, 1, device=device)

    # gen_o = netG(noise)
    # dis_o = netD(gen_o.detach())