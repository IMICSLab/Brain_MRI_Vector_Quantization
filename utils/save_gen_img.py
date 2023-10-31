# plot some fake image during traininig

import torch
import numpy as np
import matplotlib.pyplot as plt

def show(img, ind, res_dir, trail, fname, gray_scale = True, img_save_dir = "gen_img"):
    plt.figure(figsize = (10,10))
    for i in range(16):
        ax = plt.subplot(4, 4, i%16+1)
        fake_img = img[int(ind[i]), :,:].numpy()
        fake_img = (fake_img + 1) / 2
        if gray_scale:
            plt.imshow(fake_img, "gray")
        else:
            plt.imshow(fake_img)
        plt.title("Fake Slice {}".format(int(ind[i])))
        plt.axis("off")
    if trail == "": # for pytorch lightning
        plt.savefig(res_dir + "/{}/".format(img_save_dir) + fname)
    else:
        plt.savefig(res_dir + "/trail{}/{}/".format(trail, img_save_dir) + fname)
    plt.close()