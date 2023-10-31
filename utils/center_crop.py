# center crop ROI
# maske ROI at the center of the image
# target size: 128x128x128

import torch
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

import sys
sys.path.append("/home/szhou/pLGG")


def get_local_min_max(mri_slice, rmax, rmin, cmax, cmin):
    # greedy to find the largest bounding box of seg masks for each mri slice
    for i in range(mri_slice.shape[0]): # scanning rows
        row = mri_slice[i,:]
        if not np.all(row == 0.):
            rmin = min(rmin, i)
            rmax = max(rmax, i)

    for j in range(mri_slice.shape[1]):
        col = mri_slice[:,j]
        if not np.all(col == 0.):
            cmin = min(cmin, j)
            cmax = max(cmax, j)
    return rmax, rmin, cmax, cmin


# def pad_image(img, target:int):
#     '''
#     img: mri slice
#     target: target size, int
#     '''
#     row_dif = target - img.shape[0]
#     if row_dif % 2 == 0:
#         one_pad = row_dif // 2
#         row_pad1 = row_pad2 = one_pad

#     else:
#         one_pad = row_dif // 2
#         row_pad1 = one_pad
#         row_pad2 = one_pad+1
    
#     col_dif = target - img.shape[1]
#     if col_dif % 2 == 0:
#         one_pad = col_dif // 2
#         col_pad1 = col_pad2 = one_pad
#     else:
#         one_pad = col_dif // 2
#         col_pad1 = one_pad
#         col_pad2 = one_pad+1

#     img_pad = F.pad(img, (col_pad1, col_pad2, row_pad1, row_pad2))
#     return img_pad


def center_pad(img, target:int):
    '''
    padding image so that it is located in the center of 128*128 image
    img: mri slice
    target: target size, int. 2^x
    '''

    row_mid = img.shape[0] // 2
    col_mid = img.shape[1] // 2

    col_pad1 = target // 2 - col_mid
    col_pad2 = col_pad1 if img.shape[1] % 2 == 0 else col_pad1 - 1

    row_pad1 = target // 2 - row_mid
    row_pad2 = row_pad1 if img.shape[0] % 2 == 0 else row_pad1 - 1
    # row_dif = target - img.shape[0]
    # if row_dif % 2 == 0:
    #     one_pad = row_dif // 2
    #     row_pad1 = row_pad2 = one_pad

    # else:
    #     one_pad = row_dif // 2
    #     row_pad1 = one_pad
    #     row_pad2 = one_pad+1
    
    # col_dif = target - img.shape[1]
    # if col_dif % 2 == 0:
    #     one_pad = col_dif // 2
    #     col_pad1 = col_pad2 = one_pad
    # else:
    #     one_pad = col_dif // 2
    #     col_pad1 = one_pad
    #     col_pad2 = one_pad+1

    img_pad = F.pad(img, (col_pad1, col_pad2, row_pad1, row_pad2))
    return img_pad

def center_crop_roi(mri):
    '''
    mri: mri torch file, assume shape is 240*240*155
    return size 128*128*155
    this should be unnormalized one
    '''
    rmax = cmax = -1
    rmin = cmin = int(1e9)
    new_mri = torch.zeros(128, 128, mri.shape[-1]) # all black
    for s in range(mri.shape[-1]):
        if torch.any(mri[:,:,s]):
            temp_s = mri[:,:,s].numpy()
            rmax, rmin, cmax, cmin = get_local_min_max(temp_s, rmax, rmin, cmax, cmin)
            print(rmax, rmin, cmax, cmin)
            crop_roi = torch.from_numpy(temp_s[rmin:rmax+1, cmin:cmax+1])
            print(crop_roi.shape)
            # plt.imshow(crop_roi, "gray")
            # plt.show()
            #print("crop mask size: ", crop_roi.shape)
            center_crop_roi = center_pad(crop_roi, 128)
            print(center_crop_roi.shape)
            # plt.imshow(center_crop_roi, "gray")
            # plt.show()
            new_mri[:,:,s] = center_crop_roi
    return new_mri


def find_maskSeq(img):
    num = 0
    # get the longest continous mask seq
    for i in range(img.shape[-1]):
        if torch.any(img[:,:,i]):
            num += 1
    return num


def mask_start_end(img):
    '''
    return the start ind of mask and end ind, in order to get the reduced slices
    this is for original data, intensity in [0,1] or [0, max(original image)]
    '''
    assert img.shape[-1] == 155, "shape is wrong here"
    start = -1
    end = -1
    for i in range(img.shape[-1]):
        if torch.any(img[:,:,i]):
            start = i
            break
    
    for j in range(img.shape[-1]-1, -1, -1):
        if torch.any(img[:,:,j]):
            end = j
            break
    
    return start, end

def process_slice(cropped_mri):
    '''
    cropped mri: cropped mri with size 128*128*155, torch file
    # will be in the range of [0,1]
    '''
    nonzero_num = find_maskSeq(cropped_mri)
    msk_start, msk_end = mask_start_end(cropped_mri)
    #print("mask num", nonzero_num)
    if nonzero_num > 128:
        print("there are some patients with mask >= 128")
        print("now processing those")
        
        to_delete1 = (nonzero_num - 128) // 2 if (nonzero_num - 128) % 2 == 0 else (nonzero_num - 128) // 2 + 1
        to_delete2 = (nonzero_num - 128) // 2
        
        crop_slice_mri = cropped_mri[:,:,msk_start+to_delete1:msk_end+1+to_delete2]
        return crop_slice_mri
           
    #msk_start, msk_end = mask_start_end(cropped_mri)

    if msk_end+1-msk_start > nonzero_num:
        msk_end -= (msk_end+1-msk_start-nonzero_num)
        
    to_add1 = (128 - nonzero_num) // 2
    to_add2 = to_add1 if (128 - nonzero_num) % 2 == 0 else to_add1 + 1

    #print("mask start, mask end: {}, {}".format(msk_start, msk_end))

    crop_slice_mri = cropped_mri[:,:,msk_start:msk_end+1]
    #print(crop_slice_mri.shape)
    prev_ = torch.zeros(128, 128, to_add1)
    suf_ = torch.zeros(128, 128, to_add2)

    crop_slice_mri = torch.cat((prev_, crop_slice_mri, suf_), dim = -1)
    #print(crop_slice_mri.shape)
    assert crop_slice_mri.shape == (128, 128, 128), "final preproc shape is not equal to 128,128,128"
    
    return crop_slice_mri


def process_sliceV2(mri):
    '''
    deal with the 176*176*104 generated image
    data will be in the range of [0,1]
    '''
    left = 128 - mri.shape[-1]
    addition = torch.zeros(128, 128, left)
    mri = torch.cat((mri, addition), dim = -1)
    return mri
    
    
def load_object(filename):
  # opening a file in read, binary form
    file = open(filename, 'rb')

    ob = pickle.load(file)

    # close the file
    file.close()
    return ob


def get_vis(file, target_dir, fname):
    
    img = file #torch.load(file).numpy()
    print(torch.max(img), torch.min(img))
    fig = plt.figure(figsize = (5,5))
    plt.axis("off")

    ims = [[plt.imshow(img[:,:,i], cmap="gray", animated = True)] for i in range(img.shape[-1])]
    ani = animation.ArtistAnimation(fig, ims, interval = 1000, repeat_delay = 1000, blit = True)
    with open(os.path.join(target_dir, "{}.html".format(fname)), "w") as w:
        w.write(ani.to_jshtml())


if __name__ == "__main__":
    fusion = "/hpf/largeprojects/fkhalvati/Simon/data/pLGG/fusion_ROIs.pt"
    mutation = "/hpf/largeprojects/fkhalvati/Simon/data/pLGG/mutation_ROIs.pt"

    all_file = [fusion, mutation]
    #for i in range(len(hgg_)):
        #print("patient {}".format(i))

    for f in all_file:
        print("processing file: {}".format(f))
        test_file = torch.load(f)
        cropped_roi_t = center_crop_roi(test_file)
        cropped_roi = process_slice(cropped_roi_t)

        print(cropped_roi.shape)
        print(torch.max(cropped_roi), torch.min(cropped_roi))
        torch.save(cropped_roi, "/home/szhou/pLGG/data/pLGG/{}_128.pt".format(f.split("/")[-1].split(".")[0]))
    #test_file = torch.from_numpy(np.array(load_object(hgg)[16]))
    #get_vis(test_file, "Z:/Simon", "cropped_roi_vis_randLGG")

    