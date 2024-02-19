
# inference transformer model

import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../")

from model.vqgan_transformer_cond import VQGANTransformer
from model.transformer import sample_with_past
from utils.ssim_3d import ssim3D
from utils.get_data import normalize
from configs.sample_cond_configs import sampling_config


# =================================config=====================================
BATCH_SIZE = sampling_config["bs"]
z_dim = sampling_config["latent_dim"]
N = sampling_config["num_sample"]
CLS_LAB = sampling_config["cls_lab_indicator"]
class_map = {0:"LGG", 1:"HGG"}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
par_dir = sampling_config["parent_save_dir"]
save_dir_name = sampling_config["save_folder"]
num_class = sampling_config["num_classes"]
codebook_dim = sampling_config["codebook_dim"]
n_codes = sampling_config["n_codes"]
vqgan_model_weights = sampling_config["vqgan_model_weights"]
sample_chkpt = sampling_config["chkpt"]
topp = sampling_config["topp"]
topk = sampling_config["topk"]
temperature = sampling_config["temperature"]
# =============================================================================

if not os.path.exists(par_dir):
    os.mkdir(par_dir)
if not os.path.exists(os.path.join(par_dir, save_dir_name)):
    os.mkdir(os.path.join(par_dir, save_dir_name))
    
for c in CLS_LAB:
    class_specific_dir = "class_{}".format(class_map[c])
    if not os.path.exists(os.path.join(par_dir, save_dir_name, class_specific_dir)):
        os.mkdir(os.path.join(par_dir, save_dir_name, class_specific_dir))

#print("finish loading data")
img_trans = VQGANTransformer(class_num = num_class, z_dim = z_dim, codebook_dim = codebook_dim, n_codes = n_codes, model_weights = vqgan_model_weights, device = device).to(device)
chk_point = sample_chkpt
img_trans.load_state_dict(torch.load(os.path.join(par_dir, "pretrained_model/{}.pt".format(chk_point))))
#transformer.load_state_dict(torch.load(os.path.join("Z:/Simon/3dgan_res/BraTS_brainOnly/trail11_v4_stage2/pretrained_model", "epoch_999.pt")))
print("Loaded state dict of Transformer")

img_trans.eval()

print("class cond generation")


start = time.time()

# start generating
for c in CLS_LAB:
    class_specific_dir = "class_{}".format(class_map[c])
    for i in range(N): # for generated images
        #print(start_at)
        print("sample {}".format(i))
        with torch.no_grad():
            start_indices = torch.zeros((BATCH_SIZE, 0)).long().to(device)
            temp_c = c + img_trans.codebook_and_mask_dim
            class_lab_ind = torch.tensor([temp_c]).repeat(BATCH_SIZE,1).to(device)
            #start_indices = torch.cat((start_indices, class_lab_ind), dim = 1)
            #print(start_indices)
            sample_indices = img_trans.sample_inf(start_indices, class_lab_ind, steps = np.prod(z_dim), top_k=topk, top_p=topp, temperature = temperature) #sample_with_past(start_indices, img_trans.transformer, steps=np.prod(z_dim), temperature=2, top_k=100, top_p=0.95)
            if torch.min(sample_indices) < 0 or torch.max(sample_indices) > img_trans.n_codes-1:
                sample_indices = torch.clamp(sample_indices, min = 0, max = img_trans.n_codes-1)
            _,sampled_imgs = img_trans.z_to_image(sample_indices)
            sampled_imgs = sampled_imgs.squeeze(0).squeeze(0)
            sampled_imgs = (sampled_imgs + 1) / 2 # to range(0,1) should be in [128, 128, 128], [d, t, w]
            sampled_imgs = sampled_imgs.permute(1,2,0).contiguous()#.squeeze(0) # to channel last
        print("max: {},min: {}".format(torch.max(sampled_imgs), torch.min(sampled_imgs)))
        print("complete")
        torch.save(sampled_imgs.detach().cpu(), os.path.join(par_dir, save_dir_name, class_specific_dir, "augmented_mri_vol{}.pt".format(i)))


end_time = time.time()
print("time used: {}".format(end_time - start), flush = True)
print("average per sample is seconds: {}".format((end_time - start)/(len(CLS_LAB)*N)), flush = True)
