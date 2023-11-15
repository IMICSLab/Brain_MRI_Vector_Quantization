
# inference transformer model

import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../")

#from dcgan.vqgan_transformer_lat4 import VQGANTransformer
from model.vqgan_transformer import VQGANTransformer
from model.transformer import sample_with_past
from utils.get_data import normalize

z_dim = (8,8,8) #(4,4,4)
N = 10 # how many images
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
par_dir = "./transformer_dir"
save_dir_name = "inference"
if not os.path.exists(par_dir):
    os.mkdir(par_dir)
if not os.path.exists(os.path.join(par_dir, save_dir_name)):
    os.mkdir(os.path.join(par_dir, save_dir_name))
    

#print("finish loading data")
enc_dec_weights = "./vqgan_dir/pretrained_model/epoch_3999.pt"
device = "cuda:0"

transformer_model = VQGANTransformer(z_dim, enc_dec_weights, device).to(device)
#print("transformer model paras: ", sum(p.numel() for p in transformer.parameters() if p.requires_grad))

chk_point = "epoch_1499"
print(chk_point)
transformer_model.load_state_dict(torch.load(os.path.join(par_dir, "pretrained_model/{}.pt".format(chk_point))))
#transformer.load_state_dict(torch.load(os.path.join("Z:/Simon/3dgan_res/BraTS_brainOnly/trail11_v4_stage2/pretrained_model", "epoch_999.pt")))
print("Loaded state dict of Transformer")

transformer_model.eval()
BATCH_SIZE = 1 # how many images in one inference run
start = time.time()
#res = []
TYPE = "fixed"
#top_p = 1

print("start with 0 or random: ", TYPE, "start with 0 if fixed")
#generated = torch.empty(0,128,128,128).cuda()

for i in range(N): # for generated images
    
    if TYPE == "random":
        start_at = np.random.randint(0, 512)
    elif TYPE == "fixed":
        start_at = 0
    else:
        raise ValueError("TYPE should be random or fixed")
    #print(start_at)
    with torch.no_grad():
        start_indices = torch.zeros((BATCH_SIZE, 0)).long().to(device)
        sos_tokens = torch.ones(start_indices.shape[0], 1) * start_at
        #print(sos_tokens)
        sos_tokens = sos_tokens.long().to(device)
        sample_indices = transformer_model.sample_inf(start_indices, sos_tokens, steps=np.prod(z_dim), top_k=512, top_p=0.95, sample = True, temperature=1)
        # or
        #sample_indices = sample_with_past(sos_tokens, transformer_model.transformer, steps=np.prod(z_dim), temperature=1., sample_logits=True, top_k=256, top_p=0.95)
        
        _,sampled_imgs = transformer_model.z_to_image(sample_indices)

        sampled_imgs = sampled_imgs.squeeze(0).squeeze(0)
        sampled_imgs = (sampled_imgs + 1) / 2 # to 0,1 should be in [128, 128, 128], [d, t, w]
        sampled_imgs = sampled_imgs.permute(1,2,0).contiguous()#.squeeze(0) # to channel last
        #print("max: ", torch.max(sampled_imgs), "min: ", torch.min(sampled_imgs))
        
    #generated = torch.cat((generated, sampled_imgs.unsqueeze(0)), dim = 0)

    torch.save(sampled_imgs.detach().cpu(), os.path.join(par_dir, save_dir_name, "augmented_mri_vol{}.pt".format(i))) # sX: start with index X

print("time used: ", (time.time() - start), "time per image: ", (time.time() - start)/N)
