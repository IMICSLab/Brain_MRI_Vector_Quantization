import os
import numpy as np
from tqdm import tqdm
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import utils as vutils
import sys
sys.path.append("../")

import meta_config as mc
from utils import set_seed
from utils.get_data import *
from model.vqgan_transformer import VQGANTransformer
from utils.plot_loss import plot_loss
from utils.save_gen_img import show

set_seed.random_seed(0, True)

data_dir = mc.data_dir

res_dir = mc.save_dir

print("loading data: ")
Rois = torch.load(os.path.join(data_dir, "data.pt"))
print("finish loading.")

BATCH_SIZE = mc.bs
z_dim = mc.z_dim #(4,4,4) # last conv dim, bottleneck dim, for decoder
EPOCH = mc.epoch
TRAIL = mc.trail

if not os.path.exists(res_dir + "/trail{}".format(TRAIL)):
    os.makedirs(res_dir + "/trail{}".format(TRAIL))


total = list(range(Rois.shape[0]))
val_ind = list(np.random.choice(total, size=BATCH_SIZE, replace=False))
target_valid = [i for i in total if i not in val_ind]
# get dataset
#dataset = get_data_loader(Imgs, Msks, pIDs, target_valid, shuffle = True, norm = False)
dataset = get_data_loader_128(Rois, target_valid, shuffle = True, norm = False)
val_dataset = get_data_loader_128(Rois, val_ind, shuffle = False, norm = False)


train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# autoencoder weights
model_weights = mc.ae_weights
n_codes = mc.n_codes
transformer_model = VQGANTransformer(z_dim, n_codes, model_weights, device)
lr = mc.lr

def configure_optimizers(model):
    decay, no_decay = set(), set()
    whitelist_weight_modules = (nn.Linear, )
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

    for mn, m in model.transformer.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn

            if pn.endswith("bias"):
                no_decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    no_decay.add("pos_emb")

    param_dict = {pn: p for pn, p in model.transformer.named_parameters()}

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95)) # 4.5e-06
    return optimizer

optim_transformer = configure_optimizers(transformer_model)
step_trans = optim.lr_scheduler.CosineAnnealingLR(optim_transformer, T_max=EPOCH)
print("all preparation finished")
#sys.exit()
print("Starting Training Loop...")

use_mixed_precision = True
scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

ce_losses = []

for epoch in range(EPOCH):
    print("new epoch starts")
    
    epoch_start_time = time.time()
    batch_start_time = time.time()
    
    ce_loss_iter = 0
    transformer_model.train()
    for i, data_b in enumerate(train_loader):
        batch_duration = 0
        optim_transformer.zero_grad()
        
        data = get_data_batch_ROI_128(data_b, device)
        
        real_cpu = data
        #print("real cpu: ", real_cpu.shape)
        b_size = real_cpu.size(0)
        
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            _,logits, targets = transformer_model(data)
            loss = F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1)) # or compute the loss on masked
        
        scaler.scale(loss).backward(retain_graph=True)
        scaler.step(optim_transformer)
        scaler.update()
        
        ce_loss_iter += loss.item()
    
    #step_trans.step()
    ce_losses.append(ce_loss_iter / len(train_loader))

    if i % 5 == 0:
        print("[%d/%d] batches done!\n" % (i,
                                            len(dataset)//BATCH_SIZE))
        batch_end_time = time.time()
        batch_duration = batch_duration + batch_end_time - batch_start_time
        print("Training time for", i, "batches: ", batch_duration / 60,
                " minutes.")
    
    print('[%d/%d]\tEntropy: %.4f'
        % (epoch, EPOCH, ce_losses[-1]))
    
    # logging res
    if not os.path.exists(res_dir + "/trail{}/rec_img".format(TRAIL)):
        os.makedirs(res_dir + "/trail{}/rec_img".format(TRAIL))

    if not os.path.exists(res_dir + "/trail{}/pretrained_model".format(TRAIL)):
        os.makedirs(res_dir + "/trail{}/pretrained_model".format(TRAIL))
    
    # validation
    if (epoch+1) % 10 == 0:
        transformer_model.eval()
        with torch.no_grad():
            for i, data_b in enumerate(val_loader):
                data = get_data_batch_ROI_128(data_b, device)
                rand_ind = np.random.randint(0, BATCH_SIZE)
                sample_data = data[rand_ind].unsqueeze(0)
                _, rec_and_pred_img = transformer_model.log_images(sample_data)
                rec_and_pred_img = rec_and_pred_img.detach().cpu()
                # each image should be 1,1,128,128,128
                orig = rec_and_pred_img[0].squeeze(0).squeeze(0)
                rec = rec_and_pred_img[1].squeeze(0).squeeze(0)
                # image generated with half indices masked
                half_random = rec_and_pred_img[2].squeeze(0).squeeze(0)
                # image generated with full indices masked (only provide 1)
                full_random = rec_and_pred_img[3].squeeze(0).squeeze(0)
                full_random_mostlike = rec_and_pred_img[4].squeeze(0).squeeze(0)
                
                assert orig.shape == rec.shape == half_random.shape == full_random.shape == full_random_mostlike.shape, "logging images have different shape!"
                
                rand_ind = np.array(list(range(50, 50+16))) #np.random.choice(orig.shape[0], size=16, replace=False)

                show(orig, rand_ind, res_dir, TRAIL, "orig_img_epoch{}_img.png".format(epoch), img_save_dir = "rec_img")
                show(rec, rand_ind, res_dir, TRAIL, "rec_img_epoch{}_img.png".format(epoch), img_save_dir = "rec_img")
                show(half_random, rand_ind, res_dir, TRAIL, "half_rand_img_epoch{}_img.png".format(epoch), img_save_dir = "rec_img")
                show(full_random, rand_ind, res_dir, TRAIL, "generated_img_formDist_epoch{}_img.png".format(epoch), img_save_dir = "rec_img")
                show(full_random_mostlike, rand_ind, res_dir, TRAIL, "generated_img_mostlike_epoch{}_img.png".format(epoch), img_save_dir = "rec_img")
        
        # half_random = (half_random + 1) / 2
        # half_random = half_random.permute(1,2,0).contiguous()
        # print("half random range: ", torch.min(half_random), torch.max(half_random))
        # full_random = (full_random + 1) / 2
        # full_random = full_random.permute(1,2,0).contiguous()
        # print("full random range: ", torch.min(full_random), torch.max(full_random))
        # torch.save(half_random, res_dir + "/trail{}".format(TRAIL) + "/half_random_generates_epoch{}.pt".format(epoch))
        # torch.save(full_random, res_dir + "/trail{}".format(TRAIL) + "/full_random_generates_epoch{}.pt".format(epoch))
        # torch.save(transformer_model.state_dict(), res_dir + "/trail{}/pretrained_model/".format(TRAIL) + "epoch_{}.pt".format(epoch))
        
        del sample_data, rec_and_pred_img, orig, rec, half_random, full_random, full_random_mostlike
        
    plot_loss(ce_losses, "Cross Entropy Loss during training",
                    res_dir + "/trail{}/".format(TRAIL), "ce_loss")
    
    #del sample_data, rec_and_pred_img, orig, rec, half_random, full_random
