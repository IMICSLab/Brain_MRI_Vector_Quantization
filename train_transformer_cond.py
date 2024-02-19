import os
import numpy as np
from tqdm import tqdm
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
import sys
sys.path.append("../")

import meta_config as c
from utils import set_seed
from utils.get_data import *
from model.vqgan_transformer_cond import VQGANTransformer
from utils.plot_loss import plot_loss
from utils.save_gen_img import show
from configs.vqgan_cond_configs import models_config

set_seed.random_seed(0, True)

data_dir = c.data_dir

res_dir = c.res_dir_imgOnly_masked

#data_dir = "Z:/Datasets/MedicalImages/BrainData/SickKids/LGG/AI_ready"
print("loading data: ")
RoisH = torch.load(os.path.join(data_dir, "HGG_ROI_128_train.pt"))
RoisL = torch.load(os.path.join(data_dir, "LGG_ROI_128_train.pt"))
labelsL = torch.zeros(RoisL.shape[0])
labelsH = torch.ones(RoisH.shape[0])

print("finish loading.")

Rois = torch.cat((RoisL, RoisH), dim = 0)
labls = torch.cat((labelsL, labelsH))

assert Rois.shape[0] == labls.shape[0], "Rois and labels have different length, what happened?"
print(Rois.shape, labls.shape)


# ======================== configs ==========================
BATCH_SIZE = models_config["bs"]
z_dim = models_config["latent_dim"]
EPOCH = models_config["epochs"]
TRAIL = "_cond_is_abl" # condition on class label
model_weights = models_config["vqgan_model_weights"]
num_class = models_config["class_num"]
codebook_dim = models_config["codebook_dim"]
n_codes = models_config["codebook_n_embed"]
mask_ratio = models_config["mask_ratio"]
topk_ratio = models_config["topk_ratio"]
lr_decay = models_config["lr_decay"]
lr = models_config["lr"]
save_plot_every = models_config["save_plot_every"]
# ==========================================================

if not os.path.exists(res_dir + "/trail{}".format(TRAIL)):
    os.makedirs(res_dir + "/trail{}".format(TRAIL))


total = list(range(Rois.shape[0]))
val_ind = list(np.random.choice(total, size=BATCH_SIZE, replace=False))
print(val_ind)
target_valid = [i for i in total if i not in val_ind]
# get dataset
#dataset = get_data_loader(Imgs, Msks, pIDs, target_valid, shuffle = True, norm = False)
dataset = get_data_loader_128_cond(Rois, labls, target_valid, shuffle = True, norm = False)
val_dataset = get_data_loader_128_cond(Rois, labls, val_ind, shuffle = False, norm = False)

train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=BATCH_SIZE, # droplast set to True for VAE
                                            shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=BATCH_SIZE, # droplast set to True for VAE
                                            shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transformer_model = VQGANTransformer(class_num = num_class, z_dim = z_dim, codebook_dim = codebook_dim, n_codes = n_codes, mask_ratio = mask_ratio, topk_ratio = topk_ratio, model_weights = model_weights, device = device)


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
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95)) # 4.5e-6
    return optimizer

optim_transformer = configure_optimizers(transformer_model)
if lr_decay:
    stepG = torch.optim.lr_scheduler.CosineAnnealingLR(optim_transformer, T_max=EPOCH, eta_min = 0.00000001)

print("all preparation finished")
#sys.exit()
print("Starting Training Loop...")

use_mixed_precision = True
scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

def save_log_images(res_img, cls, rand_ind, epoch, res_dir, TRAIL, val = False):
    rec_and_pred_img = res_img.detach().cpu()
    # each image should be 1,1,128,128,128
    orig = rec_and_pred_img[0].squeeze(0).squeeze(0)
    rec = rec_and_pred_img[1].squeeze(0).squeeze(0)
    # image generated with half indices masked
    half_random = rec_and_pred_img[2].squeeze(0).squeeze(0)
    # image generated with full indices masked (only provide 1)
    full_random = rec_and_pred_img[3].squeeze(0).squeeze(0)
    cls_label = int(cls.detach().cpu().numpy()[0])
    if val:
        show(orig, rand_ind, res_dir, TRAIL, "orig_img_class{}_epoch{}_val.png".format(cls_label, epoch), img_save_dir = "rec_img")
        show(rec, rand_ind, res_dir, TRAIL, "rec_img_class{}_epoch{}_val.png".format(cls_label,epoch), img_save_dir = "rec_img")
        show(half_random, rand_ind, res_dir, TRAIL, "half_rand_img_class{}_epoch{}_val.png".format(cls_label,epoch), img_save_dir = "rec_img")
        show(full_random, rand_ind, res_dir, TRAIL, "generated_img_class{}_epoch{}_val.png".format(cls_label,epoch), img_save_dir = "rec_img")
    else:
        show(orig, rand_ind, res_dir, TRAIL, "orig_img_class{}_epoch{}.png".format(cls_label, epoch), img_save_dir = "rec_img")
        show(rec, rand_ind, res_dir, TRAIL, "rec_img_class{}_epoch{}.png".format(cls_label,epoch), img_save_dir = "rec_img")
        show(half_random, rand_ind, res_dir, TRAIL, "half_rand_img_class{}_epoch{}.png".format(cls_label,epoch), img_save_dir = "rec_img")
        show(full_random, rand_ind, res_dir, TRAIL, "generated_img_class{}_epoch{}.png".format(cls_label,epoch), img_save_dir = "rec_img")


ce_losses = []
cls_losses = []

grad_paras = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)
total_paras = sum(p.numel() for p in transformer_model.parameters())
print("total paras: ", total_paras, "trainable paras: ", grad_paras)
total_time_start = time.time()

for epoch in range(EPOCH):
    print("new epoch starts")
    
    epoch_start_time = time.time()
    batch_start_time = time.time()
    
    ce_loss_iter = 0
    cls_loss_iter = 0
    transformer_model.train()
    for i, data_b in enumerate(train_loader):
        batch_duration = 0
        optim_transformer.zero_grad()
        
        data, lab = get_data_batch_ROI_128(data_b, device, need_lab = True)
        #print(lab, lab.shape)
        
        real_cpu = data
        #print("real cpu: ", real_cpu.shape)
        b_size = real_cpu.size(0)
        
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            _,logits, targets = transformer_model(data, lab)
            #class_logits = transformer_model.linear_prob(logits)
            #class_label = torch.argmax(class_logits, dim = 1)
            loss = F.cross_entropy(logits.contiguous().reshape(-1, logits.size(-1)), targets.contiguous().reshape(-1))
        
        scaler.scale(loss).backward(retain_graph=True)
        scaler.step(optim_transformer)
        scaler.update()
        
        ce_loss_iter += loss.item()
    
    if lr_decay:
        stepG.step()
        
    ce_losses.append(ce_loss_iter / len(train_loader))
    #cls_losses.append(cls_loss_iter / len(train_loader))

    if i % 10 == 0:
        print("[%d/%d] batches done!\n" % (i,
                                            len(dataset)//BATCH_SIZE))
        batch_end_time = time.time()
        batch_duration = batch_duration + batch_end_time - batch_start_time
        print("Training time for", i, "batches: ", batch_duration / 60,
                " minutes.")
    
    print('[%d/%d]\tEntropy: %.4f' #\tClassification: %.4f'  
        % (epoch, EPOCH, ce_losses[-1]))
    
    # logging res
    if not os.path.exists(res_dir + "/trail{}/rec_img".format(TRAIL)):
        os.makedirs(res_dir + "/trail{}/rec_img".format(TRAIL))

    if not os.path.exists(res_dir + "/trail{}/pretrained_model".format(TRAIL)):
        os.makedirs(res_dir + "/trail{}/pretrained_model".format(TRAIL))
    
    if (epoch+1) % save_plot_every == 0:
        #transformer_model.eval()
        rand_ind = np.array(list(range(50, 50+16)))
        with torch.no_grad():
            random_from = BATCH_SIZE if data.shape[0] == BATCH_SIZE else data.shape[0]
            ind = np.random.randint(0, random_from)
            sample_data = data[ind].unsqueeze(0)
            pat_lab = lab[ind]
            #print(pat_lab)
            log, rec_and_pred_img = transformer_model.log_images(sample_data, pat_lab)
            cls = log["class"]
            save_log_images(rec_and_pred_img, cls, rand_ind, epoch, res_dir, TRAIL, val=False)
            
            for j, data_b in enumerate(val_loader):
                val_data, val_lab = get_data_batch_ROI_128(data_b, device, need_lab = True)
                _,logits, targets = transformer_model(val_data, val_lab)
                loss = F.cross_entropy(logits.contiguous().reshape(-1, logits.size(-1)), targets.contiguous().reshape(-1))
                print("val loss: ", loss.item())
                log, rec_and_pred_img = transformer_model.log_images(val_data[ind].unsqueeze(0), val_lab[ind])
                cls = log["class"]
                save_log_images(rec_and_pred_img, cls, rand_ind, epoch, res_dir, TRAIL, val=True)
    
    
        torch.save(transformer_model.state_dict(), res_dir + "/trail{}/pretrained_model/".format(TRAIL) + "epoch_{}.pt".format(epoch))
        
        del sample_data, rec_and_pred_img, val_data, val_lab, ind, rand_ind
        
    plot_loss(ce_losses, "Cross Entropy Loss during training",
                    res_dir + "/trail{}/".format(TRAIL), "ce_loss")
    # plot_loss(cls_losses, "Classification Loss during training",
    #                 res_dir + "/trail{}/".format(TRAIL), "cls_loss")
    #del sample_data, rec_and_pred_img, orig, rec, half_random, full_random
    torch.cuda.empty_cache()
print("Training finished, took {:.2f}h".format((time.time() - total_time_start)/3600))
