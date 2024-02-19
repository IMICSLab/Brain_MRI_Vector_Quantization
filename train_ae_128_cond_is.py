
# with importance score computed

import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.variable as Variable
import torch.utils.data as data_utils
import torch.optim as optim
import torch.backends.cudnn as cudnn
#from pytorch_model_summary import summary
import sys
sys.path.append("../")

import meta_config as c
from dcgan.autoencoder_128_cond_v2 import Encoder, Decoder, weight_init, pre_vq_conv, post_vq_conv, NLayerDiscriminator3D
from dcgan.norm_ema_codebook import NormEMAVectorQuantizer
from dcgan.lightweight_classfier import *

from utils import set_seed
from utils.get_data import *
from utils.plot_loss import plot_loss
from utils.save_gen_img import show, show_importance_overlay
from utils.lpips import *
from utils.img_grad_loss import *
from cond_model_configs import model_configs

# Set random seed for reproducibility
set_seed.random_seed(0, True)


class L1Loss(nn.Module): 
    "Measuring the `Euclidian distance` between prediction and ground truh using `L1 Norm`"
    def __init__(self):
        super(L1Loss, self).__init__()
        
    def forward(self, x, y): 
        #N = y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3]*y.shape[4]
        assert x.shape == y.shape, "l1 loss x, y shape not the same"
        # x: recon image, y: orig image
        return F.l1_loss(x, y) #torch.mean(((x - y).abs()))


# to ensure it doesn't run partly on another gpu
#torch.cuda.set_device(c.cuda_n[0])

data_dir = c.data_dir

res_dir = c.res_dir_imgOnly_masked

print("loading data: ")
RoisL = torch.load(os.path.join(data_dir, "LGG_ROI_128_train.pt"))
labelsL = torch.zeros(RoisL.shape[0])
RoisH = torch.load(os.path.join(data_dir, "HGG_ROI_128_train.pt"))
labelsH = torch.ones(RoisH.shape[0])

Rois = torch.cat((RoisL, RoisH), dim=0)
labels = torch.cat((labelsL, labelsH), dim=0)
print("finish loading.")
print(RoisL.shape, RoisH.shape, Rois.shape, labels.shape)
print(torch.max(RoisL), torch.min(RoisL), torch.max(RoisH), torch.min(RoisH))
del RoisL, RoisH, labelsL, labelsH

# configs
BATCH_SIZE = model_configs["bs"]
z_dim = model_configs["latent_dim"] # last conv dim, bottleneck dim
EPOCH = model_configs["epoch"]
TRAIL = "_cond_is" #3
IF_RESUME = model_configs["resume_chkp"]
lr_decay = model_configs["lr_decay"]
latent_feature_maps = model_configs["latent_nfmaps"]
codebook_dim = model_configs["codebook_dim"]
codebook_n_embed = model_configs["n_embed"]
codebook_legacy_loss_beta = model_configs["codebook_legacy_loss_beta"]
n_classes = model_configs["n_class"]
interm_linear_dim = model_configs["cls_guidance_interm_dim"]
score_pred_net_mode = model_configs["score_net_mode"]
dis_start_epoch = model_configs["dis_start_epoch"]
c1 = model_configs["c1"]
save_plot_every = model_configs["save_plot_every"]
lr_g = model_configs["lr_g"]
lr_d = model_configs["lr_d"]
l1_weight = model_configs["l1_weight"]
perp_weight = model_configs["perp_weight"]
vq_weight = model_configs["vq_weight"]
gan_feat_weight = model_configs["gan_feat_weight"]
img_grad_weight = model_configs["img_grad_weight"]



if not os.path.exists(res_dir + "/trail{}".format(TRAIL)):
    os.makedirs(res_dir + "/trail{}".format(TRAIL))


total = list(range(Rois.shape[0]))
val_ind = list(np.random.choice(total, size=3, replace=False))

target_valid = [i for i in total if i not in val_ind]
# get dataset
#dataset = get_data_loader(Imgs, Msks, pIDs, target_valid, shuffle = True, norm = False)
dataset = get_data_loader_128_cond(Rois, labels, target_valid, shuffle = True, norm = False)
val_dataset = get_data_loader_128_cond(Rois, labels, val_ind, shuffle = False, norm = False)

# get training loader
train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=BATCH_SIZE,
                                            drop_last = False, # set to True for VAE
                                            shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False)

print("the length of the training loader is {} with batch size {}".format(len(train_loader), BATCH_SIZE))
# Device selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================== define modules ==============================
preV_conv = pre_vq_conv(latent_feature_maps, codebook_dim, 1).to(device)
postV_conv = post_vq_conv(codebook_dim, latent_feature_maps, 1).to(device)

score_net = socre_net(input_dim = codebook_dim, n_classes = n_classes, interm_linear_dim = interm_linear_dim, score_pred_net_mode = score_pred_net_mode).to(device)

netE = Encoder().to(device) # encoder
netD = Decoder(z_dim).to(device)
codebook = NormEMAVectorQuantizer(n_embed = codebook_n_embed, embedding_dim = codebook_dim, beta = codebook_legacy_loss_beta).to(device)
netNL_Dis = NLayerDiscriminator3D().to(device)

# some initialization
l1_loss = L1Loss()
lpips_loss = LPIPS().to(device).eval()
#==============================================================================


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def calculate_adaptive_weight(nll_loss, g_loss, last_layer=None):
    if last_layer is not None:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    else:
        nll_grads = torch.autograd.grad(nll_loss, last_layer[0], retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer[0], retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    d_weight = d_weight * 1.0
    return d_weight

def perp_loss(recon, data, num_slice, device, loss_fn = lpips_loss):
    B, C, T, H, W = data.shape
    loss_perp = 0.
    for _ in range(num_slice):
        frame_idx = torch.randint(0+30, T-30, [B]).to(device) # concentrate in center frames that most likely contain tumor, can be changed accordingly
        frame_idx_selected = frame_idx.reshape(-1,
                                            1, 1, 1, 1).repeat(1, C, 1, H, W)
        orig_rand_2d = torch.gather(data, 2, frame_idx_selected).squeeze(2)
        recon_rand_2d = torch.gather(recon, 2, frame_idx_selected).squeeze(2)

        loss_perp += loss_fn(orig_rand_2d, recon_rand_2d).mean()
    return loss_perp


vae_lr = lr_g
dis_lr = lr_d
optimizerDis = optim.Adam(netNL_Dis.parameters(), lr = dis_lr,
                        betas=(0.5, 0.9))
optimizerV = optim.Adam(list(netD.parameters()) + list(netE.parameters()) + list(codebook.parameters()) + list(preV_conv.parameters()) + list(postV_conv.parameters()) + list(score_net.parameters()), lr = vae_lr,
                        betas=(0.5, 0.9))


if lr_decay:
    stepG = optim.lr_scheduler.CosineAnnealingLR(optimizerV, T_max=EPOCH, eta_min = 0.0000003)
    stepD = optim.lr_scheduler.CosineAnnealingLR(optimizerDis, T_max=EPOCH, eta_min = 0.0000003)


last_epoch = 0 # previous stopped epoch, for resume training. 0 for training from the beginning
if IF_RESUME:
    saved_weights_dir = os.path.join(res_dir, "trail{}".format(TRAIL), "pretrained_model")
    sorted_weights_dir = sorted(os.listdir(saved_weights_dir), key = lambda x: int(x.split("_")[1].split(".")[0]), reverse = True)
    last_epoch = int(sorted_weights_dir[0].split("_")[1].split(".")[0])
    weights = torch.load(os.path.join(saved_weights_dir, "epoch_{}.pt").format(last_epoch))
    netE.load_state_dict(weights["Encoder"])
    netD.load_state_dict(weights["Decoder"])
    codebook.load_state_dict(weights["Codebook"])
    optimizerV.load_state_dict(weights["OptimizerV_state_dict"])
    score_net.load_state_dict(weights["ScoreNet"])
    preV_conv.load_state_dict(weights["preV_conv"])
    postV_conv.load_state_dict(weights["postV_conv"])
    print("resume training from epoch {}...".format(last_epoch))

else:
    # Apply the weights_init function to randomly initialize all weights
    netE.apply(weight_init)
    # Apply the weights_init function to randomly initialize all weights
    netD.apply(weight_init)
    #netDis.apply(weight_init)
    netNL_Dis.apply(weight_init)

# Lists to keep track of progress
rec_losses = []
total_losses = []
KL_losses = []
perp_losses = []
adv_g_losses = []
adv_d_losses = []
vq_losses = []
gan_feat_losses = []
img_grad_losses = []
cls_losses = []
#start_epoch = 1

iters = 0
duration = 0

# Training Loop

print("all preparation finished")
#sys.exit()
print("Starting Training Loop...")

# Scaler that tracks the scaling of gradients
# only used when mixed precision is used
use_mixed_precision = True
scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
USE_ACCUMU_BATCH = False
ACCUMU_BATCH = 2


total_start = time.time()

# For each epoch
for epoch in range(EPOCH):
    
    if not IF_RESUME:
        adv_d_weights = 1. if epoch >= dis_start_epoch else 0
    else:
        adv_d_weights = 1.
        
    print("new epoch starts")
    print("current LR G: ", stepG.get_last_lr())
    print("current LR D: ", stepD.get_last_lr())
    epoch_start_time = time.time()
    # For each batch in the dataloader
    rec_loss_iter = 0
    kl_loss_iter = 0
    adv_g_loss_iter = 0
    adv_d_loss_iter = 0
    perp_loss_iter = 0
    vq_loss_iter = 0
    gan_feat_loss_iter = 0
    img_grad_iter = 0
    cls_loss_iter = 0
    batch_start_time = time.time()

    netE.train(), netD.train(), score_net.train(), preV_conv.train(), postV_conv.train(), codebook.train(), netNL_Dis.train()
    for i, data_b in enumerate(train_loader):
        # for each iteration in the epoch
        batch_duration = 0
        # optimizerD.zero_grad()
        # optimizerE.zero_grad()
        optimizerV.zero_grad()
        optimizerDis.zero_grad()
        #data = get_data_batch(data_b, device)
        data, lab = get_data_batch_ROI_128(data_b, device, need_lab = True)
        # print("data shape", data.shape)
        # Format batch of real data
        real_cpu = data
        #print("real cpu: ", real_cpu.shape)
        b_size = real_cpu.size(0)

        # Training within the mixed-precision autocast - enabled/disabled
        loss_perp_batch = 0
        img_grad_batch = 0

        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            im_enc = netE(data) # encode
            im_enc = preV_conv(im_enc) # conv before vq            
            # before codebook, compute score & learn important region via classification
            cls_loss, _, pred_score, pred_vis, _ = score_net(im_enc, lab)
            vq_out = codebook(im_enc) # vq
            im_enc_vq = vq_out["embeddings"]
            im_enc_vq = postV_conv(im_enc_vq) # conv after vq
            im_out = netD(im_enc_vq) # decode
            #logits_recon_fake = netDis(im_out)
            logits_recon_fake, pred_recon_fake = netNL_Dis(im_out)
            img_grad = img_grad_loss_3d(data, im_out, device)

            cls_loss_batch = cls_loss
                
            img_grad_batch += img_grad
            #logits_real = netDis(data)
            loss_adv_g_batch = -torch.mean(logits_recon_fake)            

            #logits_real = netDis(data)
            loss_adv_g_batch = -torch.mean(logits_recon_fake)

            loss_perp_batch += perp_loss(im_out, real_cpu, 6, device) #lpips_loss(orig_rand_2d, recon_rand_2d).mean()
                
            loss_rec_batch = l1_loss(im_out, real_cpu)
            
            adv_g_loss_batch = loss_adv_g_batch
            try:
                d_weight = calculate_adaptive_weight(loss_rec_batch, adv_g_loss_batch, last_layer=netD.final_conv.weight)# final_cov.weight
            except RuntimeError:
                print("somthing wrong with calculate_adaptive_weight")
                d_weight = torch.tensor(0.0)
            
            d_weight = d_weight.clamp_max_(1.0) # clamp to max 1.0
            adv_g_loss_batch = d_weight * adv_g_loss_batch

            gan_feat_temp = 0
            logits_data_real, pred_data_real = netNL_Dis(real_cpu)
            for i in range(len(pred_recon_fake)-1):
                gan_feat_temp += F.l1_loss(pred_recon_fake[i], pred_data_real[i].detach())
            
            gan_feat_loss = gan_feat_temp
            vq_loss = vq_out["commitment_loss"]
            enc_loss = c1 * (l1_weight * loss_rec_batch + perp_weight * loss_perp_batch + vq_weight * vq_loss + gan_feat_weight * gan_feat_loss + img_grad_weight * img_grad_batch + adv_d_weights * adv_g_loss_batch) + (1-c1) * cls_loss_batch
            # / ACCUMU_BATCH #+ loss_KL_batch  0.01 * loss_adv_g_batch
        
        scaler.scale(enc_loss).backward(retain_graph = True)
        scaler.unscale_(optimizerV)
        torch.nn.utils.clip_grad_value_(netE.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(netD.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(preV_conv.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(postV_conv.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(codebook.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(score_net.parameters(), 1.0)
        #stepG.step(epoch + i / total_batch_iter)
        #stepG.step()
        #if (i+1) % ACCUMU_BATCH == 0:
        scaler.step(optimizerV)
        scaler.update()
        

        with torch.cuda.amp.autocast(enabled=True):
            dis_real,_ = netNL_Dis(data.detach())
            dis_fake,_ = netNL_Dis(im_out.detach())
            loss_adv_d_batch_3d = hinge_d_loss(dis_real, dis_fake) #/ ACCUMU_BATCH
            #loss_adv_d_batch_2d = hinge_d_loss(dis_real_2d, dis_fake_2d) #/ ACCUMU_BATCH
            loss_adv_d_batch = adv_d_weights * (loss_adv_d_batch_3d) #+ loss_adv_d_batch_2d)
        
        scaler.scale(loss_adv_d_batch).backward()
        scaler.unscale_(optimizerDis)
        torch.nn.utils.clip_grad_value_(netNL_Dis.parameters(), 1.0)
        #if (i+1) % ACCUMU_BATCH == 0:
        #stepD.step()
        scaler.step(optimizerDis)

        rec_loss_iter += loss_rec_batch.item()
        #kl_loss_iter += loss_KL_batch.item()
        adv_g_loss_iter += adv_g_loss_batch.item()
        adv_d_loss_iter += loss_adv_d_batch.item()
        perp_loss_iter += (loss_perp_batch).item()
        vq_loss_iter += vq_loss.item()
        gan_feat_loss_iter += gan_feat_loss.item()
        img_grad_iter += img_grad_batch.item()
        cls_loss_iter += cls_loss_batch.item()

        total_iter = rec_loss_iter + perp_loss_iter + vq_loss_iter + gan_feat_loss_iter + adv_d_loss_iter + img_grad_iter + adv_g_loss_iter + cls_loss_iter  #adv_d_loss_iter + adv_g_loss_iter
    
    if lr_decay:
        stepG.step()
        stepD.step()
    #adv_losses.append(adv_loss_iter / len(train_loader))
    #KL_losses.append(kl_loss_iter / len(train_loader))
    rec_losses.append(rec_loss_iter / len(train_loader))
    total_losses.append(total_iter / len(train_loader))
    adv_g_losses.append(adv_g_loss_iter / len(train_loader))
    adv_d_losses.append(adv_d_loss_iter / len(train_loader))
    perp_losses.append(perp_loss_iter / len(train_loader))
    vq_losses.append(vq_loss_iter / len(train_loader))
    gan_feat_losses.append(gan_feat_loss_iter / len(train_loader))
    img_grad_losses.append(img_grad_iter / len(train_loader))
    cls_losses.append(cls_loss_iter / len(train_loader))

    del im_out, im_enc, vq_out, im_enc_vq, logits_data_real, pred_data_real, logits_recon_fake, pred_recon_fake, dis_fake, dis_real, pred_score
    #dis_fake, dis_real, std, mean, logvar dis_real, dis_fake, d_loss_fake, d_loss_real
    iters += 1
    torch.cuda.empty_cache()
    # print after every 100 batches
    if i % 5 == 0:
        print("[%d/%d] batches done!\n" % (i,
                                            len(dataset)//BATCH_SIZE))
        batch_end_time = time.time()
        batch_duration = batch_duration + batch_end_time - batch_start_time
        print("Training time for", i, "batches: ", batch_duration / 60,
                " minutes.")

    print('[%d/%d]\tLoss_rec: %.4f\tLoss_perp: %.4f\tLoss_vq: %.4f\tLoss_img_grad: %.4f\tLoss_Dis: %.4f\tLoss_Adv_G: %.4f\tLoss_Cls: %.4f'
            % (epoch, EPOCH, rec_losses[-1], perp_losses[-1], vq_losses[-1], img_grad_losses[-1], adv_d_losses[-1], adv_g_losses[-1], cls_losses[-1]))


    if not os.path.exists(res_dir + "/trail{}/rec_img".format(TRAIL)):
        os.makedirs(res_dir + "/trail{}/rec_img".format(TRAIL))

    if not os.path.exists(res_dir + "/trail{}/pretrained_model".format(TRAIL)):
        os.makedirs(res_dir + "/trail{}/pretrained_model".format(TRAIL))


    if (epoch+1) % save_plot_every == 0: # save generated image and model weights
        netE.eval(), netD.eval(), score_net.eval(), preV_conv.eval(), postV_conv.eval(), codebook.eval(), netNL_Dis.eval()
        rand_ind = list(range(52, 52+16))
        ind = 1
        rec_data = data[ind].unsqueeze(0)
        im_enc = netE(rec_data)
        im_enc = preV_conv(im_enc)
        pred_score, pred_vis = score_net(im_enc, label = None)
        vq_out = codebook(im_enc)
        im_enc_vq = vq_out["embeddings"]
        im_enc_vq = postV_conv(im_enc_vq)
        im_out = netD(im_enc_vq)
        im_out = im_out.detach().cpu()
        train_sample_img = im_out[0][0]
        train_img_orig = rec_data[0,0,:,:,:].detach().cpu()
        pred_vis = pred_vis[0,0,:,:,:].detach().cpu()
        show(train_sample_img, rand_ind, res_dir, TRAIL, "rec_img_train_epoch{}.png".format(epoch), img_save_dir = "rec_img")
        show(train_img_orig, rand_ind, res_dir, TRAIL, "orig_img_train_epoch{}.png".format(epoch), img_save_dir = "rec_img")
        show_importance_overlay(train_img_orig, pred_vis, rand_ind, res_dir, TRAIL, "orig_imgIS_train_epoch{}.png".format(epoch), gray_scale = False, img_save_dir = "rec_img")
        
        val_cls_loss = 0
        with torch.no_grad():
            for j, data_b in enumerate(val_loader):
                data, lab = get_data_batch_ROI_128(data_b, device, need_lab = True)
                im_enc = netE(data)
                im_enc = preV_conv(im_enc)
                cls_loss, cls_prob, pred_score, pred_vis, _ = score_net(im_enc, lab)
                vq_out = codebook(im_enc)
                im_enc_vq = vq_out["embeddings"]
                im_enc_vq = postV_conv(im_enc_vq)
                im_out = netD(im_enc_vq)
                im_out = im_out.detach().cpu()
                val_cls_loss += cls_loss.item()
                #print("eval im_out shape: ", im_out.shape)
                #print("saving some results:")
                print("=======================================")
                print("acutal label: ", lab)
                print("pred label: ", cls_prob)
                print("=========================================")
                sample = im_out.clone() # 3,1,128,128,128
                sample_img = sample[ind][0] # we can select mid img for vis
                # orig img
                #print("data shape", rec_data.shape)
                sample_img_orig = data[ind,0,:,:,:].detach().cpu()
                pred_vis = pred_vis[ind,0,:,:,:].detach().cpu()
                rand_ind = list(range(52, 52+16))
                
                show(sample_img, rand_ind, res_dir, TRAIL, "rec_img_epoch{}.png".format(epoch), img_save_dir = "rec_img")
                show(sample_img_orig, rand_ind, res_dir, TRAIL, "orig_img_epoch{}.png".format(epoch), img_save_dir = "rec_img")
                show_importance_overlay(sample_img_orig, pred_vis, rand_ind, res_dir, TRAIL, "orig_imgIS_epoch{}.png".format(epoch), gray_scale = False, img_save_dir = "rec_img")
        val_cls_loss = val_cls_loss / len(val_loader)
        print("val cls loss: ", val_cls_loss)
        torch.save({'Encoder': netE.state_dict(),
                    'Decoder': netD.state_dict(),
                    "ScoreNet": score_net.state_dict(),
                    'Codebook': codebook.state_dict(),
                    "preV_conv": preV_conv.state_dict(),
                    "postV_conv": postV_conv.state_dict(),
                    'OptimizerV_state_dict': optimizerV.state_dict()
                    }, res_dir + "/trail{}/pretrained_model/".format(TRAIL) + "epoch_{}.pt".format(epoch))
        #netE.train(), netD.train(), netE.train(), codebook.train(), preV_conv.train(), postV_conv.train()
        del im_out, sample, sample_img, sample_img_orig, rec_data, im_enc, vq_out, im_enc_vq, pred_vis, pred_score, val_cls_loss #, sample_masks, im_en

    torch.cuda.empty_cache()
    # plot loss
    # plot_loss(KL_losses, "KL Loss during training",
    #                     res_dir + "/trail{}/".format(TRAIL), "KL_loss")
    plot_loss(rec_losses, "Rec Loss during training",
                        res_dir + "/trail{}/".format(TRAIL), "Rec_loss")
    plot_loss(perp_losses, "Perceptual Loss during training",
                        res_dir + "/trail{}/".format(TRAIL), "Perp_loss")
    plot_loss(total_losses, "Total Loss during training",
                        res_dir + "/trail{}/".format(TRAIL), "Total_loss")
    plot_loss(adv_d_losses, "Adv-D Loss during training",
                        res_dir + "/trail{}/".format(TRAIL), "Adv_d_loss")
    plot_loss(adv_g_losses, "Total Loss during training",
                        res_dir + "/trail{}/".format(TRAIL), "Adv_g_loss")
    plot_loss(vq_losses, "vq Loss during training",
                        res_dir + "/trail{}/".format(TRAIL), "vq_loss")
    plot_loss(cls_losses, "Cls Loss for importance score learning during training",
                        res_dir + "/trail{}/".format(TRAIL), "cls_loss")

    del rec_loss_iter, perp_loss_iter, vq_loss_iter, gan_feat_loss_iter, adv_d_loss_iter, img_grad_iter, adv_g_loss_iter, cls_loss_iter
    torch.cuda.empty_cache()

print("total training time in hours: ", (time.time() - total_start) / 3600.)