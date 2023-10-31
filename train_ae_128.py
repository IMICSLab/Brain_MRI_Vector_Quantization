
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

#import meta_config as c
import model_config as mc
#from dcgan.model_v2 import Generator3D, Discriminator3D, weight_init
#from dcgan.model_v6 import Generator3D, Discriminator3D, weight_init
from dcgan.autoencoder_128_v2 import Encoder, Decoder, Codebook, Discriminator3D, weight_init, VAE, pre_vq_conv, post_vq_conv, NLayerDiscriminator3D
#from dcgan.autoencoder_128_new import Encoder, Decoder, Codebook, Discriminator3D, weight_init, VAE, pre_vq_conv, post_vq_conv, NLayerDiscriminator3D

from utils import set_seed
from utils.get_data import *
from utils.plot_loss import plot_loss
from utils.save_gen_img import show
from utils.find_longest_maskSeq import find_maskSeq
from utils.lpips import *
from utils.img_grad_loss import *

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


# to ensure it doesn't run partly on another gpu
#torch.cuda.set_device(c.cuda_n[0])

data_dir = mc.data_dir

res_dir = c.save_dir

#data_dir = "Z:/Datasets/MedicalImages/BrainData/SickKids/LGG/AI_ready"
print("loading data: ")
# Msks = load_object(os.path.join(data_dir, "Masks.p").replace("\\", "/"))
# Imgs = load_object(os.path.join(data_dir, "imgs.p").replace("\\", "/"))
# # labels = load_object("../data/labels.p")
# pIDs = load_object(os.path.join(data_dir, "pIDs.p").replace("\\", "/"))
#Rois = torch.load(os.path.join(data_dir, "LGG_ROI_128_train.pt"))
Rois = torch.load(os.path.join(data_dir, "mutation_ROIs_128_train.pt"))
print("finish loading.")
print(Rois.shape)

BATCH_SIZE = mc.bs
z_dim = mc.z_dim #(8,8,8) # last conv dim, bottleneck dim
EPOCH = mc.epoch
codebook_dim = mc.latent_channels
TRAIL = mc.trail #"1_rerunV2"#3
#LATENT_DIM = 1000 # Linear latent dim
lr_decay = mc.lr_decay

if not os.path.exists(res_dir + "/trail{}".format(TRAIL)):
    os.makedirs(res_dir + "/trail{}".format(TRAIL))

config_dict = {}

#print("this version includes the codebook loss")
config_dict["Description"] = "some description"
config_dict["Trail"] = TRAIL
config_dict["Batch Size"] = BATCH_SIZE
config_dict["z_dim"] = z_dim
config_dict["Epoch"] = EPOCH
config_dict["lr decay"] = lr_decay
config_dict["decay stragegy"] = "cosine annealing"


#total = list(range(Rois.shape[0]))
#val_ind = list(np.random.choice(total, size=3, replace=False))
target_valid = list(range(Rois.shape[0]))
# get dataset
#dataset = get_data_loader(Imgs, Msks, pIDs, target_valid, shuffle = True, norm = False)
dataset = get_data_loader_128(Rois, target_valid, shuffle = True, norm = False)
#val_dataset = get_data_loader_128(Rois, val_ind, shuffle = False, norm = False)

# get training loader
train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=BATCH_SIZE, # set to True for VAE
                                            shuffle=True)

# val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                             batch_size=BATCH_SIZE,
#                                             shuffle=False)

print("the length of the training loader is {} with batch size {}".format(len(train_loader), BATCH_SIZE))
# Device selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print('Torch CUDA Current Device: {}'.format(torch.cuda.current_device()))
# print('Torch CUDA Device: {}'.format(torch.cuda.device(torch.cuda.current_device())))
# print('Torch CUDA Device Count: {}'.format(torch.cuda.device_count()))
# print('Torch CUDA Device Name: {}'.format(torch.cuda.get_device_name(torch.cuda.current_device())))
# print('Torch CUDA Availability: {}'.format(torch.cuda.is_available()))

total_batch_iter = len(train_loader)

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

# some initialization
l1_loss = L1Loss()
#kl_loss = KLDivergence()
lpips_loss = LPIPS().to(device).eval()
preV_conv = pre_vq_conv(codebook_dim, codebook_dim, 1).to(device)
postV_conv = post_vq_conv(codebook_dim, codebook_dim, 1).to(device)


# ####Create VAE object##### #
netE = Encoder().to(device) # encoder

netD = Decoder(z_dim).to(device)
codebook = Codebook(mc.n_codes, codebook_dim).to(device)

# netVAE = VAE(device, z_dim).to(device)
#netDis = Discriminator3D().to(device)
netNL_Dis = NLayerDiscriminator3D().to(device)

# Apply the weights_init function to randomly initialize all weights
netE.apply(weight_init)
# Apply the weights_init function to randomly initialize all weights
netD.apply(weight_init)
#netDis.apply(weight_init)
netNL_Dis.apply(weight_init)
#netVAE.apply(weight_init)

#enc_lr = 0.0001
#dec_lr = 0.0001
vae_lr = mc.var_lr #0.0001 # start lr, old 0.00005
dis_lr = mc.dis_lr #0.0001
# optimizerD = optim.Adam(netD.parameters(), lr = dec_lr,
#                         betas=(0, 0.9))
# optimizerE = optim.Adam(netE.parameters(), lr = enc_lr,
#                         betas=(0, 0.9))
optimizerDis = optim.Adam(netNL_Dis.parameters(), lr = dis_lr,
                        betas=(0.5, 0.9))

optimizerV = optim.Adam(list(netD.parameters()) + list(netE.parameters()) + list(codebook.parameters()) + list(preV_conv.parameters()) + list(postV_conv.parameters()), lr = vae_lr,
                        betas=(0.5, 0.9))

if lr_decay:
    stepG = optim.lr_scheduler.CosineAnnealingLR(optimizerV, T_max=EPOCH)
    stepD = optim.lr_scheduler.CosineAnnealingLR(optimizerDis, T_max=EPOCH)


# config_dict["encoder lr"] = enc_lr
# config_dict["decoder lr"] = dec_lr
config_dict["discriminator lr"] = dis_lr

config_dict["VAE lr"] = vae_lr


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

config_dict["enable accumu grad"] = USE_ACCUMU_BATCH
config_dict["batch accumulation"] = ACCUMU_BATCH

with open(res_dir + "/trail{}/config.txt".format(TRAIL), 'w') as f: 
    for key, value in config_dict.items():
        f.write("{}:{}\n".format(key, value))
f.close()

whole_process_start = time.time()
# For each epoch
for epoch in range(EPOCH):
    
    adv_d_weights = 1. if epoch >= mc.dis_start else 0 #max((EPOCH-epoch)/EPOCH, 0.5)
    
    print("new epoch starts")
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
    batch_start_time = time.time()

    # netD.train()
    # netE.train()
    # optimizerV.zero_grad()
    # optimizerDis.zero_grad()
    netE.train(), netD.train(), codebook.train(), preV_conv.train(), postV_conv.train(), netNL_Dis.train()
    for i, data_b in enumerate(train_loader):
        # for each iteration in the epoch
        batch_duration = 0
        # optimizerD.zero_grad()
        # optimizerE.zero_grad()
        optimizerV.zero_grad()
        optimizerDis.zero_grad()
        #data = get_data_batch(data_b, device)
        #data = get_data_batch_ROI_slice_small(data_b, ROW_MIN, ROW_MAX, COL_MIN, COL_MAX, CROP_IMG_TARGET, SLICES, device)
        data = get_data_batch_ROI_128(data_b, device)
        # print("data shape", data.shape)
        # Format batch of real data
        real_cpu = data
        #print("real cpu: ", real_cpu.shape)
        b_size = real_cpu.size(0)
        #noise = (r2 - r1) * torch.rand(b_size, z_dim, 1, 1, 1, device=device) + r1

        # Training within the mixed-precision autocast - enabled/disabled

        loss_perp_batch = 0
        img_grad_batch = 0

        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            im_enc = netE(data)
            enc_to_vq = preV_conv(im_enc)
            vq_out = codebook(enc_to_vq)
            im_enc_vq = postV_conv(vq_out["embeddings"])
            im_out = netD(im_enc_vq)
            #logits_recon_fake = netDis(im_out)
            logits_recon_fake, pred_recon_fake = netNL_Dis(im_out)
            img_grad = img_grad_loss_3d(data, im_out, device)

            img_grad_batch += img_grad
            #logits_real = netDis(data)
            loss_adv_g_batch = -torch.mean(logits_recon_fake)
            adv_g_loss_batch = loss_adv_g_batch
            
            B,C,T,H,W = im_out.shape

            for k in range(mc.perp_num): # 5, +20, -20
                frame_idx = torch.randint(0+30, T-30, [B]).to(device)
                frame_idx_selected = frame_idx.reshape(-1,
                                                    1, 1, 1, 1).repeat(1, C, 1, H, W)
                orig_rand_2d = torch.gather(data, 2, frame_idx_selected).squeeze(2)
                recon_rand_2d = torch.gather(im_out, 2, frame_idx_selected).squeeze(2)

                loss_perp_batch += lpips_loss(orig_rand_2d, recon_rand_2d).mean()

            if torch.any(torch.isnan(img_grad_batch)) or torch.any(torch.isinf(img_grad_batch)):
                print(img_grad_batch, "img_grad_batch nan or inf")
                sys.exit()
            # adv generator(autoenc)
            # logits_recon_fake = netDis(im_out)
            # #logits_real = netDis(data)
            # loss_adv_g_batch = -torch.mean(logits_recon_fake)

            # # adv discriminator
            # dis_real = netDis(data.detach())
            # dis_fake = netDis(im_out.detach())
            # d_loss_real = torch.mean(F.relu(1. - dis_real))
            # d_loss_fake = torch.mean(F.relu(1. + dis_fake))
            gan_feat_temp = 0
            logits_data_real, pred_data_real = netNL_Dis(real_cpu)
            for i in range(len(pred_recon_fake)-1):
                gan_feat_temp += l1_loss(pred_recon_fake[i], pred_data_real[i].detach())
            
            gan_feat_loss = gan_feat_temp
            if torch.any(torch.isnan(gan_feat_loss)) or torch.any(torch.isinf(gan_feat_loss)):
                print(gan_feat_loss, "gan_feat_loss nan or inf")
                sys.exit()
            #disc_factor * self.gan_feat_weight * (image_gan_feat_loss + video_gan_feat_loss)

            #netD(reparametrized_noise)
            #print("im_out shape: ", im_out.shape)
            # dis_real = netDis(real_cpu)
            # dis_fake = netDis(im_out)
            loss_rec_batch = l1_loss(im_out, real_cpu)
            if torch.any(torch.isnan(loss_rec_batch)) or torch.any(torch.isinf(loss_rec_batch)):
                print(loss_rec_batch, "loss_rec_batch nan or inf")
                sys.exit()
            if torch.any(torch.isnan(loss_perp_batch)) or torch.any(torch.isinf(loss_perp_batch)):
                print(loss_perp_batch, "loss_perp_batch nan or inf")
                sys.exit()            
            
            try:
                d_weight = calculate_adaptive_weight(loss_rec_batch, adv_g_loss_batch, last_layer=netD.final_conv.weight)# final_cov.weight
                #print("d_weight: ", d_weight)
            except RuntimeError:
                print("somthing wrong with calculate_adaptive_weight")
                d_weight = torch.tensor(0.0)
            
            d_weight = d_weight.clamp_max_(1.0) # clamp to max 1.0
            adv_g_loss_batch = d_weight * adv_g_loss_batch
            if torch.any(torch.isnan(adv_g_loss_batch)) or torch.any(torch.isinf(adv_g_loss_batch)):
                print(adv_g_loss_batch, "weighted adv_g_loss_batch nan or inf")
                sys.exit()
            # Train with fake batch
            #loss_KL_batch = kl_loss(mean, logvar)
            vq_loss= vq_out["commitment_loss"]
            if torch.any(torch.isnan(vq_loss)) or torch.any(torch.isinf(vq_loss)):
                print(vq_loss, "vq_loss nan or inf")
                sys.exit()
            # 4 * (1/3) for perp
            enc_loss = (4.0 * loss_rec_batch + 1.0 * loss_perp_batch + 1.0 * vq_loss + 4.0 * gan_feat_loss + 4.0 * img_grad_batch + adv_d_weights * adv_g_loss_batch)# / ACCUMU_BATCH #+ loss_KL_batch  0.01 * loss_adv_g_batch
        
        scaler.scale(enc_loss).backward(retain_graph = True)
        scaler.unscale_(optimizerV)
        torch.nn.utils.clip_grad_value_(netE.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(netD.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(codebook.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(preV_conv.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(postV_conv.parameters(), 1.0)            
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
        # d_loss_real = torch.mean(F.relu(1. - dis_real))
        # d_loss_fake = torch.mean(F.relu(1. + dis_fake))
        
        if torch.any(torch.isnan(loss_adv_d_batch)) or torch.any(torch.isinf(loss_adv_d_batch)):
            print(loss_adv_d_batch, "loss_adv_d_batch nan or inf")
            sys.exit()        
        
        scaler.scale(loss_adv_d_batch).backward()
        scaler.unscale_(optimizerDis)
        #torch.nn.utils.clip_grad_value_(netNL_Dis2d.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(netNL_Dis.parameters(), 1.0)
        #if (i+1) % ACCUMU_BATCH == 0:
        #stepD.step()
        scaler.step(optimizerDis)
            #scaler.update()
        # #scaler.update()
        # #optimizerD.step()


        rec_loss_iter += loss_rec_batch.item()
        #kl_loss_iter += loss_KL_batch.item()
        adv_g_loss_iter += loss_adv_g_batch.item()
        adv_d_loss_iter += loss_adv_d_batch.item()
        perp_loss_iter += (loss_perp_batch).item()
        vq_loss_iter += vq_loss.item()
        gan_feat_loss_iter += gan_feat_loss.item()
        img_grad_iter += img_grad_batch.item()

        total_iter = rec_loss_iter + perp_loss_iter + vq_loss_iter + gan_feat_loss_iter + adv_d_loss_iter + img_grad_iter + adv_g_loss_iter #adv_d_loss_iter + adv_g_loss_iter
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

        #print("gp: ", gradient_penalty)
        # Calculate gradients for D in backward pass
        #scaler.update()

        # del eps
        # del interpolate
        # del d_interpolate
    del im_out, im_enc, enc_to_vq, vq_out, im_enc_vq, orig_rand_2d, recon_rand_2d, frame_idx, frame_idx_selected, logits_data_real, pred_data_real, logits_recon_fake, pred_recon_fake, dis_fake, dis_real#dis_fake, dis_real, std, mean, logvar dis_real, dis_fake, d_loss_fake, d_loss_real
        
    iters += 1

    # print after every 100 batches
    if i % 5 == 0:
        print("[%d/%d] batches done!\n" % (i,
                                            len(dataset)//BATCH_SIZE))
        batch_end_time = time.time()
        batch_duration = batch_duration + batch_end_time - batch_start_time
        print("Training time for", i, "batches: ", batch_duration / 60,
                " minutes.")

    # update learning rate
    # if lr_decay:
    #     stepG.step()
    #     stepD.step()
    # print(" End of Epoch %d \n" % epoch)
    # %.4f\tLoss_adv_d: %.4f\tLoss_adv_g:
    print('[%d/%d]\tLoss_rec: %.4f\tLoss_perp: %.4f\tLoss_vq: %.4f\tLoss_img_grad: %.4f\tLoss_Dis: %.4f'
            % (epoch, EPOCH, rec_losses[-1], perp_losses[-1], vq_losses[-1], img_grad_losses[-1], adv_d_losses[-1]))

    # Save Losses and outputs for plotting later

    # save some sample generated images

    if not os.path.exists(res_dir + "/trail{}/rec_img".format(TRAIL)):
        os.makedirs(res_dir + "/trail{}/rec_img".format(TRAIL))

    if not os.path.exists(res_dir + "/trail{}/pretrained_model".format(TRAIL)):
        os.makedirs(res_dir + "/trail{}/pretrained_model".format(TRAIL))

    #for i in range(fixed_fake.shape[0]): # batch size
    if (epoch+1) % 1000 == 0: # save generated image and model weights
        netE.eval(), netD.eval(), codebook.eval(), preV_conv.eval(), postV_conv.eval(), netNL_Dis.eval()
        with torch.no_grad():
            ind = np.random.randint(0, BATCH_SIZE)
            rec_data = data[ind].unsqueeze(0)
            im_enc = netE(rec_data)
            enc_to_vq = preV_conv(im_enc)
            vq_out = codebook(enc_to_vq)
            im_enc_vq = postV_conv(vq_out["embeddings"])
            im_out = netD(im_enc_vq)
            im_out = im_out.detach().cpu()

            #print("eval im_out shape: ", im_out.shape)
            #print("saving some results:")
            sample = im_out.clone() # select the last batch for visualization
            sample_img = sample[0][0]
            # orig img
            #print("data shape", rec_data.shape)
            sample_img_orig = rec_data[0,0,:,:,:].detach().cpu()
            #sample_masks = sample[1]
            #assert sample_img.shape == sample_masks.shape, "img and mask should have the same shape"
            rand_ind = list(range(52, 52+16))#np.random.choice(sample_img.shape[0], size=16, replace=False)
            
            show(sample_img, rand_ind, res_dir, TRAIL, "rec_img_epoch{}.png".format(epoch), img_save_dir = "rec_img")
            show(sample_img_orig, rand_ind, res_dir, TRAIL, "orig_img_epoch{}.png".format(epoch), img_save_dir = "rec_img")
        #show(sample_masks, rand_ind, res_dir, TRAIL, "generated_mask_epoch{}.png".format(epoch))
        # torch.save(sample_img, "./3DGAN_res/generated_midimg_epoch{}.pt".format(epoch))
        # torch.save(sample_masks, "./3DGAN_res/generated_midmask_epoch{}.pt".format(epoch))
        #torch.save(sample_img, res_dir + "/trail{}/gen_img/generated_img_epoch{}.pt".format(TRAIL, epoch))
        #torch.save(sample_masks, res_dir + "/trail{}/gen_img/generated_mask_epoch{}.pt".format(TRAIL, epoch))
        torch.save({'Encoder': netE.state_dict(),
                    'Decoder': netD.state_dict(),
                    'Codebook': codebook.state_dict(),
                    'preV_conv': preV_conv.state_dict(),
                    'postV_conv': postV_conv.state_dict(),
                    'OptimizerV_state_dict': optimizerV.state_dict()
                    }, res_dir + "/trail{}/pretrained_model/".format(TRAIL) + "epoch_{}.pt".format(epoch))
        #netE.train(), netD.train(), netE.train(), codebook.train(), preV_conv.train(), postV_conv.train()
        del im_out, sample, sample_img, sample_img_orig, rec_data, im_enc, enc_to_vq, vq_out, im_enc_vq#, sample_masks, im_en

    # if (epoch+1) % 800 == 0:
    #     torch.save({'Encoder': netE.state_dict(),
    #                 'Decoder': netD.state_dict(),
    #                 'Codebook': codebook.state_dict(),
    #                 'preV_conv': preV_conv.state_dict(),
    #                 'postV_conv': postV_conv.state_dict(),
    #                 'OptimizerV_state_dict': optimizerV.state_dict()
    #                 }, res_dir + "/trail{}/pretrained_model/".format(TRAIL) + "epoch_{}.pt".format(epoch))
        
    #                     'OptimizerD_state_dict': optimizerD.state_dict(),
    #               'OptimizerG_state_dict': optimizerG.state_dict(),
    #                'Scaler_dict': scaler.state_dict()

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
    # del errD_iter
    # del errG_iter
    # del Wasserstein_D_iter

    # del avg_errD
    # del avg_errG
    # del avg_Wasserstein_D

    del rec_loss_iter, kl_loss_iter, perp_loss_iter, adv_d_loss_iter, adv_g_loss_iter

print("Total time in hours: ", (time.time() - whole_process_start)/3600)