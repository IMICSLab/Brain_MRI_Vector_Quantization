
'''
specify some model parameters
'''
bs = 3
z_dim = (8,8,8) # last conv dim, bottleneck dim
epoch = 5000
latent_channels = 512
n_codes = 512
vae_lr = 0.0001
dis_lr = 0.0001
lr_decay = True
codebook_legacy_loss_beta = 0.25
dis_start = 2500
perp_num = 3
l1_weight: 4.0, # l1 loss weight
perp_weight: 1.0, # perceptual loss weight
vq_weight: 1.0, # vq loss weight
gan_feat_weight: 4.0, # feature matching loss weight
img_grad_weight: 4.0, # image gradient loss weight
# save dirs
trail = "vqgan"
data_dir = "./datapath"
save_dir = "./results"
