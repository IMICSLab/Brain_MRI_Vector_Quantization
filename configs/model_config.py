
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
dis_start = 2500
perp_num = 3
# save dirs
trail = "vqgan"
data_dir = "./datapath"
save_dir = "./results"