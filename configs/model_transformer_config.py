
'''
specify some para. for transformer
'''
bs = 3
z_dim = (8,8,8) # last conv dim, bottleneck dim
epoch = 1500 # 2000
latent_channels = 512
n_codes = 512
lr = 4.5e-06

ae_weights = "./trailvqgan/pretrained_model/epoch_3999.pt"

# save dirs
trail = "transformer"
data_dir = "./datapath"
save_dir = "./results"