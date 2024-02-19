# 3d-vqgan-cond model configs


model_configs = {
    
    "epoch": 10000,
    "latent_dim": (8,8,8), # last conv dim, bottleneck dim
    "latent_nfmaps": 512, # last conv feature maps
    "bs": 3,
    "lr_g":0.00005, # generator learning rate
    "lr_d":0.00005, # discriminator learning rate
    "codebook_dim": 128, # codebook dimension
    "n_embed": 1024, # number of codes in codebook
    "codebook_legacy_loss_beta": 1.0, # codebook legacy loss beta
    "n_class": 2, # number of classes for class conditional
    "cls_guidance_interm_dim": 1024,
    "score_net_mode": "2layer_vanilla",
    "resume_chkp": False, # resume from the last saved checkpoint
    "lr_decay": True, # learning rate decay
    "dis_start_epoch": 2000, # start epoch for discriminator training
    "c1": 0.8, # loss weight for main task    
    "save_plot_every": 2000, # save ckpts and plot figure every 2000 epochs
    "l1_weight": 4.0, # l1 loss weight
    "perp_weight": 1.0, # perceptual loss weight
    "vq_weight": 1.0, # vq loss weight
    "gan_feat_weight": 4.0, # feature matching loss weight
    "img_grad_weight": 4.0, # image gradient loss weight
}