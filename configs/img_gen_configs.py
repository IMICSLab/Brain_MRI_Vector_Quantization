# configs of image generation for vqgan and ddpm

ddpm_config_plgg = {
    "vqgan_ckpt": "/hpf/largeprojects/fkhalvati/Simon/3dgan_res/pLGG/RSNA/trail_diffusion/pretrained_model/epochg_9999.pt",
    "loss_type": "l2",
    "DDPM_CHECKPOINT": "/hpf/largeprojects/fkhalvati/Simon/3dgan_res/pLGG/RSNA/trail_diffusion_ddpm/model-7500.pt",
    "dif_img_size": 16,
    "dif_depth_size": 16,
    "dif_num_channels": 8,
    "timestep": 300,
    "dim_mults": (8,16,32,32), #unet
    "batch_size": 4,
    "train_lr": 0.0001,
}

ddpm_onfig_brats = {
    "vqgan_ckpt": "/hpf/largeprojects/fkhalvati/Simon/3dgan_res/BraTS_brainOnly/trail_diffusion4/pretrained_model/epoch_9999.pt",
    "loss_type": "l2",
    "DDPM_CHECKPOINT": "/hpf/largeprojects/fkhalvati/Simon/3dgan_res/BraTS_brainOnly/trail_diffusion3_ddpmV2/model-7500.pt",
    "dif_img_size": 16,
    "dif_depth_size": 16,
    "dif_num_channels": 8,
    "timestep": 300,
    "dim_mults": (8,16,32,32), #unet
    "batch_size": 4,
    "train_lr": 0.0001,
}

vqgan_trans_lat8_brats_config = {
    "start_index": (0,512),
    "topk": 512,
    "topp": 1,
    "model_epoch": 1499,
}

vqgan_lat8_brats_config = {
    "codebook dim": (512,512),
    "prev/post conv": (512,512),
    "vae_lr": 0.0001,
    "dis_lr": 0.0001,
    "adv_d_weights_start": 2000,
}

vqgan_trans_lat8_plgg_config = {
    "batch_size": 4,
    "pkeep": 0.8,
    "topk":1024,
    "topp": 1,
    
}

vqgan_trans_gen_lat8_plgg_config = {
    "transformer_weights_dir": "/hpf/largeprojects/fkhalvati/Simon/3dgan_res/pLGG/RSNA/trail1_stage2_rerun_v2",
    "type": "random, 0-256", # 0
    "topk":512, # 256
    "topp": 0.8,
}