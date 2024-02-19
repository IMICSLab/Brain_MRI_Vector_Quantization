# configs for temporal-agnostic masked transformer

models_config = {
    "bs": 3, # batch size
    "epochs": 5000,
    "latent_dim": (8,8,8),
    "vqgan_model_weights": "../pretrained_model/epoch_7999.pt", # path to the pretrained vqgan model
    "class_num": 2, # number of classes
    "codebook_dim": 128, # codebook dimension
    "codebook_n_embed": 512, # codebook n_codes
    "mask_ratio": 0.50, # mask ratio
    "topk_ratio": 0.25, # topk ratio in the hybrid masking strategy
    "lr_decay": True, # whether to decay the learning rate
    "lr": 4.5e-06, # learning rate for transformer
    "save_plot_every": 1000, # save the plot every n epochs

}