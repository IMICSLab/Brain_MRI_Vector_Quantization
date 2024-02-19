# sample transformer cond model configs

sampling_config = {
    "latent_dim": (8,8,8), # latent space dimension
    "num_sample": 250, # number of samples
    "num_classes": 2, # number of classes
    "cls_lab_indicator": [0,1], # class label indicator
    "cls_map": {0:"LGG", 1:"HGG"}, # class label mapping
    "parent_save_dir": "./transformer_cond", # parent directory to save the samples
    "save_folder": "./samples", # folder to save the samples
    "vqgan_model_weights": "/hpf/largeprojects/fkhalvati/Simon/3dgan_res/BraTS_masked/trail_cond_v2/pretrained_model/epoch_7999.pt",
    "chkpt": "epoch_4999", # checkpoint to load
    "bs": 1, # batch size
    "topk": None,
    "topp": None,
    "temperature": 1.0, # temperature for sampling
    
}