# Code for the paper - Unconditional generation of 3D Brain Tumor Regions in MRI using VQGAN and Transformer

[Paper link](https://arxiv.org/abs/2310.01251), under revision at Computers in Biology and Medicine

## Usage

To train the 3D-VQGAN model locally, run:
```python
python3 ./train_ae_128.py
```
You can also change the model configurations/parameters in the *model_config* file in the configs folder.

To run the script in the SLURM:
```shell
sbatch ./train.sh
```
You can also change the sever-related settings, e.g., Memory, GPU, etc. in the .sh file

To train the transformer model locally, run:
```python
python3 ./train_transformer.py
```
You can also run this in the server, simply change the python file name in train.sh accordingly.
You can change the hyperparameters of the transformer in the *model_transformer_config* file in the configs folder, and also in the vqgan_transformer.py file in the model folder, for some parameters with default values

To sample images from the trained transformer, run: 
```python
python3 ./sample_transformer.py
```
You can change the *temperature, topp, topk* parameters in the sampling method in the *img_gen_configs* in the configs folder, which controls the diversity and quality. The default is topp = None, topk = None, temperature = 1.0

===========================================================================

For MS-SSIM and MMD calculation, please refer to [this repo](https://github.com/cyclomon/3dbraingen), and for the FID score, the implementation is available at [here](https://github.com/mseitzer/pytorch-fid)
