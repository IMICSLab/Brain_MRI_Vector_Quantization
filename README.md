# Code for the paper - (Unconditional/Class-conditioned) generation of 3D Brain Tumor Regions in MRI using VQGAN and Transformer

[Paper for unconditional generation](https://arxiv.org/abs/2310.01251), under review at Computers in Biology and Medicine

[Paper for conditional generation](https://openreview.net/pdf?id=LLoSHPorlM), under review at MIDL 2024

## Usage
### Unconditional
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
You can change the *temperature* parameter in the sampling method, which controls the diversity.

### Conditional
To train the 3D-VQGAN-cond model locally, run:
```python
python3 ./train_ae_128_cond_is.py
```
You can also change the model configurations/parameters in the *cond_model_config* file in the configs folder.

To run the script in the SLURM, change the python script name and run:
```shell
sbatch ./train.sh
```
To train the temporal-agnostic masked transformer model locally, run:
```python
python3 ./train_transformer_cond.py
```
You can also run this in the server, simply change the python file name in train.sh accordingly.
You can change the hyperparameters of the transformer in the *transformer_cond_config* file in the configs folder, and also in the vqgan_transformer_cond.py file in the model folder, for some parameters with default values

To sample images from the trained transformer, run: 
```python
python3 ./sample_transformer_cond.py
```
You can change the *temperature, topp, topk* parameters in the sampling method in the *sample_cond_configs* in the configs folder, which controls the diversity and quality. The default is topp = None, topk = None, temperature = 1.0

==================================================================================================

For MS-SSIM and MMD calculation, please refer to [this repo](https://github.com/cyclomon/3dbraingen), and for the FID score, the implementation is available at [here](https://github.com/mseitzer/pytorch-fid)
