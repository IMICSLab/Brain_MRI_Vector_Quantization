# Code for the paper - Generating 3D Brain Tumor Regions in MRI using Vector-Quantization Generative Adversarial Networks

[Paper](https://arxiv.org/abs/2310.01251)

### Usage

To train the 3D-VQGAN model locally, run:
```python
python3 ./train_ae_128.py
```
You can also change the model configurations/parameters in the model_config file in the configs folder.

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
You can change the hyperparameters of the transformer in the model_transformer_config file in the configs folder, and also in the vqgan_transformer.py file in the model folder, for some parameters with default values

To sample images from the trained transformer, run: 
```python
python3 ./sample_transformer.py
```
You can change the *temperature* parameter in the sampling method, which controls the diversity.

For MS-SSIM and MMD calculation, please refer to [this repo](https://github.com/cyclomon/3dbraingen), and for the FID score, the implementation is available at [here](https://github.com/mseitzer/pytorch-fid)

This is just a simple guide to train our model. We will provide a more detailed model usage, and other useful functions in the future.
