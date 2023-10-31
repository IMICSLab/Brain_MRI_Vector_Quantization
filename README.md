# Code for the paper - Generating 3D Brain Tumor Regions in MRI using Vector-Quantization Generative Adversarial Networks

### Usage

To train the 3D-VQGAN model locally, run:
```python
python3 ./train_ae_128.py
```
You can also change the model configurations/parameters in the model_config file in the configs folder.

To run the script in the SLURM:
```shell
sbatch ./train_vae.sh
```
You can also change the sever-related settings, e.g., Memory, GPU, etc. in the .sh file

To train the transformer model locally, run:
```python
python3 ./train_transformer.py
```
You can also run this in the server, simply change the python file name in train_vae.sh accordingly.
You can change the hyperparameters of the transformer in the model_transformer_config file in the configs folder, and also in the vqgan_transformer.py file in the model folder, for some parameters with default values

Will update the script for sampling images from the transformer model in the future.
