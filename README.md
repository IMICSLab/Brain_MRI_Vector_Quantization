# Code for the paper - Generating 3D Brain Tumor Regions in MRI using Vector-Quantization Generative Adversarial Networks

### Usage

To run the script locally:
```python
python3 ./train_ae_128.py
```
You can also change the model configurations/parameters in the model_config file in the configs folder.

To run the script in SLURM:
```shell
sbatch ./train_vae.sh
```
You can also change the sever-related settings, e.g., Memory, GPU, etc. in the .sh file
