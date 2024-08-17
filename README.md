# CS-SwinGAN

Official PyTorch implementation of CS-SwinGAN described in the paper "".

<div align="center">
<img src="./asserts/framework.png" width="800px">
</div>

<div align="center">
<img src="./asserts/CS-Block.png" width="800px">
</div>

## Dependencies

```
easydict==1.10
einops==0.8.0
focal-frequency-loss==0.3.0
h5py==3.9.0
ipython==8.14.0
matplotlib==3.7.2
nibabel==5.1.0
numpy==1.25.2
opencv-python==4.8.0.74
Pillow==10.0.0
PyWavelets==1.4.1
PyYAML==6.0.1
scikit-image==0.21.0
scipy==1.11.1
# Editable install with no version control (setuptools==68.0.0)
-e /home/samuel/anaconda3/envs/SwinGAN/lib/python3.9/site-packages/setuptools-68.0.0-py3.9.egg-info
timm==0.9.2
torch==2.0.1
tqdm==4.65.0
```

## Installation
- Clone this repo:
```bash
git clone https://github.com/notmayday/CS-SwinGAN
cd CS-SwinGAN
```

## Train

<br />

```
python3 train.py 

```


## Test

<br />

python3 difference_poisson.py 

<br />
<br />


# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.


<br />

# Acknowledgements

This code uses libraries from [Subsampled-Brain-MRI-Reconstruction-by-Generative-Adversarial-Neural-Networks](https://github.com/ItamarDavid/Subsampled-Brain-MRI-Reconstruction-by-Generative-Adversarial-Neural-Networks) and [SwinGAN](https://github.com/learnerzx/SwinGAN) repositories.
