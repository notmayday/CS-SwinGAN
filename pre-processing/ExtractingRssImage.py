import numpy as np
import glob
import nibabel as nib
import os
import random
import h5py
import shutil
from multiprocessing import Pool
from PIL import Image
import matplotlib.pyplot as plt

directory_path='/media/samuel/storage/UIH/TEMP/data/h5/train1/'
save_path='/media/samuel/storage/UIH/V3+/data/h5/train1/'
h5_files = [file for file in os.listdir(directory_path) if file.endswith('.h5')]
size=np.size(h5_files)

# This python file is used for extracting RSS image from h5 files which includes kspace, RSS and other data.

new_size=[256,256]
for file_name in h5_files:
    file_path = os.path.join(directory_path, file_name)
    with h5py.File(file_path, 'r') as h5_file:
        print(f"Processing {file_name}:")
        rss = h5_file['reconstruction_rss']
        rss= np.array(rss)
        data = rss
        data = np.array(data).transpose((1, 2, 0))
        data_shape = [new_size[0], new_size[1], data.shape[2]]
        resized_imgs = np.zeros((new_size[0], new_size[1], data.shape[2]))
        for slice in range(data.shape[2]):
            img = Image.fromarray(data[:, :, slice])
            resized_img = img.resize(new_size)
            resized_imgs[:, :, slice] = resized_img
            image_1 = resized_imgs[:,:,slice]
        data = resized_imgs
        patient_name = os.path.split(file_path)[1].replace('h5', 'hdf5')
        output_file_path = save_path + patient_name
        with h5py.File(output_file_path, 'w') as f:
            dset = f.create_dataset('data', data_shape, data=data, compression="gzip", compression_opts=9)

