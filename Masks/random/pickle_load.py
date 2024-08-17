import numpy as np
import pickle
import matplotlib.pyplot as plt



size = 256
sampling_rate = 20

with open(f'mask_{sampling_rate}_{size}.pickle', 'rb') as file:
    loaded_mask = pickle.load(file)
    kk=loaded_mask["mask0"]
plt.imshow(kk,cmap='gray')
plt.title('Mask')
plt.show()