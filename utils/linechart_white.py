import matplotlib.pyplot as plt
import numpy as np
data = []
PSNR = [[34.52,37.47,39.22,32.29,32.44,32.96],[28.34,34.61,34.48,28.02,29.83,32.51],[22.77,31.63,34.48,23.36,28.75,32.25],
        [19.16,24.89,33.81,20.05,27.62,32.10],[16.55,19.77,33.19,17.54,26.55,32.01],[14.52,17.10,32.69,15.53,25.59,31.96]]
SSIM = [[0.87,0.95,0.96,0.87,0.85,0.89],[0.59,0.87,0.83,0.69,0.80,0.88],[0.35,0.74,0.84,0.45,0.74,0.87],
        [0.23,0.46,0.82,0.31,0.68,0.87],[0.17,0.29,0.80,0.21,0.62,0.87],[0.13,0.23,0.79,0.15,0.58,0.87]]
for i in range(6):
    x = [j*10 for j in range(6)]
    y = []
    for k in range(6):
        y1 = PSNR[k][i]
        y2 = SSIM[k][i]
        y.append(y2)
    data.append((x, y)) # Plotting the data
plt.figure(figsize=(8, 6))
colors = ['red', 'salmon', 'pink', 'blue', 'royalblue', 'dodgerblue']
labels = ['ZF_brain', 'DAGAN_brain', 'SwinGAN_brain', 'ZF_prostate', 'KIGAN_prostate', 'SwinGAN_prostate']
markers =['-o','-o','-o','--x','--x','--x']
for i, (x, y) in enumerate(data):
        plt.plot(x, y, markers[i],label=labels[i], color=colors[i],markersize=12)
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Line Chart for Six Groups of Data')
plt.legend()
plt.grid(False)
plt.show()