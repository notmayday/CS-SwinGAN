import matplotlib.pyplot as plt
import numpy as np
data = []
PSNR = [[34.61,37.08,38.90,32.51,33.35,33.57],[31.52,35.18,34.45,30.46,32.99,32.79],[27.37,34.00,34.46,27.24,32.15,32.54],
        [24.08,32.45,34.48,24.48,31.17,32.39],[21.55,29.19,34.16,22.24,30.20,32.29],[19.53,25.15,33.78,20.39,29.29,32.21]]
SSIM = [[0.87,0.94,0.95,0.88,0.89,0.89],[0.74,0.89,0.82,0.79,0.88,0.88],[0.53,0.85,0.83,0.64,0.86,0.88],
        [0.39,0.78,0.84,0.51,0.83,0.88],[0.29,0.63,0.84,0.40,0.79,0.87],[0.23,0.46,0.82,0.32,0.75,0.87]]
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