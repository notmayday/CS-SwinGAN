""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
import pickle
from scipy.io import loadmat, savemat
from .unet_parts import *
import cv2
import scipy
# import vision_transformer as SwinUnet
from .vision_transformer import *
import matplotlib.pyplot as plt
class UNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear=True):  #频域生成器初始参数 (self,6,2,True) 图像域初始参数 (self,1,1,True)
        """U-Net  #https://github.com/milesial/Pytorch-UNet
        """

        super(UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels_in, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

class WNet(nn.Module):

    def __init__(self, args, masked_kspace=True):
        super(WNet, self).__init__()

        self.bilinear = args.bilinear
        self.args = args
        self.masked_kspace = masked_kspace



        mask_path = args.mask_path

        if args.mask_type=='radial':
            maskfile_path = './Masks/radial/radial_{}.tif'.format(args.sampling_percentage)

            mask_shift = cv2.imread(maskfile_path, 0) / 255
            mask_shift = np.fft.fftshift(mask_shift)

            self.mask = torch.tensor(mask_shift == 1, device=self.args.device)

        elif args.mask_type=='random':
            with open(mask_path, 'rb') as pickle_file:
                masks = pickle.load(pickle_file)
                self.mask = torch.tensor(masks['mask1'] == 1, device=self.args.device)
        else:
            masks = loadmat(mask_path)
            try:
                self.mask = torch.tensor(masks['Umask'] == 1, device=self.args.device)
            except:
                try:
                    self.mask = torch.tensor(masks['maskRS2'] == 1, device=self.args.device)
                except:
                    self.mask = torch.tensor(masks['population_matrix'] == 1, device=self.args.device)

        self.maskNot = self.mask == 0

        if self.args.ST:
            self.kspace_Unet = SwinUnet(args, img_size=args.img_size, num_classes=args.num_classes + 1,in_chans=2).cuda()
            self.img_UNet_real = UNet(n_channels_in=1, n_channels_out=1, bilinear=self.bilinear)
            self.img_UNet_imag = UNet(n_channels_in=1, n_channels_out=1, bilinear=self.bilinear)
        else:
            #调用k空间U-Net和图像域U-Net
            self.kspace_Unet = UNet(n_channels_in=args.num_input_slices*2, n_channels_out=2, bilinear=self.bilinear)
            self.img_UNet = UNet(n_channels_in=1, n_channels_out=1, bilinear=self.bilinear)

    def fftshift(self, img):

        S = int(img.shape[3]/2)
        img2 = torch.zeros_like(img)
        img2[:, :, :S, :S] = img[:, :, S:, S:]
        img2[:, :, S:, S:] = img[:, :, :S, :S]
        img2[:, :, :S, S:] = img[:, :, S:, :S]
        img2[:, :, S:, :S] = img[:, :, :S, S:]
        return img2

    def inverseFT(self, Kspace):
        Kspace = Kspace.permute(0, 2, 3, 1)
        K = Kspace[:,:,:,0] + 1j * Kspace[:,:,:,1]
        img_cmplx = torch.fft.ifft2(K, dim=[1,2])
        img = torch.zeros(img_cmplx.size(0), 2, img_cmplx.size(1), img_cmplx.size(2))
        img[:, 0, :, :] = img_cmplx.real
        img[:, 1, :, :] = img_cmplx.imag
        return img

    def forward(self, Kspace, enh_img):
        rec_all_Kspace = self.kspace_Unet(Kspace)
        if self.masked_kspace:
            rec_Kspace = self.mask*Kspace[:, int(Kspace.shape[1]/2)-1:int(Kspace.shape[1]/2)+1, :, :] +\
                         self.maskNot*rec_all_Kspace
            rec_mid_img = self.inverseFT(rec_Kspace)
            rec_mid_img_real = rec_mid_img[:,0,:,:]
            rec_mid_img_real = rec_mid_img_real[:, None, :, :]
            rec_mid_img_imag = rec_mid_img[:,1,:,:]
            rec_mid_img_imag = rec_mid_img_imag[:, None, :, :]
            rec_mid_img = torch.sqrt(rec_mid_img_real**2+rec_mid_img_imag**2)
            rec_mid_img = torch.tanh(rec_mid_img)
            rec_mid_img = torch.clamp(rec_mid_img, 0, 1)


        else:
            rec_Kspace = rec_all_Kspace
            rec_mid_img = self.fftshift(self.inverseFT(rec_Kspace))
            rec_mid_img_real = rec_mid_img[:,0,:,:]
            rec_mid_img_real = rec_mid_img_real[:, None, :, :]
            rec_mid_img_imag = rec_mid_img[:,1,:,:]
            rec_mid_img_imag = rec_mid_img_imag[:, None, :, :]
            rec_mid_img = torch.sqrt(rec_mid_img_real**2+rec_mid_img_imag**2)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rec_mid_img_real = rec_mid_img_real.to(device)
        rec_mid_img_imag = rec_mid_img_imag.to(device)
        rec_mid_img = rec_mid_img.to(device)

        refine_Img_real = self.img_UNet_real(rec_mid_img_real)
        refine_Img_imag = self.img_UNet_imag(rec_mid_img_imag)
        refine_Img=torch.sqrt(refine_Img_real**2+refine_Img_imag**2)
        refine_Img = torch.tanh(refine_Img)
        refine_Img = torch.clamp(refine_Img, 0, 1)
        rec_img = torch.tanh(0.75 * refine_Img + 0.25 * rec_mid_img + 0 * enh_img)
        rec_img = torch.clamp(rec_img, 0, 1)
        #########################################


        return rec_img, rec_Kspace, rec_mid_img
