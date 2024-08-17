import matplotlib.pyplot as plt
import yaml
from types import SimpleNamespace
import logging
import os
import utils.metrics as mt
import numpy as np
import torch
from math import log10
import glob
import h5py
from pprint import pprint
import pickle
import scipy.io
from Networks.generator_PRE import Generator_PRE
from Networks.generator_model import WNet
from utils.data_vis import plot_imgs,save_imgs
from utils.data_save import save_data


def crop_toshape(kspace_cplx, args):
    if kspace_cplx.shape[0] == args.img_size:
        return kspace_cplx
    if kspace_cplx.shape[0] % 2 == 1:
        kspace_cplx = kspace_cplx[:-1, :-1]
    crop = int((kspace_cplx.shape[0] - args.img_size) / 2)
    kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]

    return kspace_cplx


def fft2(img):
    return np.fft.fftshift(np.fft.fft2(img))

def ifft2(kspace_cplx):
    return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]


def slice_preprocess(kspace_cplx, slice_num, masks, maskedNot, args):
    # crop to fix size
    kspace_cplx = crop_toshape(kspace_cplx, args)
    # split to real and imaginary channels
    kspace = np.zeros((args.img_size, args.img_size, 2))
    kspace[:, :, 0] = np.real(kspace_cplx).astype(np.float32)
    kspace[:, :, 1] = np.imag(kspace_cplx).astype(np.float32)
    # target image:
    image = np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]

    # HWC to CHW
    delta =0
    kspace = kspace.transpose(2, 0, 1)
    masked_Kspace = kspace * masks[:, :, slice_num]
    masked_Kspace_Noisy = masked_Kspace + np.random.normal(0, delta, masked_Kspace.shape)
    masked_Kspace_Pure = masked_Kspace
    return masked_Kspace_Noisy, masked_Kspace_Pure, kspace, image


def get_args():
    with open('config.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    args = SimpleNamespace(**data)

    args.masked_kspace = True
    args.mask_path = '/home/samuel/SwinGAN-main3.5/Masks/poisson/poisson_{}_{}_{}.mat'.format(args.img_size, args.img_size, args.sampling_percentage)
    pprint(data)

    return args


def generate_rician_noise(size, v, sigma):
    """
    Generate a 2D Rician noise array.

    Parameters:
    size (tuple): Size of the 2D array (height, width).
    v (float): Non-centrality parameter (mean of the underlying Gaussian distributions).
    sigma (float): Standard deviation of the underlying Gaussian distributions.

    Returns:
    np.ndarray: 2D array with Rician noise.
    """
    # Generate two independent Gaussian noise arrays
    noise1 = np.random.normal(v, sigma, size)
    noise2 = np.random.normal(0, sigma, size)

    # Combine the Gaussian noise arrays to form Rician noise
    rician_noise = np.sqrt(noise1 ** 2 + noise2 ** 2)

    return rician_noise
def mean_squared_error(image0, image1):
    """
    Compute the mean-squared error between two images.

    Parameters
    ----------
    image0, image1 : ndarray
        Images.  Any dimensionality, must have same shape.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_mse`` to
        ``skimage.metrics.mean_squared_error``.

    """
    check_shape_equality(image0, image1)
    image0, image1 = _as_floats(image0, image1)
    return np.mean((image0 - image1) ** 2, dtype=np.float64)

def check_shape_equality(im1, im2):
    """Raise an error if the shape do not match."""
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    return
def _as_floats(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    float_type = np.result_type(image0.dtype, image1.dtype, np.float32)
    image0 = np.asarray(image0, dtype=float_type)
    image1 = np.asarray(image1, dtype=float_type)
    return image0, image1

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    # Set device and GPU (currently only single GPU training is supported
    logging.info(f'Using device {args.device}')
    if args.device == 'cuda':
        logging.info(f'Using GPU {args.gpu_id}')
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    nmse_meter = mt.AverageMeter()
    psnr_meter = mt.AverageMeter()
    ssim_meter = mt.AverageMeter()
    snr1_meter = mt.AverageMeter()
    snr2_meter = mt.AverageMeter()
    snrk_meter = mt.AverageMeter()

    nmse_meter_ZF = mt.AverageMeter()
    psnr_meter_ZF = mt.AverageMeter()
    ssim_meter_ZF = mt.AverageMeter()
    snr_meter_ZF = mt.AverageMeter()

    # Load network
    logging.info("Loading model {}".format(args.model))
    net_PRE = Generator_PRE(args, masked_kspace=args.masked_kspace)
    net = WNet(args, masked_kspace=args.masked_kspace)
    net.to(device=args.device)
    net_PRE.to(device=args.device)

    checkpoint = torch.load(args.model, map_location=args.device)
    net.load_state_dict(checkpoint['G_model_state_dict'])
    net.eval()
    net_PRE.load_state_dict(checkpoint['PRE_model_state_dict'])
    net_PRE.eval()

    logging.info("Model loaded !")

    masks_dictionary = scipy.io.loadmat(args.mask_path)
    masks = np.dstack((masks_dictionary['population_matrix'], masks_dictionary['population_matrix'], masks_dictionary['population_matrix']))

    plt.figure(figsize=(8, 6))
    plt.imshow(masks[:, :, 1], cmap='gray', vmin=0, vmax=1)

    plt.axis('off')
    plt.show()

    maskNot = 1 - masks_dictionary['population_matrix']
    total_nmse=[]
    total_psnr=[]
    total_ssim=[]
    total_snr1 = []
    total_snr2= []
    total_snrk = []
    total_nmse_ZF=[]
    total_psnr_ZF=[]
    total_ssim_ZF=[]
    total_snr_ZF = []

    test_files = glob.glob(os.path.join(args.predict_data_dir, '*.hdf5'))
    for i, infile in enumerate(test_files):

        with h5py.File(infile, 'r') as f:
            fully_sampled_imgs = f['data']
            img_shape = fully_sampled_imgs.shape
            fully_sampled_imgs = np.array(fully_sampled_imgs)
        #Preprocess data:
        rec_imgs = np.zeros(img_shape)
        rec_imgs_Pure = np.zeros(img_shape)
        rec_Kspaces = np.zeros(img_shape, dtype=np.csingle) #complex
        F_rec_Kspaces = np.zeros(img_shape)

        ZF_img = np.zeros(img_shape)
        ZF_img_Pure = np.zeros(img_shape)
        for slice_num in range(1,img_shape[2]-1):
            add = int(args.num_input_slices / 2)
            with h5py.File(infile, 'r') as f:
                imgs = np.array(f['data'][:, :, slice_num])

            masked_Kspaces_np = np.zeros((args.num_input_slices * 2, args.img_size, args.img_size))
            masked_Kspaces_np_Pure = np.zeros((args.num_input_slices * 2, args.img_size, args.img_size))
            target_Kspace = np.zeros((2, args.img_size, args.img_size))
            target_img = np.zeros((1, args.img_size, args.img_size))
            enh_img = np.zeros((1, args.img_size, args.img_size))
            enh_Kspace = np.zeros((2, args.img_size, args.img_size))
            fully_sampled_imgs[:,:,slice_num] = (fully_sampled_imgs[:,:,slice_num] - np.min(fully_sampled_imgs[:,:,slice_num])) / (np.max(fully_sampled_imgs[:,:,slice_num]) - np.min(fully_sampled_imgs[:,:,slice_num]))

            for slice_j in range(args.num_input_slices):
                img = imgs[:, :]
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                kspace = np.fft.fftshift(np.fft.fft2(img))
                slice_masked_Kspace_Noisy, slice_masked_Kspace_Pure, slice_full_Kspace, slice_full_img = slice_preprocess(kspace, slice_j,
                                                                                          masks, maskNot, args)
                masked_Kspaces_np[slice_j * 2:slice_j * 2 + 2, :, :] = slice_masked_Kspace_Noisy
                masked_Kspaces_np_Pure[slice_j * 2:slice_j * 2 + 2, :, :] = slice_masked_Kspace_Pure

                if slice_j == int(args.num_input_slices / 2):
                    target_Kspace = slice_full_Kspace
                    target_img = slice_full_img
            masked_Kspaces = np.expand_dims(masked_Kspaces_np, axis=0)
            masked_Kspaces_Pure = np.expand_dims(masked_Kspaces_np_Pure, axis=0)
            ## SNR for original image
            noisy_image = abs(np.fft.ifft2((masked_Kspaces_np[0, :, :] + 1j*masked_Kspaces_np[1, :, :])))
            pure_image = abs(np.fft.ifft2((masked_Kspaces_np_Pure[0, :, :] + 1j*masked_Kspaces_np_Pure[1, :, :])))
            mean_image = np.mean(pure_image)

            std_noise = np.std(np.abs(noisy_image - pure_image))
            snr1 = mean_image / std_noise

            masked_Kspaces = torch.from_numpy(masked_Kspaces).to(device=args.device, dtype=torch.float32)
            masked_Kspaces_Pure = torch.from_numpy(masked_Kspaces_Pure).to(device=args.device, dtype=torch.float32)





            #Predict:
            enh_Kspace, enh_img, ori_img = net_PRE(masked_Kspaces)
            rec_img, rec_Kspace, F_rec_Kspace = net(enh_Kspace, enh_img)
            enh_Kspace_Pure, enh_img_Pure, ori_img_Pure = net_PRE(masked_Kspaces_Pure)
            rec_img_Pure, rec_Kspace_Pure, F_rec_Kspace_Pure = net(enh_Kspace_Pure, enh_img_Pure)

            rec_img = np.squeeze(rec_img.data.cpu().numpy())
            rec_Kspace = np.squeeze(rec_Kspace.data.cpu().numpy())
            rec_Kspace = (rec_Kspace[0, :, :] + 1j*rec_Kspace[1, :, :])

            rec_img_Pure = np.squeeze(rec_img_Pure.data.cpu().numpy())


            F_rec_Kspace = np.squeeze(F_rec_Kspace.data.cpu().numpy())

            rec_imgs[:, :, slice_num] = rec_img
            rec_imgs_Pure[:, :, slice_num] = rec_img_Pure


            rec_Kspaces[:, :, slice_num] = rec_Kspace

            F_rec_Kspaces[:, :, slice_num] = F_rec_Kspace
            ZF_img[:, :, slice_num] = np.squeeze(ifft2((masked_Kspaces_np[0, :, :] + 1j*masked_Kspaces_np[1, :, :])))
            ZF_img_Pure[:, :, slice_num] = np.squeeze(ifft2((masked_Kspaces_np_Pure[0, :, :] + 1j * masked_Kspaces_np_Pure[1, :, :])))

            fully_sampled_img =fully_sampled_imgs[:, :, slice_num]
            fully_sampled_img = (fully_sampled_img - np.min(fully_sampled_img)) / (np.max(fully_sampled_img) - np.min(fully_sampled_img))
            nmse=mt.nmse(fully_sampled_img,rec_imgs[:, :, slice_num])
            psnr=mt.psnr(fully_sampled_img,rec_imgs[:, :, slice_num])
            ssim=mt.ssim(fully_sampled_img,rec_imgs[:, :, slice_num])
            mean_image = np.mean(rec_imgs_Pure[:, :, slice_num])
            std_noise = np.std(np.abs(rec_imgs[:, :, slice_num] - rec_imgs_Pure[:, :, slice_num]))
            snr2 = mean_image/std_noise


            nmse_ZF = mt.nmse(fully_sampled_img, ZF_img[:, :, slice_num])
            psnr_ZF = mt.psnr(fully_sampled_img, ZF_img[:, :, slice_num])
            ssim_ZF = mt.ssim(fully_sampled_img, ZF_img[:, :, slice_num])
            mean_image_ZF = np.mean(ZF_img_Pure[:, :, slice_num])
            std_noise_ZF = np.std(np.abs(ZF_img[:, :, slice_num] - ZF_img_Pure[:, :, slice_num]))
            snr_ZF = mean_image_ZF / std_noise_ZF


            nmse_meter.update(nmse, 1)
            psnr_meter.update(psnr, 1)
            ssim_meter.update(ssim, 1)
            snr1_meter.update(snr1, 1)
            snr2_meter.update(snr2, 1)
            # snrk_meter.update(snrk, 1)
            nmse_meter_ZF.update(nmse_ZF, 1)
            psnr_meter_ZF.update(psnr_ZF, 1)
            ssim_meter_ZF.update(ssim_ZF, 1)
            snr_meter_ZF.update(snr_ZF, 1)


            error_map = np.abs(fully_sampled_imgs[:, :, slice_num] - rec_imgs[:, :, slice_num])
            error_map_ZF = np.abs(fully_sampled_imgs[:, :, slice_num] - ZF_img[:, :, slice_num])
            error_map_GT = np.abs(fully_sampled_imgs[:, :, slice_num] - fully_sampled_imgs[:, :, slice_num])

            window_level = 0.5 # Adjust this to change the level of the window
            window_size = 1  # Adjust this to change the size of the window
            n_bins = 256

            # Calculate the window range
            window_min = window_level - window_size / 2
            window_max = window_level + window_size / 2
            if slice_num == 1:
                cmap = 'jet'

                plt.figure(figsize=(8, 6))
                plt.imshow(np.clip(fully_sampled_imgs[:, :, slice_num], window_min, window_max), cmap='gray', vmin=0,
                           vmax=1)
                # plt.colorbar(label='Reconstructed image')
                # plt.title('Ground Truth')
                plt.axis('off')
                plt.show()

                plt.figure(figsize=(8, 6))
                plt.imshow(np.clip(error_map_GT, window_min, window_max), cmap=cmap, vmin=0, vmax=0.3)
                # plt.colorbar(label='Error map')
                # plt.title('Error Map (GT)')
                plt.axis('off')
                plt.show()
                kk = np.clip(rec_imgs[:, :, slice_num], window_min, window_max)
                plt.figure(figsize=(8, 6))
                plt.imshow(np.clip(rec_imgs[:, :, slice_num], window_min, window_max), cmap='gray', vmin=0, vmax=1)
                # plt.colorbar(label='Error')
                # plt.title('Reconstructed Img (SwinGAN)')
                plt.axis('off')
                plt.show()

                plt.figure(figsize=(8, 6))
                plt.imshow(np.clip(error_map, window_min, window_max), cmap=cmap, vmin=0, vmax=0.3)
                # plt.colorbar(label='Error')
                # plt.title('Error Map (SwinGAN)')
                plt.axis('off')
                plt.show()

                plt.figure(figsize=(8, 6))
                plt.imshow(np.clip(ZF_img[:, :, slice_num], window_min, window_max), cmap='gray', vmin=0, vmax=1)
                # plt.colorbar(label='Error')
                # plt.title('Reconstructed Img (ZF)')
                plt.axis('off')
                plt.show()

                plt.figure(figsize=(8, 6))
                plt.imshow(np.clip(error_map_ZF, window_min, window_max), cmap=cmap, vmin=0, vmax=0.3)
                # plt.colorbar(label='Error')
                # plt.title('Error Map (ZF)')
                plt.axis('off')
                plt.show()

                # plt.figure(figsize=(8, 6))
                # fig, ax = plt.subplots(sharey=True, tight_layout=True)
                # hist = ax.hist(error_map,
                #                bins=n_bins, alpha = 0.8, range = (0,0.3), histtype= 'bar', edgecolor='black', linewidth = 1)
                # # plt.colorbar(label='Error')
                # # plt.title('Error Map (SwinGAN)')
                # # plt.axis('off')
                # plt.show()
                #
                # plt.figure(figsize=(8, 6))
                # fig, ax = plt.subplots(sharey=True, tight_layout=True)
                # hist = ax.hist(error_map_ZF,
                #                bins=n_bins, alpha = 0.8, range = (0,0.3), histtype= 'bar', edgecolor='black', linewidth = 1)
                # # plt.colorbar(label='Error')
                # # plt.title('Error Map (SwinGAN)')
                # #plt.axis('off')
                # plt.show()


        total_nmse.append(nmse_meter.avg)
        total_psnr.append(psnr_meter.avg)
        total_ssim.append(ssim_meter.avg)
        total_snr1.append(snr1_meter.avg)
        total_snr2.append(snr2_meter.avg)
        # total_snrk.append(snrk_meter.avg)
        total_nmse_ZF.append(nmse_meter_ZF.avg)
        total_psnr_ZF.append(psnr_meter_ZF.avg)
        total_ssim_ZF.append(ssim_meter_ZF.avg)
        total_snr_ZF.append(snr_meter_ZF.avg)

        print("==> Evaluate Metric of image{}".format(infile))
        print("Results ----------")
        print("NMSE: {:.4}".format(nmse_meter.avg))
        print("PSNR: {:.4}".format(psnr_meter.avg))
        print("SSIM: {:.4}".format(ssim_meter.avg))
        print("SNR1: {:.4}".format(snr1_meter.avg))
        print("SNR2: {:.4}".format(snr2_meter.avg))
        # print("SNR_K: {:.4}".format(snrk_meter.avg))
        print("NMSE_ZF: {:.4}".format(nmse_meter_ZF.avg))
        print("PSNR_ZF: {:.4}".format(psnr_meter_ZF.avg))
        print("SSIM_ZF: {:.4}".format(ssim_meter_ZF.avg))
        print("SNR_ZF: {:.4}".format(snr_meter_ZF.avg))
        print("------------------")

        if args.save_prediction:
            os.makedirs(args.save_path, exist_ok=True)
            out_file_name = args.save_path +'/'+os.path.split(infile)[1]
            save_data(out_file_name, rec_imgs, F_rec_Kspaces, fully_sampled_imgs, ZF_img, rec_Kspaces)

            logging.info("reconstructions save to: {}".format(out_file_name))

        if args.visualize_images:
            logging.info("Visualizing results for image {}, close to continue ...".format(infile))
            # plot_imgs(rec_imgs, F_rec_Kspaces, fully_sampled_imgs, ZF_img)
        # save_imgs(rec_imgs,F_rec_Kspaces,fully_sampled_imgs, ZF_img)

    print("avg_NMSE: {:.4}".format(np.mean(total_nmse)))
    print("avg_PSNR: {:.4}".format(np.mean(total_psnr)))
    print("avg_SSIM: {:.4}".format(np.mean(total_ssim)))
    print("avg_SNR1: {:.4}".format(np.mean(total_snr1)))
    print("avg_SNR2: {:.4}".format(np.mean(total_snr2)))
    # print("avg_SNR_K: {:.4}".format(np.mean(total_snrk)))

    print("std_NMSE: {:.4}".format(np.std(total_nmse)))
    print("std_PSNR: {:.4}".format(np.std(total_psnr)))
    print("std_SSIM: {:.4}".format(np.std(total_ssim)))
    print("std_SNR1: {:.4}".format(np.std(total_snr1)))
    print("std_SNR2: {:.4}".format(np.std(total_snr2)))
    # print("std_SNR_K: {:.4}".format(np.std(total_snrk)))

    print("avg_NMSE_ZF: {:.4}".format(np.mean(total_nmse_ZF)))
    print("avg_PSNR_ZF: {:.4}".format(np.mean(total_psnr_ZF)))
    print("avg_SSIM_ZF: {:.4}".format(np.mean(total_ssim_ZF)))
    print("avg_SNR_ZF: {:.4}".format(np.mean(total_snr_ZF)))

    print("std_NMSE_ZF: {:.4}".format(np.std(total_nmse_ZF)))
    print("std_PSNR_ZF: {:.4}".format(np.std(total_psnr_ZF)))
    print("std_SSIM_ZF: {:.4}".format(np.std(total_ssim_ZF)))
    print("std_SNR_ZF: {:.4}".format(np.std(total_snr_ZF)))

