import numpy as np
import pywt
from scipy.optimize import minimize
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from multiprocessing import Pool, cpu_count
import cv2
import matplotlib.pyplot as plt
# Placeholder functions for wavelet transform and inverse
def wavelet_transform(img, wavelet='db1'):
    coeffs = pywt.wavedecn(img, wavelet)
    return coeffs


def inverse_wavelet_transform(coeffs, wavelet='db1'):
    img_reconstructed = pywt.waverecn(coeffs, wavelet)
    return img_reconstructed


def flatten_coeffs(coeffs):
    coeffs_flat, coeffs_slices = pywt.coeffs_to_array(coeffs)
    return coeffs_flat, coeffs_slices


def unflatten_coeffs(coeffs_flat, coeffs_slices):
    try:
        coeffs_flat = coeffs_flat.reshape([256, 256])
        coeffs = pywt.array_to_coeffs(coeffs_flat, coeffs_slices, output_format='wavedecn')
        return coeffs
    except Exception as e:
        print(f"Error in unflatten_coeffs: {str(e)}")
        print(f"coeffs_flat shape: {coeffs_flat.shape}")
        print(f"coeffs_slices: {coeffs_slices}")
        raise e


# Undersample k-space function
def undersample_kspace(img, mask):
    kspace = fftshift(fft2(img))
    undersampled_kspace = kspace * mask
    return undersampled_kspace


def total_variation(img):
    """
    Calculate the total variation of an image.
    """
    # Pad the image to ensure consistent shape for differences
    dx = np.diff(img, axis=0)
    dy = np.diff(img, axis=1)
    dx = np.pad(dx, ((0,1), (0,0)), mode='constant')
    dy = np.pad(dy, ((0,0), (0,1)), mode='constant')

    # Calculate total variation
    tv = np.sum(np.sqrt(dx**2 + dy**2))
    return tv



def objective_function(params, tv_weight):
    x, mask, original_kspace, wavelet, coeffs_slices = params
    coeffs = unflatten_coeffs(x, coeffs_slices)
    img_reconstructed = inverse_wavelet_transform(coeffs, wavelet)
# ####################
#     window_level = 0.5  # Adjust this to change the level of the window
#     window_size = 1  # Adjust this to change the size of the window
#     n_bins = 256
#     # Calculate the window range
#     window_min = window_level - window_size / 2
#     window_max = window_level + window_size / 2
#     plt.figure(figsize=(8, 6))
#     plt.imshow(np.clip(img_reconstructed, window_min, window_max), cmap='gray', vmin=0, vmax=1)
#     # plt.colorbar(label='Error')
#     # plt.title('Reconstructed Img (SwinGAN)')
#     plt.axis('off')
#     plt.show()
#    ##########################################
    kspace_reconstructed = fftshift(fft2(img_reconstructed))
    error = np.linalg.norm((kspace_reconstructed - original_kspace) * mask)
    tv_penalty = tv_weight * total_variation(img_reconstructed)
    total_error = error + tv_penalty
    return total_error


def evaluate_objective_function(args):
    params, tv_weight = args
    return objective_function(params, tv_weight)


def parallel_objective_function(x, mask, original_kspace, wavelet, coeffs_slices, tv_weight):
    params = [(x, mask, original_kspace, wavelet, coeffs_slices)] * cpu_count()
    args = [(p, tv_weight) for p in params]
    with Pool(cpu_count()) as pool:
        results = pool.map(evaluate_objective_function, args)
    return results[0]


def compressive_sensing_denoising(img, mask, wavelet='db1', tv_weight=0.1, maxiter=1, tol=100):
    undersampled_kspace = undersample_kspace(img, mask)
    initial_coeffs = wavelet_transform(ifft2(ifftshift(undersampled_kspace)).real, wavelet)
    coeffs_flat, coeffs_slices = flatten_coeffs(initial_coeffs)
    coeffs_flat = coeffs_flat.ravel()

    print(f"Initial coeffs_flat shape: {coeffs_flat.shape}")
    print(f"Initial coeffs_slices: {coeffs_slices}")

    def callback(xk):
        nonlocal prev_fval
        fval = parallel_objective_function(xk, mask, undersampled_kspace, wavelet, coeffs_slices, tv_weight)
        if prev_fval is not None and abs(prev_fval - fval) < tol:
            print(f"Optimization terminated because change in objective function is below tolerance ({tol})")
            return True
        prev_fval = fval

    prev_fval = None

    result = minimize(parallel_objective_function, coeffs_flat, args=(mask, undersampled_kspace, wavelet, coeffs_slices, tv_weight), tol=100,
                      method='Powell', options={'maxiter': maxiter}, callback=callback)

    optimized_coeffs_flat = result.x
    optimized_coeffs = unflatten_coeffs(optimized_coeffs_flat, coeffs_slices)
    img_reconstructed = inverse_wavelet_transform(optimized_coeffs, wavelet)

    return img_reconstructed


# Example usage
if __name__ == "__main__":
    img = cv2.imread('RI30_GD_A_41.png', cv2.IMREAD_UNCHANGED)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    ####################
    window_level = 0.5  # Adjust this to change the level of the window
    window_size = 1  # Adjust this to change the size of the window
    n_bins = 256
    # # Calculate the window range
    # window_min = window_level - window_size / 2
    # window_max = window_level + window_size / 2
    # plt.figure(figsize=(8, 6))
    # plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    # # plt.colorbar(label='Error')
    # # plt.title('Reconstructed Img (SwinGAN)')
    # plt.axis('off')
    # plt.show()
    # ##########################################
    img = cv2.resize(img, (256, 256))
    img = img + np.random.normal(0, 0.1, [256,256])
    mask = np.random.rand(256, 256) > 0
    # ####################
    # window_level = 0.5  # Adjust this to change the level of the window
    # window_size = 1  # Adjust this to change the size of the window
    # n_bins = 256
    # # Calculate the window range
    # window_min = window_level - window_size / 2
    # window_max = window_level + window_size / 2
    # plt.figure(figsize=(8, 6))
    # plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    # # plt.colorbar(label='Error')
    # # plt.title('Reconstructed Img (SwinGAN)')
    # plt.axis('off')
    # plt.show()
    # ##########################################
    denoised_img = compressive_sensing_denoising(img, mask, tv_weight=0.1)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised_img, cmap='gray')

    plt.show()