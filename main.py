from noisy import noisy
import matplotlib.pyplot as plt
import pywt.data
from auto_filter import AutoDwtFilter
from skimage.metrics import structural_similarity as compare_ssim

from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio

# get data
original = pywt.data.camera().astype('float')
# add noise
noise_params = {'mean': 0,
                'sigma': 20}
original_noise = noisy('gauss', original, noise_params)
# plot original and noise
fig, axs = plt.subplots(1, 2)
axs[0].imshow(original, interpolation="nearest", cmap=plt.cm.gray)
axs[0].set_title('original')
axs[1].imshow(original_noise, interpolation="nearest", cmap=plt.cm.gray)
axs[1].set_title('noisy')

# Wavelet transform of image, and plot approximation and details
level = 3
coeffs2 = pywt.wavedec2(original_noise, 'db1', level=level)
coeffs2_arr, coeffs2_slices = pywt.coeffs_to_array(coeffs2)
plt.figure()
plt.imshow(coeffs2_arr, interpolation="nearest", cmap=plt.cm.gray)
plt.title('approximation')

# auto filter using the paper algo
algo_filter = AutoDwtFilter()
filtered_original, coeffs2_filt, k = algo_filter(original_noise)
# skimage filters
# Estimate the average noise standard deviation across color channels.
# Due to clipping in random_noise, the estimate will be a bit smaller than the
im_bayes = denoise_wavelet(original_noise, method='BayesShrink', mode='soft',rescale_sigma=True)
sigma_est = estimate_sigma(original_noise, average_sigmas=True)
im_visushrink = denoise_wavelet(original_noise,method='VisuShrink', mode='soft',sigma=sigma_est, rescale_sigma=True)


fig, axs = plt.subplots(1, 3)
axs[0].imshow(original, interpolation="nearest", cmap=plt.cm.gray)
axs[0].set_title('original')
axs[1].imshow(original_noise, interpolation="nearest", cmap=plt.cm.gray)
axs[1].set_title('noisy')
axs[2].imshow(filtered_original, interpolation="nearest", cmap=plt.cm.gray)
axs[2].set_title('filtered')
plt.show()


# compare between the two images
score, diff = compare_ssim(original, original_noise, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM original-noise: {}".format(score))

score, diff = compare_ssim(original, filtered_original, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM original-filtered: {}".format(score))
