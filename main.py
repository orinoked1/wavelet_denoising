from noisy import noisy
import matplotlib.pyplot as plt
import pywt.data
from auto_filter import AutoDwtFilter
from skimage.metrics import structural_similarity as compare_ssim


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
coeffs2 = pywt.wavedec2(original_noise, 'bior1.3', level=level)
coeffs2_arr, coeffs2_slices = pywt.coeffs_to_array(coeffs2)
plt.figure()
plt.imshow(coeffs2_arr, interpolation="nearest", cmap=plt.cm.gray)
plt.title('approximation')

# auto filter using the paper algo
algo_filter = AutoDwtFilter()
filtered_original, coeffs2_filt, k = algo_filter(original_noise)
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
