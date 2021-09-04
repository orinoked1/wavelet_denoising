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
T_noise_only = 1e-4
T_r = 0.001
# auto filter using the paper algo
algo_filter = AutoDwtFilter(T_r=T_r, T_noise_only=T_noise_only)
filtered_ours, coeffs2_filt, k_opt = algo_filter(original_noise)

# plot for different values of k
k_vals = range(1, 9)
scores_k_vals = []
for k_itr in k_vals:
    filtered_ours, coeffs2_filt, k = algo_filter(original_noise, k_itr)
    score_ours, diff = compare_ssim(original, filtered_ours, full=True)
    scores_k_vals.append(score_ours)
    print("SSIM original-filtered: {}".format(score_ours), " k_val:", k_itr)
    # plot the images for comparison
    # plt.figure()
    # plt.imshow(filtered_ours, interpolation="nearest", cmap=plt.cm.gray)
    # plt.title('k_itr ' + str(k_itr))
plt.figure()
plt.scatter(k_vals, scores_k_vals, c='b', label='values')
plt.scatter(k_opt, scores_k_vals[k_opt-1], c='r', label='opt')
plt.xlabel('k values')
plt.ylabel('SSIM score')
plt.title('SSIM as a function of k')
plt.legend()
plt.grid()

# skimage filters
# Estimate the average noise standard deviation across color channels.
# Due to clipping in random_noise, the estimate will be a bit smaller than the
filtered_bayes = denoise_wavelet(original_noise, method='BayesShrink', mode='soft',rescale_sigma=True)
sigma_est = estimate_sigma(original_noise, average_sigmas=True)
filtered_visushrink = denoise_wavelet(original_noise,method='VisuShrink', mode='soft',sigma=sigma_est, rescale_sigma=True)


# compare between the two images
score_noise, diff = compare_ssim(original, original_noise, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM original-noise: {}".format(score_noise))

score_ours, diff = compare_ssim(original, filtered_ours, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM original-filtered: {}".format(score_ours))

score_visushrink, diff = compare_ssim(original, filtered_visushrink, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM original- visushrink {}".format(score_visushrink))

score_bayes, diff = compare_ssim(original, filtered_bayes, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM original- bayes {}".format(score_bayes))

fig, axs = plt.subplots(1, 5)
axs[0].imshow(original, interpolation="nearest", cmap=plt.cm.gray)
axs[0].set_title('original')
axs[1].imshow(original_noise, interpolation="nearest", cmap=plt.cm.gray)
axs[1].set_title('noisy score:' + str(score_noise))
axs[2].imshow(filtered_ours, interpolation="nearest", cmap=plt.cm.gray)
axs[2].set_title('filtered score:' + str(score_ours))
axs[3].imshow(filtered_bayes, interpolation="nearest", cmap=plt.cm.gray)
axs[3].set_title('bayes score:' + str(score_bayes))
axs[4].imshow(filtered_visushrink, interpolation="nearest", cmap=plt.cm.gray)
axs[4].set_title('visushrink score:' + str(score_visushrink))
plt.show()

