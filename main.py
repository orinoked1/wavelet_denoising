from noisy import noisy
import matplotlib.pyplot as plt
import pywt.data
from auto_filter import AutoDwtFilter


# get data
original = pywt.data.camera().astype('float')
# add noise
noise_params = {'mean': 0,
                'sigma': 20}
original_noise = noisy('gauss', original, noise_params)
# plot noise
fig = plt.figure()
ax = plt.subplot(1, 2, 1)
ax.imshow(original, interpolation="nearest", cmap=plt.cm.gray)
plt.title('original')
ax = plt.subplot(1, 2, 2)
ax.imshow(original_noise, interpolation="nearest", cmap=plt.cm.gray)
plt.title('noisy')
fig.show()
# Wavelet transform of image, and plot approximation and details
level = 3
coeffs2 = pywt.wavedec2(original_noise, 'bior1.3', level=level)
coeffs2_arr, coeffs2_slices = pywt.coeffs_to_array(coeffs2)
fig = plt.figure()
plt.imshow(coeffs2_arr, interpolation="nearest", cmap=plt.cm.gray)
fig.show()
# auto filter using the paper algo
filter = AutoDwtFilter()
filtered_original ,coeffs2_filt,k= filter(original_noise)
fig = plt.figure()
ax = plt.subplot(1, 3, 1)
ax.imshow(original, interpolation="nearest", cmap=plt.cm.gray)
plt.title('original')
ax = plt.subplot(1, 3, 2)
ax.imshow(original_noise, interpolation="nearest", cmap=plt.cm.gray)
plt.title('noisy')
ax = plt.subplot(1, 3, 3)
ax.imshow(filtered_original, interpolation="nearest", cmap=plt.cm.gray)
plt.title('filtered')
fig.show()
a=1