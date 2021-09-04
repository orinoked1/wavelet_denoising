from noisy import noisy
import matplotlib.pyplot as plt
import pywt.data
from auto_filter import AutoDwtFilter
from skimage.metrics import structural_similarity as compare_ssim
from scipy.stats import loguniform
import numpy as np
from skimage.restoration import (denoise_wavelet, estimate_sigma)
# get data
original = pywt.data.camera().astype('float')
# add noise
noise_params = {'mean': 0,
                'sigma': 20}
original_noise = noisy('gauss', original, noise_params)


T_noise_only_min = 0.000025/100
T_noise_only_max = 0.000025*100
T_noise_list = loguniform.rvs(T_noise_only_min, T_noise_only_max, size=20000)

T_r_min = 0.0025/100
T_r_max = 0.0025*100
T_r_list = loguniform.rvs(T_r_min, T_r_max, size=20000)
score_list=np.zeros_like(T_r_list)
for i_conf in range(len(T_r_list)):
    # auto filter using the paper algo
    T_r = T_r_list[i_conf]
    T_noise_only = T_noise_list[i_conf]
    algo_filter = AutoDwtFilter(T_r=T_r,T_noise_only=T_noise_only)
    try:
        filtered_ours, coeffs2_filt, k = algo_filter(original_noise)
        score_ours, diff = compare_ssim(original, filtered_ours, full=True)
        score_list[i_conf] = score_ours
    except:
        score_list[i_conf] = np.nan

best_score = np.nanmax(score_list)
idx_conf = np.nanargmax(score_list)
T_r_list[idx_conf]
T_noise_list[idx_conf]
a=1