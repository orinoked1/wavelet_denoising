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
T_noise_only_min = 1e-4/10
T_noise_only_max = 1e-4*10
T_noise_list = 10**np.linspace(np.log10(T_noise_only_min),np.log10(T_noise_only_max),100)
T_r_min = 0.001/10
T_r_max = 0.001*10
T_r_list = 10**np.linspace(np.log10(T_r_min),np.log10(T_r_max),100)

n_noised = 1
best_score = np.zeros(n_noised)
best_T_r = np.zeros(n_noised)
best_T_noise_only = np.zeros(n_noised)
for i_noised in range(n_noised):
    # add noise
    noise_params = {'type':'gauss','mean': 0,'sigma': 20}
    original_noise = noisy( original, noise_params)
    score_list=np.zeros((len(T_noise_list),len(T_r_list)))
    for i_T_noise in range(len(T_noise_list)):
         for i_T_r in range(len(T_r_list)):
            # auto filter using the paper algo
            T_r = T_r_list[i_T_r]
            T_noise_only = T_noise_list[i_T_noise]
            algo_filter = AutoDwtFilter(T_r=T_r,T_noise_only=T_noise_only)
            try:
                filtered_ours, coeffs2_filt, k = algo_filter(original_noise)
                score_ours, diff = compare_ssim(original, filtered_ours, full=True)
                score_list[i_T_noise,i_T_r] = score_ours
            except:
                score_list[i_T_noise,i_T_r] = np.nan
    plt.figure()
    plt.imshow(score_list, cmap='coolwarm')
    plt.colorbar()
    plt.xlabel('T_r')
    plt.ylabel('T noise only')
    x_ticks = T_r_list[np.linspace(0, len(T_r_list) - 1, 20).astype(int)]
    x_ticks = ['%.0e' % i for i in x_ticks.tolist()]
    plt.xticks(np.linspace(0,len(T_r_list),20).astype(int), x_ticks, rotation=45)
    y_ticks = T_noise_list[np.linspace(0, len(T_noise_list) - 1, 20).astype(int)]
    y_ticks = ['%.0e' % i for i in y_ticks.tolist()]
    plt.yticks(np.linspace(0,len(T_noise_list),20).astype(int), y_ticks)


    best_score[i_noised] = np.nanmax(score_list)
    idx_conf = np.unravel_index(np.nanargmax(score_list),(len(T_noise_list),len(T_r_list)))
    best_T_r[i_noised] =T_r_list[idx_conf[1]]
    best_T_noise_only[i_noised]  = T_noise_list[idx_conf[0]]
    print('in noise rng ' + str(i_noised) + ' optimal score was ' + str(best_score[i_noised]) +
          ' with T_R '+str( best_T_r[i_noised]) +
          ' with T noise only '+str( best_T_noise_only[i_noised])  )
a=1