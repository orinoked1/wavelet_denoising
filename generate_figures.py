from noisy import noisy
import matplotlib.pyplot as plt
import pywt.data
from auto_filter import AutoDwtFilter
from skimage.metrics import structural_similarity as compare_ssim
from skimage.restoration import denoise_wavelet
from skimage import data
import os
out_path  = r"C:\Users\User\Downloads\nd_project"
# get data
camera = pywt.data.camera().astype('float')
astronaut = data.astronaut().mean(2)
aero = pywt.data.aero().astype('float')
ascent = pywt.data.ascent().astype('float')
image_list = [camera,astronaut,aero,ascent]
# get noise params
noise_params_list =[ {'name':'LG','type': 'gauss','mean': 0,'sigma': 20},
                     {'name':'HG','type': 'gauss','mean': 10,'sigma': 40},
                     {'name':'S&P','type': 's&p','s_vs_p': 0.5,'amount': 0.1}]
T_noise_only = 1e-4
T_r = 0.001

for i_image,image in enumerate(image_list):
    fig, axs = plt.subplots(3, 4)

    for i_noise,noise_params in enumerate(noise_params_list):
        # add noise
        image_noise = noisy(image, noise_params)
        algo_filter = AutoDwtFilter(T_r=T_r, T_noise_only=T_noise_only)
        filtered_ours, coeffs2_filt, k_opt = algo_filter(image_noise)
        filtered_bayes = denoise_wavelet(image_noise, method='BayesShrink', mode='soft', rescale_sigma=True)
        score_noise, _ = compare_ssim(image, image_noise, full=True)
        score_ours, _ = compare_ssim(image, filtered_ours, full=True)
        score_bayes, _ = compare_ssim(image, filtered_bayes, full=True)
        axs[i_noise,0].imshow(image, interpolation="nearest", cmap=plt.cm.gray)
        axs[i_noise,0].tick_params(axis='both',which='both',left=False,bottom=False,labelleft=False,labelbottom=False)
        axs[i_noise,0].set_title('original image')
        axs[i_noise,1].imshow(image_noise, interpolation="nearest", cmap=plt.cm.gray)
        axs[i_noise, 1].tick_params(axis='both', which='both', left=False,bottom=False, labelleft=False, labelbottom=False)
        axs[i_noise,1].set_title(noise_params['name'] + ' noised image SSIM:{:.3f}'.format(score_noise) )
        axs[i_noise,2].imshow(filtered_ours, interpolation="nearest", cmap=plt.cm.gray)
        axs[i_noise, 2].tick_params(axis='both', which='both', left=False,bottom=False, labelleft=False, labelbottom=False)
        axs[i_noise,2].set_title(noise_params['name'] + ' denoised image ours SSIM:{:.3f}'.format(score_ours) )
        axs[i_noise,3].imshow(filtered_bayes, interpolation="nearest", cmap=plt.cm.gray)
        axs[i_noise, 3].tick_params(axis='both', which='both', left=False,bottom=False, labelleft=False, labelbottom=False)
        axs[i_noise,3].set_title(noise_params['name'] + ' denoised image BayesShrink SSIM:{:.3f}'.format(score_bayes))
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(32, 18)  # set figure's size manually to your full screen (32x18)
    plt.savefig(os.path.join(out_path,str(i_image)+'.png'), bbox_inches='tight')
