import numpy as np
import pywt
import pywt.data


class AutoDwtFilter(object):
    def __init__(self):
        self.T_noise_only = 0.01 ** 2
        self.T_r = 0.2 ** 2
        self.wavelets = 'db1'

    def __call__(self, image):
        # calc decomposition level
        S_j = self.calc_S_j(image)
        k = np.where(np.all(S_j < self.T_r, 1))[0][-1] + 1
        # calc reference S_r_L & S_r_L (required for threshold)
        S_r_L, S_r_H = self.calc_S_r(image, k)
        # do decomposition
        coeffs2 = pywt.wavedec2(image, self.wavelets, level=k)
        lambda_j_L, lambda_j_H = self.calc_thresholds(coeffs2, k, S_r_L, S_r_H, S_j)
        coeffs2_filt = self.filter_dwt(coeffs2, k, lambda_j_L, lambda_j_H)
        filtered_image = pywt.waverec2(coeffs2_filt, self.wavelets)
        return filtered_image, coeffs2_filt, k

    def calc_S_j(self, image):
        # calc subjective level S_j eq.5
        max_level = np.floor(np.log2(image.shape[0] - 1)).astype('int')
        S_j = np.zeros((max_level, 3))
        for level in range(1, max_level + 1):
            coeffs2 = pywt.wavedec2(image, self.wavelets, level=level)
            for i_component in range(3):
                curr_w_j = coeffs2[1][i_component]
                nominator = np.max(np.abs(curr_w_j))
                denominator = np.sum(np.abs(curr_w_j))
                S_j[level - 1][i_component] = nominator / denominator
        return S_j

    def calc_S_r(self, image, k):
        # prep work between eq 13 tp 14 calc S_r_L S_r_H
        S_r_L = np.zeros((3))
        S_r_H = np.zeros((3))
        coeffs2 = pywt.wavedec2(image, self.wavelets, level=k + 1)
        for i_component in range(3):
            w_k = coeffs2[k][i_component]
            w_kp1 = coeffs2[k + 1][i_component]
            # pos and neg coeffs
            w_k_L = w_k[w_k < 0]
            w_k_H = w_k[w_k >= 0]
            # pos and neg coeffs
            w_kp1_L = w_kp1[w_kp1 < 0]
            w_kp1_H = w_kp1[w_kp1 >= 0]
            # eq. 14
            S_k_L = np.max(np.abs(w_k_L)) / np.sum(np.abs(w_k_L))
            S_kp1_L = np.max(np.abs(w_kp1_L)) / np.sum(np.abs(w_kp1_L))
            # eq. 15
            S_k_H = np.max(np.abs(w_k_H)) / np.sum(np.abs(w_k_H))
            S_kp1_H = np.max(np.abs(w_kp1_H)) / np.sum(np.abs(w_kp1_H))
            S_r_L[i_component] = (S_k_L + S_kp1_L) / 2
            S_r_H[i_component] = (S_k_H + S_kp1_H) / 2
        return S_r_L, S_r_H

    def calc_thresholds(self, coeffs2, k, S_r_L, S_r_H, S_j):

        lambda_j_L = np.zeros((k, 3))
        lambda_j_H = np.zeros((k, 3))

        for level in range(1, k + 1):
            for i_component in range(3):
                # get curr coeffs
                curr_w_j = coeffs2[level][i_component]
                # pos and neg coeffs
                curr_w_j_L = curr_w_j[curr_w_j < 0]
                curr_w_j_H = curr_w_j[curr_w_j >= 0]
                # eq. 8
                mu_j = np.mean(curr_w_j)
                # eq. 9
                sigma_j = np.std(curr_w_j)
                # eq. 10
                kappa_j_L_min = (mu_j - np.max(np.abs(curr_w_j_L))) / sigma_j
                # eq. 11
                kappa_j_H_min = (np.max(np.abs(curr_w_j_H)) - mu_j) / sigma_j
                # eq. 14
                S_j_L = np.max(np.abs(curr_w_j_L)) / np.sum(np.abs(curr_w_j_L))
                # eq. 15
                S_j_H = np.max(np.abs(curr_w_j_H)) / np.sum(np.abs(curr_w_j_H))
                # eq. 12
                kappa_j_L = (S_r_L[i_component] - S_j_L) / S_r_L[i_component] * kappa_j_L_min
                # eq. 13
                kappa_j_H = (S_r_H[i_component] - S_j_H) / S_r_H[i_component] * kappa_j_H_min
                # eq. 6
                if S_j[level - 1, i_component] < self.T_noise_only:
                    lambda_j_L[level - 1, i_component] = mu_j - kappa_j_L_min* sigma_j
                else:
                    lambda_j_L[level - 1, i_component] = mu_j - kappa_j_L* sigma_j

                # eq. 7
                if S_j[level - 1, i_component] < self.T_noise_only:
                    lambda_j_H[level - 1, i_component] = mu_j + kappa_j_H_min * sigma_j
                else:
                    lambda_j_H[level - 1, i_component] = mu_j + kappa_j_H * sigma_j

        return lambda_j_L, lambda_j_H

    def filter_dwt(self, coeffs2, k, lambda_j_L, lambda_j_H):
        # do filtering
        coeffs2_filt = coeffs2
        for level in range(1, k + 1):
            filtered_w_j_i = []
            for i_component in range(3):
                curr_w_j_i = coeffs2_filt[level][i_component]
                curr_w_j_i[np.logical_and(curr_w_j_i <= lambda_j_H[level - 1, i_component],
                                          curr_w_j_i >= lambda_j_L[level - 1, i_component])] = 0
                filtered_w_j_i.append(curr_w_j_i)
            coeffs2_filt[level] = tuple(filtered_w_j_i)
        return coeffs2_filt
