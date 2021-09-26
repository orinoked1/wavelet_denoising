

import numpy as np

def noisy(image,params):
    # Parameters
    # ----------
    # image : ndarray
    #     Input image data. Will be converted to float.
    # mode : str
    #     One of the following strings, selecting the type of noise to add:
    #
    #     'gauss'     Gaussian-distributed additive noise.
    #     'poisson'   Poisson-distributed noise generated from the data.
    #     's&p'       Replaces random pixels with 0 or 1.
    #     'speckle'   Multiplicative noise using out = image + n*image,where
    #                 n is uniform noise with specified mean & variance.
    noise_typ = params['type']
    if noise_typ == "gauss":
      row,col= image.shape
      mean = params['mean']
      sigma = params['sigma']

      gauss = np.random.normal(mean,sigma,(row,col))
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      noisy = noisy.round()
      noisy[noisy<0]=0
      noisy[noisy >255] = 255
      return noisy
    elif noise_typ == "s&p":
      s_vs_p = params['s_vs_p']
      amount = params['amount']
      out = np.copy(image)
      pixel_idxs = np.random.choice(np.arange(image.size-1), int(amount * image.size), replace=False)
      # Salt mode
      coords = pixel_idxs[:int(len(pixel_idxs)*s_vs_p)]
      out[np.unravel_index(coords, image.shape)] = 255
      # Pepper mode
      coords = pixel_idxs[int(len(pixel_idxs) * s_vs_p):]
      out[np.unravel_index(coords, image.shape)] = 0
      return out
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      noisy = noisy.round()
      noisy[noisy<0]=0
      noisy[noisy >255] = 255
      return noisy

    elif noise_typ =="speckle":

      row,col = image.shape
      gauss = np.random.randn(row,col)
      gauss = gauss.reshape(row,col)
      noisy = image + image * gauss
      return noisy