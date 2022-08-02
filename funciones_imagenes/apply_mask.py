from skimage import measure
from scipy import ndimage
import funciones_imagenes.image_funct as im
import cv2
import numpy as np

def quitar_trozos(mask):
    mask = measure.label(mask)
    # regions = regionprops(all_labels)
    ntotal = {k: (k==mask).sum() for k in np.unique(mask) if k >0}
    print(ntotal)
    k = list(ntotal.keys())[np.argmax(list(ntotal.values()))]
    print(k)
    mask = k==mask
    mask = ndimage.binary_fill_holes(mask)
    return mask


def apply_mask(img, model):
    pix = img.shape[1]
    img_2 = im.normalize(im.recolor_resize(img, 256))
    mask = model.predict(img_2[np.newaxis,...])[0,...]
    mask = quitar_trozos(mask > 0.5)
    mask = cv2.resize(mask, (pix, pix))
    return img*mask


def des_normalize(img):
    return cv2.normalize(img, None, alpha = 0, beta = 255,
                         norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)