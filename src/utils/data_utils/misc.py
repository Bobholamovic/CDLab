import torch
import numpy as np
import cv2
from scipy.io import loadmat
from skimage.io import imread
from imageio import mimsave


def default_loader(path_):
    return imread(path_)


def mat_loader(path_):
    return loadmat(path_)


def save_gif(uri, img_seq):
    mimsave(uri, img_seq)
    

def to_tensor(arr):
    if any(s < 0 for s in arr.strides):
        arr = np.ascontiguousarray(arr)
    if arr.ndim < 3:
        return torch.from_numpy(arr)
    elif arr.ndim == 3:
        return torch.from_numpy(np.transpose(arr, (2,0,1)))
    else:
        raise ValueError


def to_array(tensor):
    if tensor.ndim <= 4:
        arr = tensor.data.cpu().numpy()
        if tensor.ndim in (3, 4):
            arr = np.moveaxis(arr, -3, -1)
        return arr
    else:
        raise ValueError


def normalize_minmax(x):
    EPS = 1e-32
    return (x-x.min()) / (x.max()-x.min()+EPS)


def normalize_8bit(x):
    return x/255.0


def to_pseudo_color(gray, color_map=cv2.COLORMAP_JET):
    # Reverse channels to convert BGR to RGB
    return cv2.applyColorMap(gray, color_map)[...,::-1]


def quantize_8bit(x):
    # [0.0,1.0] float => [0,255] uint8
    # or [0,1] int => [0,255] uint8
    return (x*255).astype('uint8')