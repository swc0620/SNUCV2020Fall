from PIL import Image
import numpy as np

def load_image(filepath):
    '''
    Load image from filepath and return ndarray.
    '''
    image = Image.open(filepath)
    return np.asarray(image)

def save_image(ndarray, filepath):
    '''
    Save ndarray as image file.
    '''
    image = Image.fromarray(ndarray)
    image.save(filepath)

def show_image(ndarray):
    '''
    Show ndarray as image.
    '''
    image = Image.fromarray(ndarray)
    image.show()

def downsample(ndarray):
    '''
    Return downsampled image as ndarray. (anti-aliased)
    '''
    h, w, _ = ndarray.shape
    image = Image.fromarray(ndarray)
    image = image.resize((round(w/2), round(h/2)), Image.ANTIALIAS)
    return np.asarray(image)

def upsample(ndarray):
    '''
    Return upsampled image as ndarray. (anti-aliased)
    '''
    h, w, _ = ndarray.shape
    image = Image.fromarray(ndarray)
    image = image.resize((w*2, h*2), Image.ANTIALIAS)
    return np.asarray(image)

def concat(left_ndarray, right_ndarray):
    '''
    Return concatenation of left half of first image and right half of second
    image. The arrays should be the same size.
    '''
    h, w, _ = left_ndarray.shape
    return np.hstack((left_ndarray[:, :w//2], right_ndarray[:, w//2:]))

