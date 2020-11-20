from utils import *
import math

def get_pixel_at(pixel_grid, i, j):
    '''
    Get pixel values at row i and column j.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        i (int): Row index.
        j (int): Column index.

    Returns:
        ndarray: 1D numpy array representing RGB values.
    '''
    org_img_height, org_img_width, _ = pixel_grid.shape
    if 0 <= i < org_img_height and 0 <= j < org_img_width:
        return pixel_grid[i, j]
    else:
        return np.array([0, 0, 0])

def get_patch_at(pixel_grid, i, j, size):
    '''
    Get an image patch at row i and column j.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        i (int): Row index.
        j (int): Column index.
        size (int): Patch size.

    Returns:
        ndarray: 3D numpy array representing an image patch.
    '''
    half = (size-1)//2
    patch = []
    for m in range(-half, half+1):
        row = []
        for n in range(-half, half+1):
            row.append(get_pixel_at(pixel_grid, i+m, j+n))
        row = np.array(row)
        patch.append(row)
    patch = np.array(patch)

    return patch

def apply_gaussian_filter(pixel_grid, size):
    '''
    Apply gaussian filter for every pixel in pixel_grid, and return the
    resulting pixel grid.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        size (int): Kernel size.

    Returns:
        ndarray: 3D numpy array representing an RGB image after filtering.
    '''
    org_img_height, org_img_width, _ = pixel_grid.shape

    half = (size-1)//2
    sigma = 5
    mu = (size-1)//2

    gk_1d = np.fromfunction(lambda x: (np.pi*2*sigma**2)**(-0.5) * np.exp(-0.5*((x-mu)/sigma)**2), (size,))
    gk_1d_rgb = np.repeat(gk_1d[:, np.newaxis], 3, axis=1)
    
    pad = max(org_img_height, org_img_width)//2
    pad_pg = np.pad(pixel_grid, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    output1 = np.zeros(pixel_grid.shape)
    for i in range(org_img_height):
        for j in range(org_img_width):
            patch = pad_pg[i+pad, j+pad-half:j+pad+half+1]
            output1[i][j] = np.sum(gk_1d_rgb * patch, axis=0)
        print(i)
    output1 = output1.astype(np.uint8)

    output1 = np.pad(output1, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    output2 = np.zeros(pixel_grid.shape)
    for i in range(org_img_height):
        for j in range(org_img_width):
            patch = output1[i+pad-half:i+pad+half+1, j+pad]
            output2[i][j] = np.sum(gk_1d_rgb * patch, axis=0)
    output2 = output2.astype(np.uint8)

    return output2

def apply_median_filter(pixel_grid, size):
    '''
    Apply median filter for every pixel in pixel_grid, and return the
    resulting pixel grid.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        size (int): Kernel size.

    Returns:
        ndarray: 3D numpy array representing an RGB image after filtering.
    '''    
    org_img_height, org_img_width, _ = pixel_grid.shape

    half = (size-1)//2

    pad = max(org_img_height, org_img_width)//2
    pad_pg = np.pad(pixel_grid, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    output = np.zeros(pixel_grid.shape)
    for i in range(org_img_height):
        for j in range(org_img_width):
            patch = pad_pg[i+pad-half:i+pad-half+1, j+pad-half:j+pad+half+1]

            unfolded_patch = patch.transpose(2,0,1).reshape(3,-1)
            output[i][j] = np.median(unfolded_patch, axis=1)
        print(i)

    output = output.astype(np.uint8)

    return output


def build_gaussian_pyramid(pixel_grid, size, levels=5):
    '''
    Build and return a Gaussian pyramid.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        size (int): Kernel size.
        levels (int): Number of levels.

    Returns:
        list of ndarray: List of 3D numpy arrays representing Gaussian
        pyramid.
    '''
    gaussian_pyramid = []
    after_downsample = pixel_grid
    gaussian_pyramid.append(after_downsample)
    for i in range(levels-1):
        before_downsample = apply_gaussian_filter(after_downsample, size)
        after_downsample = downsample(before_downsample)
        gaussian_pyramid.append(after_downsample)

    return gaussian_pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    '''
    Build and return a Laplacian pyramid.

    Args:
        gaussian_pyramid (list of ndarray): Gaussian pyramid. 

    Returns:
        list of ndarray: List of 3D numpy arrays representing Laplacian
        pyramid
    '''
    laplacian_pyramid = []
    laplacian_pyramid.append(gaussian_pyramid[-1].astype(np.float32))
    for i in range(len(gaussian_pyramid)-2, -1, -1):
        after_upsample = upsample(gaussian_pyramid[i+1])
        difference = gaussian_pyramid[i].astype(np.float32) - after_upsample.astype(np.float32)
        laplacian_pyramid.append(difference)
    
    return laplacian_pyramid

def blend_images(left_image, right_image):
    '''
    Smoothly blend two images by concatenation.
    
    Tip: This function should build Laplacian pyramids for both images,
    concatenate left half of left_image and right half of right_image
    on all levels, then start reconstructing from the smallest one.

    Args:
        left_image (ndarray): 3D numpy array representing an RGB image.
        right_image (ndarray): 3D numpy array representing an RGB image.

    Returns:
        ndarray: 3D numpy array representing an RGB image after blending.
    '''
    left_gp= build_gaussian_pyramid(left_image, 31)
    right_gp = build_gaussian_pyramid(right_image, 31)
    left_lp = build_laplacian_pyramid(left_gp)
    right_lp = build_laplacian_pyramid(right_gp)

    concat_lp = []
    for i in range(len(left_lp)):
        concat_lp.append(concat(left_lp[i], right_lp[i]))

    concatenated_img_before_upsample = concat_lp[0].astype(np.uint8)
    for i in range(1, len(left_lp)):
        concatenated_img_after_upsample = upsample(concatenated_img_before_upsample)
        concatenated_img_before_upsample = concat_lp[i] + concatenated_img_after_upsample.astype(np.float32)
        concatenated_img_before_upsample = np.clip(concatenated_img_before_upsample, 0, 255)
        concatenated_img_before_upsample = concatenated_img_before_upsample.astype(np.uint8)

    return concatenated_img_before_upsample

if __name__ == "__main__":
    ### Test Gaussian Filter ###
    dog_gaussian_noise = load_image('./images/dog_gaussian_noise.png')
    after_filter = apply_gaussian_filter(dog_gaussian_noise, 31)
    save_image(after_filter, './dog_gaussian_noise_after.png')
    
    ### Test Median Filter ###
    # dog_salt_and_pepper = load_image('./images/dog_salt_and_pepper.png')
    # after_filter = apply_median_filter(dog_salt_and_pepper, 7)
    # save_image(after_filter, './dog_salt_and_pepper_after.png')

    ### Test Image Blending ###
    # player1 = load_image('./images/player1.png')
    # player2 = load_image('./images/player2.png')
    # after_blending = blend_images(player1, player2)
    # save_image(after_blending, './player3.png')

    # Simple concatenation for comparison.
    # save_image(concat(player1, player2), './player_simple_concat.png')


