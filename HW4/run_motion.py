import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold

def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and
    # RectBivariateSpline. Never use OpenCV.
    # dp = 0 # you should delete this

    dp = 0

    img1_h, img1_w = img1.shape
    img2_h, img2_w = img2.shape

    # T(x)
    T_x = img1

    # I(W(x; p))
    # warp image
    img1_x1 = np.array([0, 0, 1])
    img1_x2 = np.array([img1_w, img1_h, 1])
    W = np.array([[1+p[0], 0+p[1], 0+p[2]], [0+p[3], 1+p[4], 0+p[5]]])
    warped_x1 = W @ img1_x1.T
    warped_x2 = W @ img1_x2.T
    # create meshgrid of warped image
    warped_x_ls = np.linspace(warped_x1[0], warped_x2[0], img1_w)
    warped_y_ls = np.linspace(warped_x1[1], warped_x2[1], img1_h)
    warped_x_mg, warped_y_mg = np.meshgrid(warped_x_ls, warped_y_ls)
    # RectBivariateSpline_interpolation of img2
    img2_x_ls = np.linspace(0, img2_w-1, img2_w)
    img2_y_ls = np.linspace(0, img2_h-1, img2_h)
    img2_x_mg, img2_y_mg = np.meshgrid(img2_x_ls, img2_y_ls)
    img2_rbsi = RectBivariateSpline(img2_y_ls, img2_x_ls, img2)
    # I(W(x; p))
    I_W_x = img2_rbsi.ev(warped_y_mg, warped_x_mg)

    # T(x) - I(W(x; p))
    T_I = (T_x - I_W_x).reshape(-1, 1)

    # scaling gradient
    Gx_max = Gx.max()
    Gy_max = Gy.max()
    G_max = max(Gx_max, Gy_max)

    # Gradient_I
    Gx_f = Gx.reshape(-1, 1)
    Gy_f = Gy.reshape(-1, 1)
    gradient_I = np.hstack((Gx_f, Gy_f))
    gradient_I /= G_max

    # GJ : Gradient_I * Jacobian matrix
    GJ = np.zeros((img1_w*img1_h, 6))
    for y in range(img1_h):
        for x in range(img1_w):
            jacobian = np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]])
            temp = gradient_I[img1_w*y+x] @ jacobian
            GJ[img1_w*y+x] = temp
        # print(img1_w*y)

    GJ = GJ * G_max

    # H
    H = GJ.T @ GJ

    # dp
    dp = np.linalg.inv(H) @ GJ.T @ (T_I)
    dp = dp.flatten()

    ### END CODE HERE ###
    return dp

def subtract_dominant_motion(img1, img2):
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize = 5)
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize = 5)

    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and
    # RectBivariateSpline. Never use OpenCV.
    # moving_image = np.abs(img2 - img1) # you should delete this

    epsilon = 0.0171

    # initialize p
    dp = np.array([1, 0, 0, 0, 1, 0])
    p = np.zeros(6)    

    while np.linalg.norm(dp) > epsilon:
        dp = lucas_kanade_affine(img1, img2, p, Gx, Gy)
        p += dp

    # M
    M = np.array([[1+p[0], 0+p[1], 0+p[2]], [0+p[3], 1+p[4], 0+p[5]]])
    
    img1_h, img1_w = img1.shape
    img2_h, img2_w = img2.shape

    # T(x)
    T_x = img1

    # I(W(x; p))
    # warp image
    img1_x1 = np.array([0, 0, 1])
    img1_x2 = np.array([img1_w, img1_h, 1])
    warped_x1 = M @ img1_x1.T
    warped_x2 = M @ img1_x2.T
    # create meshgrid of warped image
    warped_x_ls = np.linspace(warped_x1[0], warped_x2[0], img1_w)
    warped_y_ls = np.linspace(warped_x1[1], warped_x2[1], img1_h)
    warped_x_mg, warped_y_mg = np.meshgrid(warped_x_ls, warped_y_ls)
    # RectBivariateSpline_interpolation of img2
    img2_x_ls = np.linspace(0, img2_w-1, img2_w)
    img2_y_ls = np.linspace(0, img2_h-1, img2_h)
    img2_x_mg, img2_y_mg = np.meshgrid(img2_x_ls, img2_y_ls)
    img2_rbsi = RectBivariateSpline(img2_y_ls, img2_x_ls, img2)
    # I(W(x; p))
    I_W_x = img2_rbsi.ev(warped_y_mg, warped_x_mg)
    # T(x) - I(W(x; p))
    T_I = T_x - I_W_x

    moving_image = np.abs(T_I)

    th_hi = 0.2 * 256 # you can modify this
    th_lo = 0.15 * 256 # you can modify this

    ### END CODE HERE ###

    hyst = apply_hysteresis_threshold(moving_image, th_lo, th_hi)
    return hyst

data_dir = 'data'
video_path = 'motion.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, 150/20, (320, 240))
tmp_path = os.path.join(data_dir, "{}.jpg".format(0))
T = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)
for i in range(1, 150):
    img_path = os.path.join(data_dir, "{}.jpg".format(i))
    I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    clone = I.copy()
    moving_img = subtract_dominant_motion(T, I)
    clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
    clone[moving_img, 2] = 255
    out.write(clone)
    T = I
out.release()

