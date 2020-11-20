import math
import numpy as np
from PIL import Image

def compute_h(p1, p2):
    # TODO ...
    
    # number of pairs of corresponding interest points 
    N = p1.shape[0]

    # construct A
    A = [-p2[0][1], -p2[0][0], -1, 0, 0, 0, p1[0][1]*p2[0][1], p1[0][1]*p2[0][0], p1[0][1]]
    for i in range(N):
        arr1 = [-p2[i][1], -p2[i][0], -1, 0, 0, 0, p1[i][1]*p2[i][1], p1[i][1]*p2[i][0], p1[i][1]]
        arr2 = [0, 0, 0, -p2[i][1], -p2[i][0], -1, p1[i][0]*p2[i][1], p1[i][0]*p2[i][0], p1[i][0]]
        if i != 0:
            A = np.vstack((A, arr1))
        A = np.vstack((A, arr2))
    
    # apply SVD
    U, S, Vh = np.linalg.svd(A)
    V = Vh.T
    
    # find smallest eigenvalue and its corresponding eigenvector, which should be H
    minev = V[:, np.argmin(S)]
    H = np.reshape(minev, (3, 3))
    return H

def compute_h_norm(p1, p2):
    # TODO ...
    # number of pairs of corresponding interest points 
    N = p1.shape[0]

    # construct Normalization matrix T1, T2
    p2_mean = np.mean(p2, axis=0)
    p1_mean = np.mean(p1, axis=0)
    sump2 = 0
    sump1 = 0
    for i in range(N):
        sump2 += ((p2[i][0]-p2_mean[0])**2 + (p2[i][1]-p2_mean[1])**2)**0.5
        sump1 += ((p1[i][0]-p1_mean[0])**2 + (p1[i][1]-p1_mean[1])**2)**0.5
    s2 = (math.sqrt(2)*N)/sump2
    s1 = (math.sqrt(2)*N)/sump1
    T2 = s2 * np.array([[1, 0, -p2_mean[1]], [0, 1, -p2_mean[0]], [0, 0, 1/s2]])
    T1 = s1 * np.array([[1, 0, -p1_mean[1]], [0, 1, -p1_mean[0]], [0, 0, 1/s1]])

    # normalize p1, p2 using T1, T2
    p2_norm = np.zeros(p2.shape)
    p1_norm = np.zeros(p1.shape)
    for i in range(N):
        coord2 = np.array([p2[i][1], p2[i][0], 1])
        result2 = np.matmul(T2, coord2.T)
        p2_norm[i][0] = result2[1]
        p2_norm[i][1] = result2[0]

        coord1 = np.array([p1[i][1], p1[i][0], 1])
        result1 = np.matmul(T1, coord1.T)
        p1_norm[i][0] = result1[1]
        p1_norm[i][1] = result1[0]

    # construct A
    A = [-p2_norm[0][1], -p2_norm[0][0], -1, 0, 0, 0, p1_norm[0][1]*p2_norm[0][1], p1_norm[0][1]*p2_norm[0][0], p1_norm[0][1]]
    for i in range(N):
        arr1 = [-p2_norm[i][1], -p2_norm[i][0], -1, 0, 0, 0, p1_norm[i][1]*p2_norm[i][1], p1_norm[i][1]*p2_norm[i][0], p1_norm[i][1]]
        arr2 = [0, 0, 0, -p2_norm[i][1], -p2_norm[i][0], -1, p1_norm[i][0]*p2_norm[i][1], p1_norm[i][0]*p2_norm[i][0], p1_norm[i][0]]
        if i != 0:
            A = np.vstack((A, arr1))
        A = np.vstack((A, arr2))
    
    # apply SVD
    U, S, Vh = np.linalg.svd(A)
    V = Vh.T
    minev = V[:, np.argmin(S)]
    H_bar = np.reshape(minev, (3, 3))

    # undo normalzation
    T1_inv = np.linalg.inv(T1)
    mid = np.matmul(T1_inv, H_bar)
    H = np.matmul(mid, T2)

    return H

def warp_image(igs_in, igs_ref, H):
    # TODO ...
    H_inv = np.linalg.inv(H)
    new = np.zeros((1680, 2240, 3), dtype=np.uint8)
    new_pd = np.pad(igs_ref, ((350, 262), (1200, 0), (0, 0)), mode='constant', constant_values=0)
    for i in range(-350, 1330):
        for j in range(-1200, 1040):
            coord = np.array([j, i, 1])
            result = np.matmul(H_inv, coord.T)
            alpha = 1 / result[2]
            results = np.array([result[1]*alpha, result[0]*alpha])

            # bilinear interpolation
            if 0 <= results[0] < igs_in.shape[0] and 0 <= results[1] < igs_in.shape[1]:
                m = np.floor(results[0]).astype(int)
                n = np.floor(results[1]).astype(int)
                a = results[0] - np.floor(results[0])
                b = results[1] - np.floor(results[1])
                if m != igs_in.shape[0]-1 and n != igs_in.shape[1] -1:
                    pixel_ij = igs_in[m][n]
                    pixel_i_1_j = igs_in[m + 1][n]
                    pixel_i_j_1 = igs_in[m][n + 1]
                    pixel_i_1_j_1 = igs_in[m + 1][n + 1]

                    new_pixel = (1-a)*(1-b)*pixel_ij + a*(1-b)*pixel_i_1_j + a*b*pixel_i_1_j_1 + (1-a)*b*pixel_i_j_1
                    new[i + 350][j + 1200] = new_pixel
                    new_pd[i + 350][j + 1200] = new_pixel
        print(i)

    igs_warp = new
    igs_merge = new_pd

    return igs_warp, igs_merge

def rectify(igs, p1, p2):
    # TODO ...
    H = compute_h_norm(p2, p1)    
    H_inv = np.linalg.inv(H)
    igs_rec = np.zeros(igs.shape, dtype=np.uint8)
    for i in range(igs.shape[0]):
        for j in range(igs.shape[1]):
            coord = np.array([j, i, 1])
            result = np.matmul(H_inv, coord.T)
            alpha = 1 / result[2]
            results = np.array([result[1]*alpha, result[0]*alpha])

            # bilinear interpolation
            if 0 <= results[0] < igs.shape[0] and 0 <= results[1] < igs.shape[1]:
                m = np.floor(results[0]).astype(int)
                n = np.floor(results[1]).astype(int)
                a = results[0] - np.floor(results[0])
                b = results[1] - np.floor(results[1])
                if m != igs.shape[0]-1 and n != igs.shape[1] -1:
                    pixel_ij = igs[m][n]
                    pixel_i_1_j = igs[m + 1][n]
                    pixel_i_j_1 = igs[m][n + 1]
                    pixel_i_1_j_1 = igs[m + 1][n + 1]

                    new_pixel = (1-a)*(1-b)*pixel_ij + a*(1-b)*pixel_i_1_j + a*b*pixel_i_1_j_1 + (1-a)*b*pixel_i_j_1
                    igs_rec[i][j] = new_pixel
        print(i)

    return igs_rec

def set_cor_mosaic():
    # TODO ...
    p_in = np.array([[826, 1343], [600, 1334], [754, 1302], [691, 1067], [659, 959], [591, 799], [591, 789], [726, 798], [461, 691], [971, 1087], [1002, 1374], [919, 1394], [1000, 1511], [944, 1398], [951, 1324], [990, 1362]])
    p_ref = np.array([[816, 744], [604, 759], [750, 715], [693, 500], [660, 388], [585, 214], [584, 201], [735, 197], [434, 92], [979, 492], [982, 759], [901, 782], [969, 873], [924, 783], [937, 719], [971, 749]])

    return p_in, p_ref

def set_cor_rec():
    # TODO ...
    c_in = np.array([[25, 151], [35, 263], [11, 167], [21, 252], [263, 167], [251, 252], [248, 151], [236, 263]])
    c_ref = np.array([[25, 151], [25, 263], [11, 167], [11, 252], [263, 167], [263, 252], [248, 151], [248, 263]])
    return c_in, c_ref

def main():
    ##############
    # step 1: mosaicing
    ##############

    # read images
    img_in = Image.open('data/porto1.png').convert('RGB')
    img_ref = Image.open('data/porto2.png').convert('RGB')

    # shape of igs_in, igs_ref: [y, x, 3]
    igs_in = np.array(img_in)
    igs_ref = np.array(img_ref)

    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    p_in, p_ref = set_cor_mosaic()

    # p_ref = H * p_in
    H = compute_h_norm(p_ref, p_in)
    igs_warp, igs_merge = warp_image(igs_in, igs_ref, H)

    # plot images
    img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    img_merge = Image.fromarray(igs_merge.astype(np.uint8))

    # save images
    img_warp.save('porto1_warped.png')
    img_merge.save('porto_merged.png')

    ##############
    # step 2: rectification
    ##############

    img_rec = Image.open('data/iphone.png').convert('RGB')
    igs_rec = np.array(img_rec)

    c_in, c_ref = set_cor_rec()

    igs_rec = rectify(igs_rec, c_in, c_ref)

    img_rec_output = Image.fromarray(igs_rec.astype(np.uint8))
    img_rec_output.save('iphone_rectified.png')

if __name__ == '__main__':
    main()
