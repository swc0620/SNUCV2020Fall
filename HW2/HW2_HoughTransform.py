
import math
import glob
import numpy as np
from PIL import Image
import cv2


# parameters

datadir = './data'
resultdir='./results'

sigma=2
threshold=0.03
rhoRes=1
thetaRes=math.pi/180
nLines=20


def ConvFilter(Igs, G):
    print('1. ConvFilter')
    # TODO ...
    Igs_h, Igs_w = Igs.shape            
    G_h, G_w = G.shape

    half = max(G_h, G_w) // 2
    t_half = half * 2

    Igs_pad = np.pad(Igs, ((half, half), (half, half)), mode='edge')          

    Iconv = np.zeros(Igs.shape)
    for i in range(Igs_h):
        for j in range(Igs_w):
            patch = Igs_pad[i:i+t_half+1, j:j+t_half+1]
            
            Iconv[i][j] = np.einsum('ij,ij', np.flip(G), patch)
    return Iconv


def EdgeDetection(Igs, sigma):
    # TODO ...
    print('2. Edge Detection')
    size = sigma * 6 + 1
    G = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
    Ig = ConvFilter(Igs, G)

    x_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ix = ConvFilter(Ig, x_sobel)
    
    y_sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Iy = ConvFilter(Ig, y_sobel)

    Im = np.hypot(Ix, Iy)
    Io = np.arctan2(Iy, Ix)

    # Im = Im / Im.max() * 255
    # image = Image.fromarray(Im)
    # image.show()

    # non maximal suppression
    Im_nms = np.zeros(Im.shape)
    Im_h, Im_w = Im.shape
    for i in range(1, Im_h-1):
        for j in range(1, Im_w-1):
            if (0 <= Io[i, j] < (np.pi / 4) or -np.pi <= Io[i, j] < -(np.pi * 3 / 4)):
                a = np.abs(np.tan(Io[i, j]))
                p = a * Im[i-1, j+1] + (1-a) * Im[i, j+1]
                r = a * Im[i+1, j-1] + (1-a) * Im[i, j-1]
            elif ((np.pi / 4) <= Io[i, j] < (np.pi / 2) or -(np.pi * 3 / 4) <= Io[i, j] < -(np.pi / 2)):
                a = np.abs(1/np.tan(Io[i, j]))
                p = a * Im[i-1, j+1] + (1-a) * Im[i-1, j]
                r = a * Im[i+1, j-1] + (1-a) * Im[i+1, j]
            elif ((np.pi / 2) <= Io[i, j] < (np.pi * 3 / 4) or -(np.pi / 2) <= Io[i, j] < -(np.pi / 4)):
                a = np.abs(1/np.tan(Io[i, j]))
                p = a * Im[i-1, j-1] + (1-a) * Im[i-1, j]
                r = a * Im[i+1, j+1] + (1-a) * Im[i+1, j]
            elif ((np.pi * 3 / 4) <= Io[i, j] <= np.pi or -(np.pi / 4) <= Io[i, j] < 0):
                a = np.abs(np.tan(Io[i, j]))
                p = a * Im[i-1, j-1] + (1-a) * Im[i, j-1]
                r = a * Im[i+1, j+1] + (1-a) * Im[i, j+1]
            
            if Im[i, j] > p and Im[i, j] > r:
                Im_nms[i, j] = Im[i, j]
            else:
                Im_nms[i, j] = 0

    # Im_nms = Im_nms / Im_nms.max() * 255
    # image = Image.fromarray(Im_nms)
    # image.show()

    return Im_nms, Io, Ix, Iy

def HoughTransform(Im,threshold, rhoRes, thetaRes):
    # TODO ...
    print('3. Hough Transform')
    # threshold
    Im[Im < threshold] = 0

    # Im = Im / Im.max() * 255
    # image = Image.fromarray(Im)
    # image.show()

    # # Hough transform
    theta = np.linspace(-90, 90, num=int(np.pi/thetaRes)+1)
    theta = np.deg2rad(theta)
    Im_h, Im_w = Im.shape
    diagonal = int(np.ceil(np.hypot(Im_h, Im_w)))

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    H = np.zeros((int(2*diagonal // rhoRes), len(theta)-1))
    for x in range(Im_w):
        for y in range(Im_h):
            if Im[y, x] > 0:
                for k in range(len(theta)-1):
                    rho = int(round(x * cos_theta[k] + y * sin_theta[k])) + diagonal
                    rho = int(rho // rhoRes)
                    H[rho, k] += 1

    # image = Image.fromarray(H)
    # image.show()

    return H

def HoughLines(H,rhoRes,thetaRes,nLines):
    # TODO ...
    print('4. Hough Lines')
    # http://fourier.eng.hmc.edu/e161/dipum/houghpeaks.m
    # https://gist.github.com/ri-sh/45cb32dd5c1485e273ab81468e531f09

    # nhood_size = H
    nhood_size_y, nhood_size_x = H.shape
    nhood_size_y = int(nhood_size_y // 10 + 1)
    nhood_size_x = int(nhood_size_x // 10 + 1)

    # loop through number of peaks to identify
    indicies = []
    lRho = []
    lTheta = []
    H1 = np.copy(H)
    for i in range(nLines):
        idx = np.argmax(H1) # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape) # remap to shape of H
        indicies.append(H1_idx)

        # surpess indicies in neighborhood
        idx_y, idx_x = H1_idx # first separate x, y indexes from argmax(H)
        lRho.append(idx_y)
        lTheta.append(idx_x)
        # if idx_x is too close to the edges choose appropriate values
        if idx_x - (nhood_size_x // 2) < 0: 
            min_x = 0
        else: 
            min_x = idx_x - (nhood_size_x // 2)
        if (idx_x + (nhood_size_x // 2) + 1) > H.shape[1]: 
            max_x = H.shape[1]
        else: 
            max_x = idx_x + (nhood_size_x // 2) + 1

        # if idx_y is too close to the edges choose appropriate values
        if idx_y - (nhood_size_y / 2) < 0: 
            min_y = 0
        else: 
            min_y = idx_y - (nhood_size_y // 2)
        if (idx_y + (nhood_size_y // 2) + 1) > H.shape[0]: 
            max_y = H.shape[0]
        else: 
            max_y = idx_y + (nhood_size_y // 2) + 1

        # bound each index by the neighborhood size and set all values to 0
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                # remove neighborhoods in H1
                H1[y, x] = 0

                # highlight peaks in original H
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255

    lRho = np.array(lRho)
    lTheta = np.array(lTheta)
    lTheta = lTheta / ((math.pi/thetaRes)/180)
    lTheta -= 90

    # image = Image.fromarray(H)
    # image.show()

    return lRho, lTheta

def HoughLineSegments(lRho, lTheta, Im, threshold):
    # TODO ...
    print('5. Hough Line Segments')
    # threshold
    Im[Im < threshold] = 0

    sigma1 = 2
    size1 = 13
    G = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma1**2)) * np.exp((-1*((x-(size1-1)/2)**2+(y-(size1-1)/2)**2))/(2*sigma1**2)), (size1, size1))
    Im = ConvFilter(Im, G)

    Im = Im / Im.max() * 255
    # image = Image.fromarray(Im)
    # image.show()

    Im_h, Im_w = Im.shape
    l = []
    for k in range(nLines):
        new = np.zeros(Im.shape)
        diagonal = int(np.ceil(np.hypot(Im_h, Im_w)))
        cos_theta = np.cos(np.deg2rad(lTheta))
        sin_theta = np.sin(np.deg2rad(lTheta))

        if lTheta[k] != 0:
            x1 = 0
            y1 = int(round(((lRho[k]*rhoRes) - diagonal - x1 * cos_theta[k]) / sin_theta[k]))
            start_x = 0
            start_y = y1
            x2 = Im_w - 1
            y2 = int(round(((lRho[k]*rhoRes) - diagonal - x2 * cos_theta[k]) / sin_theta[k]))
            end_x = Im_w - 1
            end_y = y2
            cv2.line(new, (start_x, start_y), (end_x, end_y), (255, 255, 255), 1)
                
        else:
            x = int(round((lRho[k]*rhoRes) - diagonal / cos_theta[k]))
            cv2.line(new, (0, x), (Im_w, x), (255, 255, 255), 1)

        subtract = np.nonzero(new)
        segment = np.zeros(Im.shape)
        end = False
        start_cand = []
        end_cand = []
        distance = []
        for i in range(len(subtract[0])):
            if 30 <= new[subtract[0][i], subtract[1][i]] - Im[subtract[0][i], subtract[1][i]] < 230:
                if not end:
                    start_cand.append([subtract[1][i], subtract[0][i]])
                    end = True
            else:
                if end:
                    end_cand.append([subtract[1][i], subtract[0][i]])
                    end = False

        if end:
            end_cand.append([subtract[1][-1], subtract[0][-1]])
        
        if not start_cand and not end_cand:
            start = tuple(['start', tuple([-1, -1])])
            end = tuple(['end', tuple([-1, -1])])
        else:
            for i in range(len(start_cand)):
                dist = ((end_cand[i][0] - start_cand[i][0]) ** 2 + (end_cand[i][1] - start_cand[i][1]) ** 2) ** 0.5
                distance.append(dist)

        if distance:
            argidx = np.argmax(np.array(distance))
            start = tuple(['start', tuple(start_cand[argidx])])
            end = tuple(['end', tuple(end_cand[argidx])])

        l_idx = dict([start, end])
        l.append(l_idx)

    return l

def main():

    # read images
    for img_path in glob.glob(datadir+'/*.jpg'):
        print(img_path)
        # load grayscale image
        img = Image.open(img_path).convert("L")           

        Igs = np.array(img)
        Igs = Igs / 255.

        # Hough function
        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma)
        H= HoughTransform(Im,threshold, rhoRes, thetaRes)
        lRho,lTheta =HoughLines(H,rhoRes,thetaRes,nLines)

        # Plot HoughLines
        # Im_h, Im_w = Im.shape
        # for k in range(nLines):
        #     diagonal = int(np.ceil(np.hypot(Im_h, Im_w)))
        #     cos_theta = np.cos(np.deg2rad(lTheta))
        #     sin_theta = np.sin(np.deg2rad(lTheta))

        #     if lTheta[k] != 0:
        #         x1 = 0
        #         y1 = int(round(((lRho[k]*rhoRes) - diagonal - x1 * cos_theta[k]) / sin_theta[k]))
        #         start_x = 0
        #         start_y = y1
        #         x2 = Im_w - 1
        #         y2 = int(round(((lRho[k]*rhoRes) - diagonal - x2 * cos_theta[k]) / sin_theta[k]))
        #         end_x = Im_w - 1
        #         end_y = y2
        #         cv2.line(Igs, (start_x, start_y), (end_x, end_y), (255, 255, 255), 1)
                    
        #     else:
        #         x = int(round((lRho[k]*rhoRes) - diagonal / cos_theta[k]))
        #         cv2.line(Igs, (0, x), (Im_w, x), (255, 255, 255), 1)

        # Igs = Igs / Igs.max() * 255
        # image = Image.fromarray(Igs)
        # image.show()

        l = HoughLineSegments(lRho, lTheta, Im, threshold)

        # Plot HoughLineSegments
        # Igs = Igs / Igs.max() * 255
        # for i in range(len(l)):
        #     cv2.line(Igs, l[i]['start'], l[i]['end'], (255, 255, 255), 1)

        # image = Image.fromarray(Igs)
        # image.show()

        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments
        
        

if __name__ == '__main__':
    main()