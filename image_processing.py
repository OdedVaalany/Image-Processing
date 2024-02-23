import numpy as np
from scipy.signal import convolve2d


def gaussian_blur(image, kernel_size, sigma):
    kernel_x = np.matrix([[1, 2, 1]])/4
    kernel_y = np.matrix([[1, 2, 1]])/4
    for i in range(kernel_size//2-1):
        kernel_x = convolve2d(kernel_x, kernel_x, mode='full')
        kernel_y = convolve2d(kernel_y, kernel_y, mode='full')
    return convolve2d(convolve2d(image, kernel_x, mode='same', boundary='symm'), kernel_y, mode='same', boundary='symm')
