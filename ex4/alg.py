import numpy as np
from decorators import algColorImageSupport
from kernels import gaussian_blur, derevative_x, derevative_y
from pyramids import buildGaussianPyramid
from scipy.signal import convolve2d
from transforms import translate


@algColorImageSupport
def harrisCornerDedection(image, k=0.04, window_size=5):
    """
    Applies Harris corner detection algorithm on the given image.

    Parameters:
    - image: numpy.ndarray
        The input image.
    - k: float, optional
        Harris corner constant. Default is 0.04.
    - window_size: int, optional
        Size of the window used for corner detection. Default is 5.

    Returns:
    - numpy.ndarray
        The image with detected corners.
    """
    if (k > 0.06 or k < 0.04):
        raise ValueError("k should be between 0.04 and 0.6")
    if (window_size % 2 == 0):
        raise ValueError("window_size should be odd")
    if (window_size < 3):
        raise ValueError("window_size should be greater than or equal to 3")
    base = np.matrix([[1, 2, 1]])/4
    ker = np.copy(base)
    for i in range(1, window_size//2):
        ker = convolve2d(ker, base, mode='full')
    ker = convolve2d(ker, ker.T, mode='full')
    emp = np.zeros(image.shape[:2])
    im = gaussian_blur(np.copy(image), 2)
    for i in range(window_size//2, image.shape[0]-window_size//2):
        for j in range(window_size//2, image.shape[1]-window_size//2):
            window = im[i-window_size//2:i+window_size //
                        2+1, j-window_size//2:j+window_size//2+1]
            der_x = derevative_x(window)
            der_y = derevative_y(window)
            M = np.matrix([[np.sum(ker*der_x**2), np.sum(ker*der_x*der_y)],
                          [np.sum(ker*der_x*der_y), np.sum(ker*der_y**2)]])
            emp[i, j] = np.linalg.det(M)-k*(np.trace(M)**2)
    nemp = np.zeros(emp.shape)
    for i in range(2, emp.shape[0]-2):
        for j in range(2, emp.shape[1]-2):
            if (emp[i, j] == np.max(emp[i-1:i+2, j-1:j+2])):
                nemp[i, j] = emp[i, j]
    return nemp


def harrisCornerDedectionMultiScale(image, k=0.04, window_size=5):
    """
    Performs multi-scale Harris corner detection on an image.

    Args:
        image (numpy.ndarray): The input image.
        k (float, optional): Harris corner constant. Defaults to 0.04.
        window_size (int, optional): Size of the window for corner detection. Defaults to 5.

    Returns:
        list: List of the R image on each levels in the pyramid.
    """
    gauss_pyramid = buildGaussianPyramid(image)
    interst_points = []
    for im in gauss_pyramid:
        if min(im.shape) < window_size:
            break
        interst_points.append(harrisCornerDedection(im, k, window_size))
        if (window_size > 3):
            window_size = (window_size)//2 + (1-((window_size)//2) % 2)
    return interst_points


def CC(im1, im2):
    """
    Cross correlation between two images.

    Args:
        im1 (numpy.ndarray): First image.
        im2 (numpy.ndarray): Second image.

    Returns:
        numpy.ndarray: Cross correlation between the two images.
    """
    if (min(im1.shape) < min(im2.shape)):
        raise ValueError("Image 1 should be larger than image 2")
    h, w = im1.shape
    h2, w2 = im2.shape
    im2_ = np.pad(im2, ((0, h-h2), (0, w-w2)))
    emp = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            emp[i, j] = np.sum(im1*translate(im2_, j, i))
    return emp


def NCC(im1, im2):
    """
    Normalized cross correlation between two images.

    Args:
        im1 (numpy.ndarray): First image.
        im2 (numpy.ndarray): Second image.

    Returns:
        numpy.ndarray: Normalized cross correlation between the two images.
    """
    return CC(im1-np.mean(im1), im2-np.mean(im2))/(np.sqrt(np.sum((im1-np.mean(im1))**2)*np.sum((im2-np.mean(im2))**2)))


def LKTranslation(image1, image2):
    """
    Lucas-Kanade translation estimation.

    Args:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image.

    Returns:
        tuple: The estimated translation.
    """
    Ix = derevative_x(image1)
    Iy = derevative_y(image1)
    x, y = 0, 0
    M = np.matrix([[np.sum(Ix**2), np.sum(Ix*Iy)],
                   [np.sum(Ix*Iy), np.sum(Iy**2)]])
    It = image2 - image1
    while np.sum(np.abs(It/255)) > 0.1:
        It = translate(image2, x, y, 1000) - image1
        b = np.matrix([[np.sum(Ix*It)], [np.sum(Iy*It)]])
        v = np.linalg.solve(M, -b)
        x += v[0, 0]
        y += v[1, 0]
        print(x, y)
        if (abs(x) > image1.shape[1] or abs(y) > image1.shape[0]):
            break
    return (x, y)
