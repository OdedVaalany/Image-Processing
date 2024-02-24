import numpy as np
from decorators import colorImageSupport
from kernels import expends, gaussian_blur
from matplotlib import pyplot as plt


@colorImageSupport
def pyramidUp(image):
    """
    Expands the image by a factor of 2.

    Args:
        image (ndarray): The input image.

    Returns:
        ndarray: The expanded image.
    """
    return expends(image)


@colorImageSupport
def pyramidDown(image, blur_size=1):
    """
    reduce the image by a factor of 2.

    Args:
        image (ndarray): The input image.

    Returns:
        ndarray: The small image.
    """
    if min(image.shape) < 2:
        return image
    return gaussian_blur(image, blur_size)[::2, ::2]


def buildGaussianPyramid(image, levels=-1):
    """
    Builds a Gaussian pyramid for an image.

    Args:
        image (ndarray): The input image.
        levels (int): The number of levels in the pyramid.

    Returns:
        list: The Gaussian pyramid.
    """
    if levels == -1:
        levels = int(np.floor(np.log2(min(image.shape[:2]))))
    pyramid = [image]
    for i in range(levels-1):
        pyramid.append(pyramidDown(pyramid[-1]))
    return pyramid


def buildLaplacianPyramid(image, levels=-1):
    """
    Builds a Laplacian pyramid for an image.

    Args:
        image (ndarray): The input image.
        levels (int): The number of levels in the pyramid.

    Returns:
        list: The Laplacian pyramid.
    """
    if levels == -1:
        levels = int(np.floor(np.log2(min(image.shape[:2]))))
    pyramid = []
    for i in range(levels-1):
        pyramid.append(image - pyramidUp(pyramidDown(image))
                       [:image.shape[0], :image.shape[1]])
        image = pyramidDown(image)
    pyramid.append(image)
    return pyramid


def buildImageFromLaplacianPyramid(pyramid):
    """
    Reconstructs an image from a Laplacian pyramid.

    Args:
        pyramid (list): The Laplacian pyramid.

    Returns:
        ndarray: The reconstructed image.
    """
    image = pyramid[-1]
    for i in range(len(pyramid)-2, -1, -1):
        tmp = pyramid[i]
        h, w, _ = tmp.shape
        image = pyramidUp(image)[:h, :w, :] + pyramid[i]
    return image.astype('int')
