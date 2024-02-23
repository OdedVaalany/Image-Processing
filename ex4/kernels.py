import numpy as np
from scipy.signal import convolve2d
from decorators import colorImageSupport


@colorImageSupport
def gaussian_blur(image, NN=1):
    """
    blur using gaussian ketnel.

    Args:
        image (ndarray): The input image.
        NN (int): The number of neighbors to include in the blurring.

    Returns:
        ndarray: The Gaussian Image.

    Raises:
        ValueError: If `NN` is less than 1.
    """
    if NN < 1:
        raise ValueError("NN must be greater than or equal to 1.")

    base = np.matrix([1, 2, 1])/4
    base_x_gaussian_kernel = np.matrix([1, 2, 1])/4
    im = np.copy(image)
    for i in range(1, NN):
        base_x_gaussian_kernel = convolve2d(
            base_x_gaussian_kernel, np.matrix([1, 2, 1])/4, mode='full')
    return convolve2d(convolve2d(im, base_x_gaussian_kernel, mode='same', boundary='symm'), base_x_gaussian_kernel.T, mode='same', boundary='symm')


@colorImageSupport
def derevative_x(image):
    """
    Compute the derivative of the image in the x direction.

    Args:
        image (ndarray): The input image.

    Returns:
        ndarray: The derivative of the image in the x direction.
    """
    kernel = np.matrix([[-1, 1]])
    return convolve2d(image, kernel, mode='same', boundary='symm')


@colorImageSupport
def derevative_y(image):
    """
    Compute the derivative of the image in the y direction.

    Args:
        image (ndarray): The input image.

    Returns:
        ndarray: The derivative of the image in the y direction.
    """
    kernel = np.matrix([[-1], [1]])
    return convolve2d(image, kernel, mode='same', boundary='symm')


@colorImageSupport
def laplacian(image):
    """
    Compute the laplacian of the image.

    Args:
        image (ndarray): The input image.

    Returns:
        ndarray: The laplacian of the image.
    """
    kernel = np.matrix([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return convolve2d(image, kernel, mode='same')


@colorImageSupport
def expends(image):
    """
    Expends the image by a factor of 2.

    Args:
        image (ndarray): The input image.

    Returns:
        ndarray: The expanded image.
    """
    padded = np.kron(image, np.matrix([[1, 0], [0, 0]]))
    return convolve2d(convolve2d(padded, np.matrix([[1/2, 1, 1/2]]), mode='same'), np.matrix([[1/2], [1], [1/2]]), mode='same')
