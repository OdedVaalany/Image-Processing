import numpy as np
from decorators import colorImageSupport
from typing import List, Tuple, Union


def normalize(image):
    """
    Normalize an image to the range [0, 1].

    Args:
        image (ndarray): The input image.

    Returns:
        ndarray: The normalized image.
    """
    return (image - np.min(image))/(np.max(image) - np.min(image))


def getTheNBiggestValue(image, n=1):
    """
    Get the n biggest values in an image.

    Args:
        emp (ndarray): The input image.
        n (int): The number of values to get.

    Returns:
        list: The n biggest values.
    """
    if (n < 1):
        raise ValueError("n should be greater than or equal to 1")
    vals, counts = np.unique(image, return_counts=True)
    for i in range(-1, -len(vals)-1, -1):
        if (np.sum(counts[i:]) >= n):
            return vals[i]
    return vals[0]


@colorImageSupport
def quantizeImage(image, n, mode: Union["mid", "min", "max", "avg"] = "mid"):
    """
    Quantize an image to n levels.

    Args:
        image (ndarray): The input image.
        n (int): The number of levels.

    Returns:
        ndarray: The quantized image.
    """
    if mode == "mid":
        return normalize(np.round(image/(255/(n-1)))) * 255
    elif mode == "min":
        return normalize(np.floor(image/(255/(n-1)))) * 255
    elif mode == "max":
        return normalize(np.ceil(image/(255/(n-1)))) * 255
    return normalize(np.round(image/(255/(n-1)))) * 255


@colorImageSupport
def equazition(image):
    """
    Equalize an image.

    Args:
        image (ndarray): The input image.

    Returns:
        ndarray: The equalized image.
    """
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    return np.interp(image.flatten(), bins[:-1], cdf_normalized).reshape(image.shape)


def selectNLines(arr, N=1):
    """
    Selects N random lines from the given array.

    Parameters:
    arr (ndarray): The input array.
    N (int): The number of lines to select. Default is 1.

    Returns:
    ndarray: An array containing N randomly selected lines from the input array.
    """
    return arr[np.random.randint(0, arr.shape[0], N, dtype='int')]
