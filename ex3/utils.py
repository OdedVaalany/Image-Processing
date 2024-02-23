import cv2
import numpy as np


def gaussian_pyramid(image, levels=None):
    """
    Generates a Gaussian pyramid for the given image.

    Parameters:
    image (numpy.ndarray): The input image.
    levels (int): The number of levels in the pyramid. If None, it is calculated based on the image size.

    Returns:
    list: The Gaussian pyramid as a list of images.
    """
    pyramid = [image]
    if levels is None:
        levels = int(np.log2(min(image.shape[:2])))
    for i in range(levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid
