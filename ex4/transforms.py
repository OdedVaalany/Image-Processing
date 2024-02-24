import numpy as np
from decorators import colorImageSupport


@colorImageSupport
def translate(im1, dx, dy, constant=0):
    """
    Translates an image by a given displacement in the x and y directions.

    Args:
        im1: The input image.
        dx: The displacement in the x direction.
        dy: The displacement in the y direction.

    Returns:
        The translated image.
    """
    emp = np.ones(im1.shape)*constant
    h, w = im1.shape
    dx
    dy
    coords = np.zeros((emp.shape[0]*emp.shape[1], 3))
    coords[:, 1] = np.repeat(np.arange(emp.shape[0]), emp.shape[1])-dy
    coords[:, 0] = np.tile(np.arange(emp.shape[1]), emp.shape[0])-dx
    coords[:, 2] = 1
    return backWarp(im1, emp, coords)


@colorImageSupport
def scale(im1, sy, sx, mode='same'):
    """
    Scales an image by a given factor in the x and y directions.

    Args:
        im1: The input image.
        sy: The scaling factor in the y direction.
        sx: The scaling factor in the x direction.
        mode: The scaling mode. Default is 'same'.

    Returns:
        The scaled image.
    """
    TM = np.matrix([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
    ITM = np.linalg.inv(TM)
    emp = np.zeros(im1.shape)
    if (mode == 'full'):
        emp = np.zeros((int(im1.shape[0]*sy), int(im1.shape[1]*sx)))
    coords = np.zeros((emp.shape[0]*emp.shape[1], 3))
    coords[:, 1] = np.repeat(np.arange(emp.shape[0]), emp.shape[1])
    coords[:, 0] = np.tile(np.arange(emp.shape[1]), emp.shape[0])
    coords[:, 2] = 1
    coords = (ITM @ coords.T).T
    return backWarp(im1, emp, coords)


@colorImageSupport
def rotate(im1, theta, rotate_origin=(0, 0)):
    """
    Rotates an image by a given angle around a specified origin.

    Args:
        im1: The input image.
        theta: The rotation angle in radians.
        rotate_origin: The rotation origin. Default is (0, 0).

    Returns:
        The rotated image.
    """
    cx, cy = rotate_origin
    TM = np.matrix([[np.cos(theta), -np.sin(theta), cx],
                   [np.sin(theta), np.cos(theta), cy], [0, 0, 1]])
    ITM = np.linalg.inv(TM)
    emp = np.zeros(im1.shape)
    coords = np.zeros((im1.shape[0]*im1.shape[1], 3))
    coords[:, 1] = np.repeat(np.arange(im1.shape[0]), im1.shape[1])
    coords[:, 0] = np.tile(np.arange(im1.shape[1]), im1.shape[0])
    coords[:, 2] = 1
    coords = (ITM @ coords.T).T
    return backWarp(im1, emp, coords+[cx, cy, 0])


def backWarp(src, dest, dest_cords):
    """
    Performs a backward warping operation to map pixels from a source image to a destination image.

    Args:
        src: The source image.
        dest: The destination image.
        dest_cords: The destination coordinates.

    Returns:
        The warped image.
    """
    h, w = dest.shape[:2]
    bh, bw = src.shape[:2]
    for i in range(h):
        for j in range(w):
            x, y = dest_cords[i*w+j, 0], dest_cords[i*w+j, 1]
            if (x < 0 or x > bw-1 or y < 0 or y > bh-1):
                continue
            dest[i, j] = src[int(y), int(x)]*(1-x % 1)*(1-y % 1) +\
                src[int(y), min(int(x)+1, bw-1)]*(x % 1)*(1-y % 1) +\
                src[min(int(y)+1, bh-1), int(x)]*(1-x % 1)*(y % 1) + \
                src[min(int(y)+1, bh-1), min(int(x)+1, bw-1)]*(x % 1)*(y % 1)
    return dest


@colorImageSupport
def flipX(im1):
    """
    Flips an image horizontally.

    Args:
        im1: The input image.

    Returns:
        The flipped image.
    """
    emp = np.zeros(im1.shape)
    emp = im1[:, ::-1]
    return emp


@colorImageSupport
def flipY(im1):
    """
    Flips an image vertically.

    Args:
        im1: The input image.

    Returns:
        The flipped image.
    """
    emp = np.zeros(im1.shape)
    emp = im1[::-1, :]
    return emp
