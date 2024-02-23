import numpy as np
import PIL.Image as Image


def readImage(filename, toGrayscale=False):
    img = np.asarray(Image.open(filename))
    if (toGrayscale):
        img = RGB2Grays(img)
    return img


def writeImage(image, filename):
    img = Image.fromarray(image)
    img.save(filename)


def RGB2Grays(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])
