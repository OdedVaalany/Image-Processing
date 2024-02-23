import numpy as np
from imageIO import RGB2Grays


def colorImageSupport(func):
    """
    A simple decorator that applies a function to each channel of a 3-dimensional image array.

    Args:
        func: The function to be applied to each channel.

    Returns:
        The decorated function.
    """
    def wrapper(*args, **kwargs):
        if (args[0].ndim >= 3):
            output = np.stack([func(args[0][:, :, i], *args[1:], **kwargs)
                              for i in range(args[0].shape[2])], axis=2)
            return output
        return func(*args, **kwargs)
    return wrapper


def algColorImageSupport(func):
    """
    A simple decorator that applies a function to each channel of a 3-dimensional image array.

    Args:
        func: The function to be applied to each channel.

    Returns:
        The decorated function.
    """
    def wrapper(*args, **kwargs):
        if (args[0].ndim >= 3):
            return func(RGB2Grays(args[0]), *args[1:], **kwargs)
        return func(*args, **kwargs)
    return wrapper
