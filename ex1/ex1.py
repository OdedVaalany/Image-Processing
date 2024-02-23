from PIL import Image
import mediapy
import numpy as np
from matplotlib import pyplot as plt


def video2Grays(video: np.array) -> np.array:
    """
    Util function helps to convert video to grays sacle using the class formula
    :param video: need to be array of pictures in np
    :return grayscales video as np.array
    """
    return np.round(0.299*video[:, :, :, 0]+0.587 * video[:, :, :, 1]+0.114*video[:, :, :, 2]).astype(np.uint8)


def videoToHistograms(video: np.array, toCum: bool = False) -> np.array:
    """
    Util function helps to convert video to histogram
    :param video: need to be array of pictures in np
    :param toCum: choosing the historam to be cumulative or not
    :return array of histograms of types 'float'
    """
    rangeVals = np.arange(257)
    dens = np.zeros((video.shape[0], 256), dtype='float')
    for i in range(video.shape[0]):
        dens[i, :] = np.histogram(video[i], bins=rangeVals, density=True)[0]
    if toCum:
        return np.cumsum(dens, axis=1)
    return dens


def main(video_path: str, video_type: str):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
    video = np.asarray(mediapy.read_video(video_path))
    grayScaleVideo = video2Grays(video)
    cumulutiveHistogram = videoToHistograms(grayScaleVideo, True)
    meanCumulutiveHistogram = np.mean(cumulutiveHistogram, axis=1)
    cumHistGaps = np.abs(
        meanCumulutiveHistogram - np.roll(meanCumulutiveHistogram, 1, axis=0))
    cutEndFrame = np.argmax(cumHistGaps[1:])
    return (cutEndFrame, cutEndFrame+1)
