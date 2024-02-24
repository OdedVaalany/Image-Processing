import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import PIL


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


if __name__ == "__main__":
    # first read the images
    image1 = np.asarray(Image.open('./in/desert_low_res.jpg'))
    image2 = np.asarray(Image.open('./in/desert_high_res.png'))

    # seperating the mask from image 2
    mask = image2[:, :, 3]
    image2 = image2[:, :, :3]

    # find intrest points using harris for image 1 on 2 scales below
    image1grays = cv2.pyrDown(cv2.pyrDown(cv2.cvtColor(
        image1.astype('float32'), cv2.COLOR_RGB2GRAY)))
    image2grays = cv2.pyrDown(cv2.pyrDown(cv2.cvtColor(
        image2.astype('float32'), cv2.COLOR_RGB2GRAY)))
    intrestPointImage1 = cv2.goodFeaturesToTrack(
        image1grays, 50, 0.01, 10)[:, 0, :]
    intrestPointImage2 = cv2.goodFeaturesToTrack(
        image2grays, 50, 0.01, 10)[:, 0, :]
    ############################## plot the intrest points of both images ##############################
    # copyIm1 = image1grays.copy()
    # im = np.zeros_like(image1grays)
    # im[intrestPointImage1[:, 1].astype(int),
    #    intrestPointImage1[:, 0].astype(int)] = 255
    # cv2.dilate(im, np.ones((11, 11), np.uint8), im)
    # copyIm1 = cv2.add(copyIm1, im)
    # copyIm2 = image2grays.copy()
    # im = np.zeros_like(image2grays)
    # im[intrestPointImage2[:, 1].astype(int),
    #    intrestPointImage2[:, 0].astype(int)] = 255
    # cv2.dilate(im, np.ones((11, 11), np.uint8), im)
    # copyIm2 = cv2.add(copyIm2, im)

    # f, a = plt.subplots(1, 2)
    # a[0].imshow(copyIm1)
    # a[1].imshow(copyIm2)
    # plt.show()

    ############################## find the best matches between the intrest points of both images ##############################

    NN = 2
    pairs = []
    for pt1 in intrestPointImage1.astype(int):
        for pt2 in intrestPointImage2.astype(int):
            if np.median(np.abs(image1grays[pt1[1]-NN:pt1[1]+NN+1, pt1[0]-NN:pt1[0]+NN+1] - image2grays[pt2[1]-NN:pt2[1]+NN+1, pt2[0]-NN:pt2[0]+NN+1])) < 20:
                pairs.append([pt1, pt2])
    pairs = np.array(pairs)
    print("number of pairs: ", pairs.shape[0])

    ############################## find the best transformation matrix between the intrest points of both images ##############################
    pairs = np.pad(pairs, ((0, 0), (0, 0), (0, 1)),
                   'constant', constant_values=1).astype('float32')
    pairs[:, :, :] -= [image1grays.shape[1]//2, image1grays.shape[0]//2, 0.5]
    bestMatrix = np.eye(3)
    bestMatches = 0
    for i in range(10000):
        selected = pairs[np.random.randint(0, pairs.shape[0], 3)]
        aPoints = selected[:, 0]
        bPoints = selected[:, 1]
        if (np.linalg.det(bPoints) == 0):
            continue
        M = aPoints.T @ np.linalg.inv(bPoints.T)
        transformedBPoints = (M @ pairs[:, 1].T).T
        dis = np.sqrt(
            np.sum((pairs[:, 0]-transformedBPoints)**2, axis=1))
        count = np.count_nonzero(dis < 9)
        if (count > bestMatches):
            bestMatrix = M
            bestMatches = count
    print("bestMatches: ", bestMatches)
    print("bestMatrix: ", bestMatrix)

    ############################## warp the second image to the first image using the best transformation matrix ##################################
    coords = np.zeros((image2.shape[0]*image2.shape[1], 2))
    coords[:, 1] = np.repeat(np.arange(image2.shape[0]),
                             image2.shape[1])-image2.shape[0]//2
    coords[:, 0] = np.tile(np.arange(image2.shape[1]),
                           image2.shape[0])-image2.shape[1]//2
    destCoords = (np.linalg.inv(bestMatrix[:2, :2]) @ coords.T).T
    destCoords[:] += bestMatrix[:2, 2].T
    destCoords[:] += [image2.shape[1]//2, image2.shape[0]//2]

    em = np.zeros(image2.shape)
    wrapedMask = np.zeros(mask.shape)
    backWarp(image2, em, destCoords)
    backWarp(mask, wrapedMask, destCoords)
    # plt.imshow(em.astype('int'))
    # plt.show()
    # plt.imshow(wrapedMask, cmap='gray')
    # plt.show()

    ############################## blend the two images together using the mask ##################################

    mask_ = wrapedMask/255
    mask_ = np.stack([mask_, mask_, mask_], axis=2)
    blendedImage = mask_*em + (1-mask_)*image1
    plt.imshow(blendedImage.astype('int'))
    plt.show()

    # h, w = image2.shape[:2]
    # corners = np.array(
    #     [[-w//2, w//2, w//2, -w//2], [-h//2, -h//2, h//2, h//2], [1, 1, 1, 1]])
    # cornersAfterTransforms = bestMatrix @ corners
    # width = int(
    #     np.max(cornersAfterTransforms[0])-np.min(cornersAfterTransforms[0]))
    # height = int(
    #     np.max(cornersAfterTransforms[1])-np.min(cornersAfterTransforms[1]))
    # moveX = int(np.mean([
    #     cornersAfterTransforms[0, 0]+w//2, cornersAfterTransforms[0, 0]+w//2]))
    # moveY = int(np.mean([
    #     cornersAfterTransforms[0, 1]+h//2, cornersAfterTransforms[-1, 1]+h//2]))
    # print(width, height, w, h, moveX, moveY)
    # rows, cols = h, w
    # M = bestMatrix
    # M[0, 2] -= MoveVec[0]*4
    # M[1, 2] += MoveVec[1]*4
    # dst = cv2.warpPerspective(image2, M, (cols, rows))
    # plt.imshow(dst)
    # plt.show()

    # f, a = plt.subplots(1, 3)
    # a[0].imshow(image1)
    # a[1].imshow(image2)
    # a[2].imshow(mask)
    # plt.show()
