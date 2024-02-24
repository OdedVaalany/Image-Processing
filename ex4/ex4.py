import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import PIL
from transforms import backWarp

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

    ############################## calculate the best translations between the intrest points of both images ##############################
    pairsAfterTransform = pairs.copy()[:, :, :2]
    pairsAfterTransform[:, 1] = (bestMatrix @ pairs[:, 1].T).T[:, :2]
    bestMoveVec = [0, 0]
    bestMatches = 0
    for i in range(len(pairsAfterTransform)):
        selected = pairsAfterTransform[np.random.randint(0, pairs.shape[0], 1)]
        aPoints = selected[:, 0]
        bPoints = selected[:, 1]
        moveVec = aPoints-bPoints
        transformedBPoints = pairsAfterTransform[:, 1]+moveVec
        dis = np.sqrt(
            np.sum((pairsAfterTransform[:, 0]-transformedBPoints)**2, axis=1))
        count = np.count_nonzero(dis < 10)
        if (count > bestMatches):
            bestMoveVec = moveVec
            bestMatches = count
    print("bestMatches: ", bestMatches, " out of ", len(pairsAfterTransform))
    print("bestMoveVec: ", bestMoveVec)
    ############################## warp the second image to the first image using the best transformation matrix ##################################
    # coords = np.zeros((image2.shape[0]*image2.shape[1], 2))
    # coords[:, 1] = np.repeat(np.arange(image2.shape[0]),
    #                          image2.shape[1])-image2.shape[0]//2
    # coords[:, 0] = np.tile(np.arange(image2.shape[1]),
    #                        image2.shape[0])-image2.shape[1]//2
    # destCoords = (np.linalg.inv(bestMatrix[:2, :2]) @ coords.T).T
    # print(bestMatrix[:2, 2].T)
    # destCoords[:] += 4*bestMatrix[:2, 2].T
    # destCoords[:] += [image2.shape[1]//2, image2.shape[0]//2]
    # em = np.zeros(image2.shape)
    # backWarp(image2, em, destCoords)
    # plt.imshow(em)
    # plt.show()

    rows, cols = image2.shape[:2]
    M = bestMatrix
    M[0, 2] -= image2.shape[1]//2
    M[1, 2] -= image2.shape[0]//2
    dst = cv2.warpPerspective(image2, M, (cols, rows))
    plt.imshow(dst)
    plt.show()

    # f, a = plt.subplots(1, 3)
    # a[0].imshow(image1)
    # a[1].imshow(image2)
    # a[2].imshow(mask)
    # plt.show()
