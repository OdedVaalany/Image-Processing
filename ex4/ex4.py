import cv2
import numpy as np
import imageIO
import utils
import transforms
from matplotlib import pyplot as plt

low_res = imageIO.readImage('./in/desert_low_res.jpg')
high_res = imageIO.readImage('./in/desert_high_res.png')

gray_low_res = imageIO.RGB2Grays(low_res)
gray_high_res = imageIO.RGB2Grays(high_res)


def NMS(image, NN=2):
    return image


def get3LevelsCorners(image):
    level1 = NMS(cv2.cornerHarris(np.float32(image), 2, 3, 0.04))
    level2 = NMS(cv2.cornerHarris(np.float32(cv2.pyrDown(image)), 2, 3, 0.04))
    level3 = NMS(cv2.cornerHarris(np.float32(
        cv2.pyrDown(cv2.pyrDown(image))), 2, 3, 0.04))
    # f, axis = plt.subplots(2, 2)
    p1 = np.where(level1 >= 0.01*level1.max(), 255.0, 0)
    p2 = np.where(level2 >= 0.01*level2.max(), 255.0, 0)
    p3 = np.where(level3 >= 0.01*level3.max(), 255.0, 0)
    p4 = NMS(
        cv2.pyrUp(p2, dstsize=level1.shape[::-1])+p1+cv2.pyrUp(cv2.pyrUp(p3, dstsize=level2.shape[::-1]), dstsize=level1.shape[::-1]))

    # axis[0, 0].imshow(p1, cmap='gray')
    # axis[0, 1].imshow(p2, cmap='gray')
    # axis[1, 0].imshow(p3, cmap='gray')
    # axis[1, 1].imshow(p4, cmap='gray')
    intrestPointImage1 = np.where(p4 >= 700)
    return np.stack([intrestPointImage1[1], intrestPointImage1[0]], axis=1)


def matchPoints(image1, image2, ws=2):
    intrestPointImage1 = get3LevelsCorners(image1)
    intrestPointImage2 = get3LevelsCorners(image2)
    return intrestPointImage1, intrestPointImage2


def showImagesIntrestPoints(image, intrestPointImage):
    t = image.copy()
    s = np.zeros(image.shape[:2])
    print(intrestPointImage.shape)
    s[(intrestPointImage[:, 1], intrestPointImage[:, 0])] = 255
    cv2.dilate(s, np.ones((9, 9), np.uint8), s)
    t[s == 255] = [255, 0, 0]
    plt.imshow(t)
    plt.show()


def runRANSAC(A, B, iterations=10000, threshold=10):
    bestMatrix = np.eye(2)
    bestMatches = 0
    while bestMatches < 150:
        aPoints = A[np.random.randint(
            0, A.shape[0], 2, dtype='int')]
        bPoints = B[np.random.randint(
            0, B.shape[0], 2, dtype='int')]
        if (np.linalg.det(bPoints) == 0):
            continue
        M = aPoints.T @ np.linalg.inv(bPoints.T)
        transformedBPoints = (M @ B.T).T
        dis = np.sqrt(
            np.sum((A[:, np.newaxis]-transformedBPoints)**2, axis=2))
        count = np.count_nonzero(dis <= threshold)
        if (count > bestMatches):
            bestMatrix = M
            bestMatches = count
    print("Best pairs are match is: ", bestMatches)
    return bestMatrix


def transformImage(image, M):
    dest = np.zeros(image.shape)
    coords = np.zeros((image.shape[0]*image.shape[1], 2))
    coords[:, 1] = np.repeat(np.arange(image.shape[0]),
                             image.shape[1])-image.shape[0]//2
    coords[:, 0] = np.tile(np.arange(image.shape[1]),
                           image.shape[0])-image.shape[1]//2
    destCoords = (np.linalg.inv(M) @ coords.T).T
    destCoords[:] += [image.shape[1]//2, image.shape[0]//2]
    transforms.backWarp(image, dest, destCoords)
    return dest


point1, point2 = matchPoints(gray_low_res, gray_high_res)
print("number of points from the low res image: ", point1.shape[0])
print("number of points from the high res image: ", point2.shape[0])
point1[:, 0] -= gray_low_res.shape[1]//2
point2[:, 0] -= gray_low_res.shape[1]//2
point1[:, 1] -= gray_low_res.shape[0]//2
point2[:, 1] -= gray_low_res.shape[0]//2
M = runRANSAC(point1, point2, threshold=15)
dest = transformImage(gray_high_res, M)
plt.imshow(dest, cmap='gray')
plt.show()
