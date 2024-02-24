from imageIO import *
from matplotlib import pyplot as plt
from pyramids import *
from alg import *
from utils import *
from transforms import backWarp
import cv2
import numpy as np
image1 = readImage('./in/desert_low_res.jpg', True)
image2 = readImage('./in/desert_high_res.png', True)
# calculate the matrix first
shrink2image1 = pyramidDown(pyramidDown(image1))
shrink2image2 = pyramidDown(pyramidDown(image2))

shrink2corner1 = harrisCornerDedection(shrink2image1, window_size=9)
shrink2corner2 = harrisCornerDedection(shrink2image2, window_size=9)


treshhold1 = getTheNBiggestValue(shrink2corner1, 50)
treshhold2 = getTheNBiggestValue(shrink2corner2, 50)
f, axis = plt.subplots(3, 2)
axis[0, 0].imshow(shrink2image1, cmap='gray')
axis[1, 0].imshow(np.log(shrink2corner1+1), cmap='gray')
axis[2, 0].imshow(np.where(shrink2corner1 >= treshhold1, 255, 0), cmap='gray')
axis[0, 1].imshow(shrink2image2, cmap='gray')
axis[1, 1].imshow(np.log(shrink2corner2+1), cmap='gray')
axis[2, 1].imshow(np.where(shrink2corner2 >= treshhold2, 255, 0), cmap='gray')
plt.show()


centerX = shrink2image2.shape[1]//2
centerY = shrink2image2.shape[0]//2
intrestPointImage1 = np.stack(
    np.where(shrink2corner1 >= treshhold1)[::-1], axis=1)
intrestPointImage2 = np.stack(
    np.where(shrink2corner2 >= treshhold2)[::-1], axis=1)
intrestPointImage1[:] -= [centerX, centerY]
intrestPointImage2[:] -= [centerX, centerY]
bestMatrix = np.matrix([[1, 0], [0, 1]])
bestMatches = 0
while (bestMatches < 42):
    aPoints = selectNLines(intrestPointImage1, 2).T
    bPoints = selectNLines(intrestPointImage2, 2).T
    if (np.linalg.det(bPoints) == 0):
        continue
    M = aPoints @ np.linalg.inv(bPoints)
    transformedBPoints = (M @ intrestPointImage2.T).T
    dis = np.sqrt(
        np.sum((intrestPointImage1[:, np.newaxis]-transformedBPoints)**2, axis=2))
    count = np.count_nonzero(np.abs(dis) < 6)
    if (count > bestMatches):
        bestMatrix = M
        bestMatches = count

print(bestMatches)
print(bestMatrix)

coords = np.zeros((shrink2image2.shape[0]*shrink2image2.shape[1], 2))
coords[:, 1] = np.repeat(np.arange(shrink2image2.shape[0]),
                         shrink2image2.shape[1])-centerY
coords[:, 0] = np.tile(np.arange(shrink2image2.shape[1]),
                       shrink2image2.shape[0])-centerX
destCoords = (np.linalg.inv(bestMatrix) @ coords.T).T
destCoords[:] += [centerX, centerY]
em = np.zeros(shrink2image2.shape)
backWarp(shrink2image2, em, destCoords)
print("dfsdfd", em.shape, em.min(),    em.max())
# plt.imshow(em.astype('int'), cmap='gray')
# plt.show()

# newIntrestPointImage2 = (np.linalg.inv(bestMatrix) @
#                          intrestPointImage2.T).T.astype('int')
# pairs = []
# for i, pointA in enumerate(intrestPointImage1):
#     if (np.sum(em[pointA[1]+centerY-2:pointA[1]+centerY+3, pointA[0]+centerX-2:pointA[0]+centerX+3]) == 0):
#         continue
#     windowA = shrink2image1[pointA[1]+centerY-2:pointA[1] +
#                             centerY+3, pointA[0]+centerX-2:pointA[0]+centerX+3]
#     minVal = np.Inf
#     bestPoint = None
#     for j, pointB in enumerate(newIntrestPointImage2):
#         windowB = em[pointB[1]+centerY-2:pointB[1]+centerY +
#                      3, pointB[0]+centerX-2:pointB[0]+centerX+3]
#         if minVal > np.sum(((windowA-windowB)**2)/(np.sqrt(windowA**2+windowB**2))):
#             minVal = np.sum(((windowA-windowB)**2) /
#                             (np.sqrt(windowA**2+windowB**2)))
#             bestPoint = pointB
#     pairs.append((pointA+[centerX, centerY],
#                  bestPoint+[centerX, centerY], minVal))
# a, b, _ = min(pairs, key=lambda p: p[2])
# print(a-b)
# coords = np.zeros((shrink2image2.shape[0]*shrink2image2.shape[1], 2))
# coords[:, 1] = np.repeat(
#     np.arange(shrink2image2.shape[0]), shrink2image2.shape[1])
# coords[:, 0] = np.tile(np.arange(shrink2image2.shape[1]),
#                        shrink2image2.shape[0])
# destCoords[:] = coords
# destCoords[:] -= (a-b)[::-1]
# MovedImage = np.zeros(shrink2image2.shape)
# backWarp(em, MovedImage, destCoords)
# plt.imshow(MovedImage, cmap='gray')
# plt.show()
