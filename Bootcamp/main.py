import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
# print("hello world this is boot camp")

# x = np.random.rand(9, 1)
# x = x.reshape([3, 3])
# print(x)
# print("matrix shape", x.shape)
# print("matrix type", x.dtype)

# y = np.matmul(x[:, 0], x[:, 2])
# print("value of the first and third column multipication", y)
# y = np.array([[y]*9]*9)
# print("The matrix 9x9 filled with the previouse value\n", y)
# z = np.random.rand(9, 9)
# print("a random values matrix of 9x9\n", y)
# t = np.stack([z, y], 2)
# print("after stacking\n", t)
# print("the sahpe is", t.shape)
# print("The mean of the new matrix: ", np.mean(t, 2))
# print("The sum of the new matrix: ", np.sum(t, 2))
# print("The standard deviation of the new matrix: ", np.std(t, 2))
# tt = np.where(t > np.mean(t), t, 0)

# print(tt)

# t1 = np.random.rand(512, 512, 3)
# t2 = np.random.rand(512, 512, 3)
# print(np.einsum('ijk,ijk->ijk', t1, t2))


x = np.asarray(Image.open(
    "/Users/odedvaalany/development/Intro To Image Proccessing/Bootcamp/temp.jpg"))
print(Image.open(
    "/Users/odedvaalany/development/Intro To Image Proccessing/Bootcamp/temp.jpg"))
# print(repr(x))
print((0.299*x[:, :, 0]+0.587*x[:, :, 1]+0.114*x[:, :, 2]).shape)
imgplot = plt.imshow(0.299*x[:, :, 0]+0.587 *
                     x[:, :, 1]+0.114*x[:, :, 2], cmap='Grays', interpolation='nearest')
plt.show()
