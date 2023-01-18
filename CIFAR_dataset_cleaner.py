import numpy
import keras
from keras.datasets import cifar10

# load testing and training data
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
# randomize the datasets
idx = numpy.argsort(numpy.random.random(ytrain.shape[0]))
xtrain = xtrain[idx]
ytrain = ytrain[idx]
idx = numpy.argsort(numpy.random.random(ytest.shape[0]))
xtest = xtest[idx]
ytest = ytest[idx]

# save the datasets
numpy.save("CIFAR10 dataset/cifar10_train_images.npy", xtrain)
numpy.save("CIFAR10 dataset/cifar10_train_labels.npy", ytrain)
numpy.save("CIFAR10 dataset/cifar10_test_images.npy", xtest)
numpy.save("CIFAR10 dataset/cifar10_test_labels.npy", ytest)

# vectorize the data
xtrain_vector = xtrain.reshape((50000, 32*32*3))
xtest_vector = xtest.reshape((10000, 32*32*3))
# save the vectorized data
numpy.save("CIFAR10 dataset/cifar10_vector_train_images.npy", xtrain_vector)
numpy.save("CIFAR10 dataset/cifar10_vector_test_images.npy", xtest_vector)




import numpy
from PIL import Image

def augment(im, dim):
    img = Image.fromarray(im)
    if (numpy.random.random() < 0.5):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if (numpy.random.random() < 0.3333):
        z = (32-dim)/2
        r = 10*numpy.random.random()-5
        img = img.rotate(r, resample=Image.BILINEAR)
        img = img.crop((z,z,32-z,32-z))
    else:
        x = int((32-dim-1)*numpy.random.random())
        y = int((32-dim-1)*numpy.random.random())
        img = img.crop((x, y, x+dim, y+dim))
    return numpy.array(img)

def main():
    x = numpy.load("CIFAR10 dataset/cifar10_train_images.npy")
    y = numpy.load("CIFAR10 dataset/cifar10_train_labels.npy")
    factor = 10
    dim = 28
    z = (32-dim)/2
    newx = numpy.zeros((x.shape[0]*factor, dim,dim,3), dtype="uint8")
    newy = numpy.zeros(y.shape[0]*factor, dtype="uint8")
    k = 0
    for i in range(x.shape[0]):
        im = Image.fromarray(x[i,:])
        im = im.crop((z,z,32-z,32-z))
        newx[k,...] = numpy.array(im)
        newy[k] = y[i]
        k+=1
        for j in range(factor-1):
            newx[k,...] = augment(x[i,:], dim)
            newy[k] = y[i]
            k+=1
    idx = numpy.argsort(numpy.random.random(newx.shape[0]))
    newx = newx[idx]
    newy = newy[idx]
    numpy.save("CIFAR10 dataset/cifar10_aug_train_images.npy", newx)
    numpy.save("CIFAR10 dataset/cifar10_aug_train_labels.npy", newy)

main()
