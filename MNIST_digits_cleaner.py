import numpy
import keras
from keras.datasets import mnist

# download dataset, randomly sort the data for both test and training data
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
idx = numpy.argsort(numpy.random.random(ytrain.shape[0]))
xtrain = xtrain[idx]
ytrain = ytrain[idx]
idx = numpy.argsort(numpy.random.random(ytest.shape[0]))
xtest = xtest[idx]
ytest = ytest[idx]

# save randomized datasets to file
numpy.save("MNIST digits dataset/mnist_train_images.npy", xtrain)
numpy.save("MNIST digits dataset/mnist_train_labels.npy", ytrain)
numpy.save("MNIST digits dataset/mnist_test_images.npy", xtest)
numpy.save("MNIST digits dataset/mnist_test_labels.npy", ytest)

# vectorize the data set by removing the additional dimension and converting it into a single dimension of 28*28
xtrain_vectorized = xtrain.reshape((60000, 28*28))
xtest_vectorized = xtest.reshape((10000, 28*28))
numpy.save("MNIST digits dataset/mnist_train_vectors.npy", xtrain_vectorized)
numpy.save("MNIST digits dataset/mnist_test_vectors.npy", xtest_vectorized)

# randomize the order of the values within the vector but keep the random values the same for both training and test
idx = numpy.argsort(numpy.random.random(28*28))
for i in range(60000):
    xtrain_vectorized[i, :] = xtrain_vectorized[i, idx]
for i in range(10000):
    xtest_vectorized[i, :] = xtest_vectorized[i, idx]
numpy.save("MNIST digits dataset/mnist_train_scrambled_vectors.npy", xtrain_vectorized)
numpy.save("MNIST digits dataset/mnist_test_scrambled_vectors.npy", xtest_vectorized)

# un-vectorize the scrambled data set
t = numpy.zeros((60000, 28, 28))
for i in range(60000):
    t[i, :, :] = xtrain_vectorized[i, :].reshape((28, 28))
numpy.save("MNIST digits dataset/mnist_train_scrambled_images.npy", t)
t = numpy.zeros((10000, 28, 28))
for i in range(10000):
    t[i, :, :] = xtest_vectorized[i, :].reshape((28, 28))
numpy.save("MNIST digits dataset/mnist_test_scrambled_images.npy", t)