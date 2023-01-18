import numpy

# read dataset into a python list
with open("iris.data") as f:
    lines = []
    for i in f.readlines():
        lines.append(i[:-1])

# create a list of the classifications
n = ["Iris-setosa","Iris-versicolor","Iris-virginica"]
# list comprehension converting n from strings into integers (ex: Iris-setosa = 0)
x = [n.index(i.split(",")[-1]) for i in lines if i != ""]
# make x into a numpy array
x = numpy.array(x, dtype="uint8")

# discard the classifcation and split all of the features that are separated by commas and put them in a 2d list
y = [[float(j) for j in i.split(",")[:-1]] for i in lines if i != ""]
# convert 2d list into numpy array
y = numpy.array(y)

# x.shape[0] returns the first dimension of the labels (which is the size of the dataset 150)
# we select a random value within this dataset and then use that value to randomize the dataset
i = numpy.argsort(numpy.random.random(x.shape[0]))
x = x[i]
y = y[i]

# save the prepared data set
numpy.save("iris dataset/iris_labels.npy", x)
numpy.save("iris dataset/iris_features.npy", y)






from sklearn import decomposition

# create PCA components given numpy array
def generateData(pca, x, start):
    original = pca.components_.copy()
    ncomp = pca.components_.shape[0]
    a = pca.transform(x)
    for i in range(start, ncomp):
        pca.components_[i,:] += numpy.random.normal(scale=0.1, size=ncomp)
    b = pca.inverse_transform(a)
    pca.components_ = original.copy()
    return b

def main():
    x = numpy.load("iris dataset/iris_features.npy")
    y = numpy.load("iris dataset/iris_labels.npy")
    N = 120
    x_train = x[:N]
    y_train = y[:N]
    x_test = x[N:]
    y_test = y[N:]
    pca = decomposition.PCA(n_components=4)
    pca.fit(x)
    print(pca.explained_variance_ratio_)
    start = 2
    nsets = 10
    nsamp = x_train.shape[0]
    newx = numpy.zeros((nsets*nsamp, x_train.shape[1]))
    newy = numpy.zeros(nsets*nsamp, dtype="uint8")
    for i in range(nsets):
        if(i==0):
            newx[0:nsamp,:] = x_train
            newy[0:nsamp] = y_train
        else:
            newx[(i*nsamp):(i*nsamp+nsamp),:] = generateData(pca, x_train, start)
            newy[(i*nsamp):(i*nsamp+nsamp)] = y_train
    idx = numpy.argsort(numpy.random.random(nsets*nsamp))
    newx = newx[idx]
    newy = newy[idx]
    numpy.save("iris dataset/iris_train_features_augmented.npy", newx)
    numpy.save("iris dataset/iris_train_labels_augmented.npy", newy)
    numpy.save("iris dataset/iris_test_features_augmented.npy", x_test)
    numpy.save("iris dataset/iris_test_labels_augmented.npy", y_test)


main()