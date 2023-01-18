import numpy
import matplotlib.pyplot

# strip the newline character and read it into the lines list if not blank
with open("wdbc.data") as f:
    lines = [i[:-1] for i in f.readlines() if i != ""]

# create a list n to classify malignant vs benign and create a numpy array with benign and malignant converted to number
n = ["B", "M"]
x = numpy.array([n.index(i.split(",")[1]) for i in lines], dtype="uint8")
# create a 2d numpy array containing all the features of the tumors
y = numpy.array([[float(j) for j in i.split(",")[2:]] for i in lines])
# randomize the data set
i = numpy.argsort(numpy.random.random(x.shape[0]))
x = x[i]
y = y[i]

# standardize the features of the data set
z = (y - y.mean(axis=0)) / y.std(axis=0)

# save standardized and non-standardized data sets and produce a boxplot of the data
numpy.save("breast cancer dataset/bc_features.npy", y)
numpy.save("breast cancer dataset/bc_features_standard.npy", z)
numpy.save("breast cancer dataset/bc_labels.npy", x)
matplotlib.pyplot.boxplot(z)
matplotlib.pyplot.show()