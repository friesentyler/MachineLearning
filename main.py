'''import numpy

a = numpy.array([1,2,3,4,5,6]).reshape(2,3)
b = numpy.array([1,2,3,4,5,6]).reshape(3,2)

d = numpy.dot(a, b)
a = a[:,1:]
b = b[1:]
c = a*b

print(a)
print(b)
print()
print(c)
print()
print(d)

z = numpy.load("file.npy")
print(z)'''
import numpy
from PIL import Image
from sklearn.datasets import load_sample_images
china = load_sample_images().images[0]
#china = numpy.array(china)
china = china.convert("L")
china.show()
