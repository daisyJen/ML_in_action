import matplotlib
import matplotlib.pyplot as plt
import KNN
from numpy import *

fig = plt.figure()
ax = fig.add_subplot(111)
datingDataMat, datingLabels = KNN.file2matrix('datingTestSet.txt')
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()
