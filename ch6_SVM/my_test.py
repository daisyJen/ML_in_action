import svmMLiA
from numpy import *
dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
b, alphas = svmMLiA.smoP(dataArr, labelArr , 0.6, 0.001, 40)
# print('b........%s' % b)
# print('alphas........%d, %d' % shape(alphas[alphas>0]))
for i in range(100):
    if alphas[i] > 0.0 : print(dataArr[i] , labelArr[i])