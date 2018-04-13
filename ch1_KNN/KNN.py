# encoding:utf-8
from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#k-近邻算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #行，4
    #在行向量方向上重复inX共dataSetSize次，列向量方向上重复inX共1次
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort() #argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)#按照第二个元素降序排列
    return sortedClassCount[0][0]

#将文本记录到转换NumPy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

#归一化特征值:newValue = (oldValue-min)/(max-min)
def autoNorm(dataSet):
    minVals = dataSet.min(0) #minVals:[0.       0.       0.001156]
    maxVals = dataSet.max(0) #maxVals:[9.1273000e+04 2.0919349e+01 1.6955170e+00]
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet)) #按照dataSet的行列建立数组，只是里面的值全是0
    m = dataSet.shape[0] #行数1000
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

#分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0] #行数1000
    numTestVecs = int(m*hoRatio) #100,前100行为测试数据
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i:], normMat[numTestVecs:m, :],datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is %d" %(classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is:%f" %(errorCount/float(numTestVecs)))


#约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent filer miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    print("normMat: %s" % normMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probaly like this person:", resultList[classifierResult-1])


#图像转为测试向量
def img2Vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


#手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)#trainingDigits目录下的文件数，m=1934
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2Vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat,hwLabels , 3)
    print('the classifier came back with : %d' % errorCount)
    print('the total error rate is : %f' % (errorCount / float(mTest)))


handwritingClassTest()