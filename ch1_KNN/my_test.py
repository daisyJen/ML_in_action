from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1, 1.0], [1.0, 1.0, 1.0], [0, 0, 1.0], [0, 0.1, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#k-近邻算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #行，4
    #在行向量方向上重复inX共dataSetSize次，列向量方向上重复inX共1次
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)#axis=1是列相加
    distances = sqDistances**0.5
    #print("distances : %s" % distances)
    sortedDistIndicies = distances.argsort()
    #print("sortedDistIndicies : %s" % sortedDistIndicies)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #print("i: %d, voteIlabel: %s" % (i, voteIlabel))
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        #print("classCount[voteIlabel]: %s" % (classCount[voteIlabel]))
        #print("classCount: %s" % (classCount))
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #print("sortedClassCount: %s" % (sortedClassCount))
    return sortedClassCount[0][0]



#将文本记录到转换NumPy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3)) #建numberOfLines行，3列的数组
    #print("returnMat:%s" % returnMat)
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
    #print("returnMat：%s" % returnMat)
    return returnMat, classLabelVector



#归一化特征值:newValue = (oldValue-min)/(max-min)
def autoNorm(dataSet):
    minVals = dataSet.min(0) #minVals:[0.       0.       0.001156]
    #print("minVals:%s" % minVals)
    maxVals = dataSet.max(0) #maxVals:[9.1273000e+04 2.0919349e+01 1.6955170e+00]
    #print("maxVals:%s" % maxVals)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet)) #按照dataSet的行列建立数组，只是里面的值全是0
    # print("normDataSet： %s " % normDataSet)
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals



#分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0] #行数1000
    print("m %s" % m)
    numTestVecs = int(m*hoRatio) #100
    print("numTestVecs %s" % numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is %d" %(classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is:%f" %(errorCount/float(numTestVecs)))


group, labels = createDataSet()
classify0([0, 0, 0], group, labels, 3)
datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
normMat, ranges, minVals = autoNorm(datingDataMat)
datingClassTest()


