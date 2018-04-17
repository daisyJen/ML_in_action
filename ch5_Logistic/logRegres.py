from numpy import *
#便利函数，主要功能是打开文件testSet.txt并逐行读取，每行前两个值分别是X1和X2，第三个是数据对应的类别标签
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

#sigmoid函数
def sigmoid(inX):
    return longfloat(1.0/(1+exp(-inX)))

#Logistic回归梯度上升优化算法：返回回归系数
# dataMatIn是一个2维NumPy数组，每列分别代表每个不同的特征，每行代表每个训练样本，dataMatrix是100*3的矩阵
#classLabels是一个类别标签，是一个1*100的行向量
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose() #transpose()转置
    m, n = shape(dataMatrix) #shape(dataMatrix) 返回矩阵的行和列
    alpha = 0.001 #alpha向目标移动的步长
    maxCycles = 500 #maxCycles是迭代次数
    weights = ones((n, 1)) #ones((n, 1))构造n行1列
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat -h)
        weights = weights + alpha * dataMatrix.transpose()* error
    return weights


#画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(wei):
    import  matplotlib.pyplot as plt
    weights = wei.getA() #.getA()是array的子功能，将matrix转化为array
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [] ; ycord1 = []
    xcord2 = []; ycord2 =[]
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)  #arange()类似于内置函数range()，通过指定开始值、终值和步长创建表示等差数列的一维数组，注意得到的结果数组不包含终值
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


#随机梯度上升算法
def stoGradAscent0(dataMatrix, classLabels):
    dataMatrix = array(dataMatrix)
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n) #ones(n)表示n列的行向量
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    weights = mat(weights).reshape((3, 1))
    return weights


#改进的随机梯度上升算法
def stoGradAscent1(dataMatrix, classLabels, numIter = 150):
    dataMatrix = array(dataMatrix)
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m)) #dataIndex=[0, 1, 2, 3,……, m-1]
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))#随机产生1到len(dataIndex)-1之间的数字
            h = sigmoid(sum(dataMatrix[randIndex] * weights)) #随机选取更新
            error = classLabels[randIndex] -h
            weights = weights + alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    # weights = mat(weights).reshape((3, 1))
    return weights


#Logistic回归分类函数:以回归系数和特征向量作为输入来计算对应的Sigmoid值
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stoGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print ('the error rate of this test is:%f' % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is:%f' %(numTests, errorSum/float(numTests)))