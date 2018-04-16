from math import log
import operator
import treePlotter

def createDataSet():
    dataSet=[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

#计算给定数据集的香农熵
def calcSHannonEnt(dataSet):
    numEntries = len(dataSet) #5
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    #labelCounts:{'yes': 2, 'no': 3}
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt


#按照给定特征划分数据集:待划分的数据集、划分数据集的特征、特征的返回值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) -1 #2
    baseEntropy = calcSHannonEnt(dataSet) #整个数据集的原始香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] #[1, 1, 1, 0, 0]
        uniqueVals = set(featList) #{0, 1}
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcSHannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return  bestFeature

#
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCountp[vote] += 1
        sortedClassCount = sorted(classCount.items(), operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]


#创建树的函数代码
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet] #['yes', 'yes', 'no', 'no', 'no']
    if classList.count(classList[0]) == len(classList): #classList[0]的数量和整个数组相同，即所有的类标签完全相同
        return classList[0]
    #使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组
    if len(dataSet[0]) == 1:
        return  majorityCnt(classList)
    beatFeat = chooseBestFeatureToSplit(dataSet) #0
    beatFeatLabel = labels[beatFeat] #no surfacing
    myTree = {beatFeatLabel:{}}
    del(labels[beatFeat])
    featValues =  [example[beatFeat] for example in dataSet] #[1, 1, 1, 0, 0]
    uniqueVals =set(featValues) #{0,1}
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[beatFeatLabel][value] = createTree(splitDataSet(dataSet, beatFeat, value), subLabels)
    return myTree

#使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0] # no surfacing, flippers
    secondDict = inputTree[firstStr] #{0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}
    featIndex = featLabels.index(firstStr) #使用index方法查找当前列表中第一个匹配firstStr变量的元素 0, 1
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict:
                classLabel = classify(secondDict[key] ,  featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb+')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


# dataSet, labels = createDataSet()
# myTree = treePlotter.retrieveTree(0)
# print('myTree... %s' % myTree)
# classify(myTree, labels, [1,0] )
myTree = treePlotter.retrieveTree(0)
storeTree(myTree, 'classifierStorage.txt')
grabTree('classifierStorage.txt')

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses, lensesLabels)
treePlotter.createPlot(lensesTree)
