from numpy import *
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so' , 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
            ]
    classVec = [0, 1, 0, 1, 0, 1] # 1 代表侮辱性文字，0 代表正常言论
    return postingList, classVec


#返回没有重复的单词
# ['mr', 'not', 'to', 'dalmation', 'steak', 'dog', 'quit',
# 'flea', 'has', 'my', 'cute', 'licks', 'park', 'stupid', 'please',
# 'garbage', 'him', 'so', 'stop', 'maybe', 'how', 'ate', 'worthless', 'is',
#  'problem', 'take', 'buying', 'help', 'food', 'posting', 'love', 'I']
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) #创建两个集合的并集,|操作符用于求两个集合的并集
    return list(vocabSet)

#该函数的输入参数为词汇表及某个文档，输出的是文档向量，向量的每一元素为1或0，分别表示词汇表中的单词在输入文档中是否出现
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList) #创建一个和词汇等长的向量，并将其元素都设置为0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word )
    return  returnVec


#朴素贝叶斯分类器训练函数
#文档矩阵trainMatrix，以及由每篇文档类别标签所构成的向量trainCategory
#p(1):侮辱性文档，p(0):非侮辱性文档
#trainMat:[[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]]
#trainCategory:[0, 1, 0, 1, 0, 1]
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix) #词汇表大小 6
    numWords = len(trainMatrix[0]) #32
    #sum(trainCategory):3
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Demo = 2.0 ; p1Demo = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Demo += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Demo += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Demo)
    p0Vect = log(p0Num/p0Demo)
    return p0Vect, p1Vect, pAbusive

#朴素贝叶斯分类函数
#4个输入：要分类的向量vec2Classify以及使用函数trainNB0()计算得到的三个概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) +log(pClass1)
    p0 = sum(vec2Classify * p0Vec) +log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))


testingNB()
# listOPosts, listClasses = loadDataSet()
# myVocabList = createVocabList(listOPosts)
# #print("myVocabList....%s" % myVocabList)
# #listOPosts[0]:['my', 'dog', 'has', 'flea', 'problem', 'help', 'please']
# #print(setOfWords2Vec(myVocabList, listOPosts[0]))
# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
#
# p0V, p1V, pAb = trainNB0(trainMat, listClasses)
# print('pAb....%s ' % pAb)
# print('p0V....%s ' % p0V)
#a=[1, 2, 3];b=[4, 5, 6];a.append(b);print(a) #[1, 2, 3, [4, 5, 6]]
#a=[1, 2, 3];a.extend(b);print(a) #[1, 2, 3, 4, 5, 6]
