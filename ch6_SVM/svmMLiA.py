from numpy import *
from optStruct import *
#SMO（序列最小优化）算法中的辅助函数
def loadDataSet(fileName):
    dataMat = [] ; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append((float(lineArr[2])))
    return dataMat, labelMat


#i是第一个alpha的下标，m是所有alpha的数目
def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j


#用于调整大于H或小于L的alpha值
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


#简化版SMO算法:
#dataMatIn是数据集， classLabels是类别标签， C是常数， toler是容错率，maxIter是取消当前最大的循环次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n =shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fxi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b #第i个样本的预测分类
            Ei = fxi - float(labelMat[i]) #误差
            if((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m) #随机选择第j个样本
                fxj = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) + b #第j个样本的预测分类
                Ej = fxj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C+alphas[j] -alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C )
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print('L == H'); continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T -dataMatrix[i, :]*dataMatrix[i, :].T - dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0 : print('eta >= 0'); continue
                alphas[j] -= labelMat[j] *(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if(abs(alphas[j] - alphaJold) < 0.00001):print('j not moving enough');continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold-alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T- labelMat[j] *(alphas[j] - alphaJold)*dataMatrix[i,:]*dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T- labelMat[j] *(alphas[j] - alphaJold)*dataMatrix[j,:]*dataMatrix[j, :].T
                if ( 0 < alphas[i]) and (C > alphas[i]) : b = b1
                elif (0 < alphas[j]) and (C > alphas[j]) : b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print('iter: %d i :%d, paris changed %d' %(iter, i , alphaPairsChanged))
        if(alphaPairsChanged == 0):iter += 1
        else:iter = 0
        print('iteration number: %d' % iter)
    return b, alphas


#完整版Platt SMO的外循环代码
def smoP(dataMatIn, classLabels, C, toler, maxIter, KTup=('lin', 0)):
    os = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet =True
    alphaPairsChanged = 0
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(os.m):
                alphaPairsChanged += optStruct.innerL(i, os)
            print('fullSet, iter: %d i: %d, pairs changed %d' %(iter, i, alphaPairsChanged ))
            iter += 1
        else:
            nonBoundIs = nonzero((os.alphas.A > 0) * (os.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += optStruct.innerL(i, os)
                print('non-bound iter:%d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet: entireSet =False
        elif (alphaPairsChanged == 0):entireSet = True
        print('iteration number: %d' % iter)
    return os.b, os.alphas

