from numpy import *
import svmMLiA

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2))) # m*2的矩阵


    #误差缓存
    def calcEk(os, k):
        fxk = float(multiply(os.alphas, os.labelMat).T * (os.X * os.X[k, :].T)) + os.b # .T是转置
        Ek = fxk - float(os.labelMat[k])
        return Ek


    #用于选择合适的第二个alpha值或者内循环中的启发式方法
    def selectJ(i, os, Ei):
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        os.eCache[i] = [1, Ei]
        #python中一个matrix矩阵名.A 代表将 矩阵转化为array数组类型
        validEcacheList = nonzero(os.eCache[:, 0].A)[0] #nonzero(os.eCache[:, 0].A)[0]构建出一个非零表
        if (len(validEcacheList)) > 1:
            for k in validEcacheList:
                if k == i : continue
                Ek = optStruct.calcEk(os, k)
                deltaE = abs(Ei - Ek)
                if(deltaE > maxDeltaE):
                    maxK = k
                    maxDeltaE =deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = svmMLiA.selectJrand(i, os.m)
            Ej = optStruct.calcEk(os, j)
        return j, Ej

    #计算误差值并存入缓存中
    def updateEk(os, k):
        Ek = optStruct.calcEk(os, k)
        os.eCache[k] = [1, Ek]

    # 完整Platt SMO算法中的优化例程 :
    def innerL(i, os):
        Ei = optStruct.calcEk(os, i)
        if ((os.labelMat[i] * Ei < -os.tol) and (os.alphas[i] < os.C)) or ((os.labelMat[i] * Ei > os.tol) and (os.alphas[i] > 0)):
            j, Ej = optStruct.selectJ(i, os, Ei)
            alphaIold = os.alphas[i].copy()
            alphaJold = os.alphas[j].copy()
            if(os.labelMat[i] != os.alphas[j]):
                L = max(0, os.alphas[j] - os.alphas[i])
                H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
            else:
                L = max(0, os.alphas[j] + os.alphas[i] - os.C)
                H = min(os.C, os.C + os.alphas[j] + os.alphas[i])
            if L == H:print('L == H');reutrn (0)
            eta = 2.0 *os.X[i, :] * os.X[j, :].T - os.X[i, :]* os.X[i, :].T - os.X[j, :] * os.X[j, :].T
            if eta >= 0:print(eta >= 0);return(0)
            os.alphas[j] -= os.labelMat[j] * (Ei - Ej )/eta
            os.alphas[j] -= svmMLiA.clipAlpha(os.alphas[j], H, L)
            optStruct.updateEk(os, j)
            if(abs(os.alphas[j] - alphaJold) < 0.00001):
                print('j not moving enough'); return(0)
            os.alphas[i] += os.labelMat[j] * os.labelMat[i] *(alphaJold - os.alphas[j])
            updateEk(os, i)
            b1 = os.b - Ej - os.labelMat[i] * (os.alphas[i] - alphaIold) * os.X[i, :] * os.X[i, :].T -os.labelMat[j] * (os.alphas[j] - alphaJold) * os.X[i, :]*os.X[j,:].T
            b2 = os.b - Ej - os.labelMat[i] * (os.alphas[i] - alphaIold) * os.X[i, :] * os.X[j, :].T - os.labelMat[
                j] * (os.alphas[j] - alphaJold) * os.X[j, :] * os.X[j, :].T
            if( 0 < os.alphas[i]) and (os.C > os.alphas[i]) :os.b = b1
            elif(o < os.alphas[j]) and (os.C > os.alphas[j]): os.b = b2
            else:os.b =(b1 + b2)/2
            return 1
        else:
            return 0
