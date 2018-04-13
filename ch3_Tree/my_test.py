import tree

myDat, labels = tree.createDataSet()
#result = tree.splitDataSet(myDat, 0 , 0)
result = tree.chooseBestFeatureToSplit(myDat)
print(result)
