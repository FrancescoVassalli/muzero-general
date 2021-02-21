import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from os.path import exists

class DataSet:
    def __init__(self):
        path = "../ranos/DL30/XTotalAPPS.at"
        pathY = "../ranos/DL30/XTotalAPPS.price.at"
        df  = pd.read_csv(path,header=None)
        self.dfy  = pd.read_csv(pathY,header=None)
        self.dataSize = len(df.index)
        self.split=0.8
        self.pca = PCA(n_components=20,whiten=True)
        self.train_features = pd.DataFrame(self.pca.fit_transform(df[:int(self.split*self.dataSize)]))
        self.test_featues = pd.DataFrame(self.pca.transform(df[int(self.split*self.dataSize):]))
        self.log = dict()
        base="results/AT/formalTest"
        ext=".csv"
        start=1
        self.trueName = "True"
        self.randomName="Random"
        while exists(base+str(start)+ext):
            start +=1
        self.logCSVName = base+str(start)+ext


    def getFeatures(self,train):
        if train:
            return self.train_features
        else:
            return self.test_featues

    def getPrices(self,train):
        if train:
            return self.dfy.head(int(self.split*self.dataSize))
        else:
            return self.dfy.tail(self.dataSize-int(self.split*self.dataSize))

    def getSize(self,train):
        if train:
            return int(self.split*self.dataSize)
        else:
            return self.dataSize-int(self.split*self.dataSize)

    def getLogName(self,logName):
        if logName is None:
            return 0
        else:
            logName+=1
        self.log[self.trueName+str(logName)]=[]
        self.log[self.randomName+str(logName)]=[]
        return logName

    def write(self):
        lens = [ len(val) for _, val in self.log.items() ] 
        maxLen  = max(lens)
        for key, val in self.log.items():
            if len(val)!=maxLen:
                nanlist = [np.nan]*(maxLen-len(val))
                self.log[key].extend(nanlist)
        print(lens)
        self.log = pd.DataFrame.from_dict(self.log)
        print("Writing "+self.logCSVName)
        self.log.to_csv(self.logCSVName)

    def logReturn(self,lastReturn,logName,real):
        if real:
            self.log[self.trueName+str(logName)].append(lastReturn)
        else:
            self.log[self.randomName+str(logName)].append(lastReturn)
