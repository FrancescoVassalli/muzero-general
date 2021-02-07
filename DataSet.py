import pandas as pd
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
        base="results/AT"
        ext=".csv"
        start=1
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
            logName = 1
        else:
            logName+=1
        self.log[logName]=[]

    def write(self):
        self.log = pd.DataFrame.from_dict(self.log)
        self.log.to_csv(self.logCSVName)

    def logReturn(self,lastReturn,logName):
        self.log[logName].append(lastReturn)
