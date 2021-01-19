import pandas as pd
from sklearn.decomposition import PCA

class DataSet:
    def __init__(self):
        path = "../ranos/DL30/XTotalAPPS.at"
        pathY = "../ranos/DL30/XTotalAPPS.price.at"
        df  = pd.read_csv(path,header=None)
        self.dfy  = pd.read_csv(path,header=None)
        self.dfy = self.dfy.rename(columns={0:"Y"})
        self.dataSize = len(df.index)
        self.split=0.8
        self.pca = PCA(n_components=3,whiten=True)
        self.train_features = pd.DataFrame(self.pca.fit_transform(df[:int(self.split*self.dataSize)]))
        self.test_featues = self.pca.transform(df[int(self.split*self.dataSize):])


    def getFeatures(self,train):
        if train:
            return self.train_features
        else:
            return self.test_featues

    def getPrices(self,train):
        if train:
            return self.dfy["Y"][:int(self.split*self.dataSize)]
        else:
            return self.dfy["Y"][int(self.split*self.dataSize):]

    def getSize(self,train):
        if train:
            return int(self.split*self.dataSize)
        else:
            return self.dataSize-int(self.split*self.dataSize)
    
