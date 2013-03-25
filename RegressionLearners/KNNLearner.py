import numpy as np
import math, copy, numpy
import datetime as dt



class KNNLearner(object):

    def __init__(self, k=3):
        self.k = k

    def addEvidence(self, Xtrain, Ytrain):
        self.xdata = Xtrain
        self.ydata = Ytrain
        self.data = np.column_stack((self.xdata, self.ydata))
    
    def query(self, point):
        distances = list()
        for entry in self.xdata:
            distances.append(numpy.sqrt(numpy.sum((entry-point)**2)))
        sortmatrix = np.column_stack((self.xdata, np.array(distances)))
        sortmatrix = self.data[sortmatrix[:,2].argsort()]
        return np.mean(sortmatrix[0:self.k,2])

    def getName(self):
        print ("\n\n====KNN Learner: K = %s====" %(self.k))
        pass
            
def getflatcsv(fname):
    inf = open(fname)
    return numpy.array([map(float,s.strip().split(',')) for s in inf.readlines()])

def testgendata():
    fname = 'data-classification-prob.csv'
    querys = 1000
    data = getflatcsv(fname)
    xpoints = data[0:60,0:2]
    ypoints = data[0:60,2]
    learner = KNNLearner(k=3)
    learner.addEvidence(xpoints,ypoints)
    learner.query(data[61,0:2])

def test():
    testgendata()

if __name__=="__main__":
    test()
