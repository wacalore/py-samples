import numpy as np
import math, copy, numpy
import datetime as dt


class LinRegLearner(object):

    def __init__(self):
        pass

    def addEvidence(self, Xtrain, Ytrain):
        self.xdata = Xtrain
        self.ydata = Ytrain
        self.data = np.column_stack((self.xdata, self.ydata))
        self.factors = Xtrain.shape[1]

        xi = np.arange(0,9)
        A = np.column_stack((self.xdata, np.ones(self.xdata.shape[0])))
        w = np.linalg.lstsq(A,self.ydata)[0]
        self.w = w

    def query(self, point):
        result = 0.0
        for ix in range(len(point)):
            result += self.w[ix]*point[ix]
        result += self.w[self.factors]
        return result
   
    def getName(self):
        print '\n\n====Linear Regression Learner===='
        return

def getflatcsv(fname):
    inf = open(fname)
    return numpy.array([map(float,s.strip().split(',')) for s in inf.readlines()])

def testgendata():
    fname = 'data-classification-prob.csv'
    querys = 1000
    data = getflatcsv(fname)
    xpoints = data[0:60,0:2]
    ypoints = data[0:60,2]
    learner = LinRegLearner()
    learner.addEvidence(xpoints,ypoints)
    Y = learner.query(data[61,0:2])

def test():
    testgendata()

if __name__=="__main__":
    test()
