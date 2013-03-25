from knnlearner import KNNLearner
from os.path import expanduser
from LinRegLearner import LinRegLearner
from RandomForestLearner import RandomForestLearner
import numpy as np
import math, copy, numpy, time
import matplotlib.pyplot as plt
from pylab import *


def getflatcsv(fname):
    inf = open(fname)
    return numpy.array([map(float,s.strip().split(',')) for s in inf.readlines()])

def testgendata(fname):

    data = getflatcsv(fname)
    print data.shape
    test_length = math.ceil(.6*data.shape[0])
    learners = list()

    xpoints = data[0:test_length,0:2]
    ypoints = data[0:test_length,2]

    Xtestset = data[test_length:,0:2]
    Ytestset = data[test_length:,2]

    knn_learner = KNNLearner(k=3)
    lin_learner = LinRegLearner()
    rf_learner = RandomForestLearner(k=63)
    learners.append(rf_learner)
    learners.append(knn_learner)
    learners.append(lin_learner)

    for learner in learners:

        learner.getName()
        stime = time.time()
        learner.addEvidence(xpoints,ypoints)
        etime = time.time()
    
        print ("Average Training time per Instance: %s" %((etime-stime)/float(test_length)))
        results = list() 
        set_length = Xtestset.shape[0]
        stime = time.time()
        for point in Xtestset: 
            y = learner.query(point)
            results.append(y)
#            print y
        etime = time.time()
        print ("Average Time per Instance: %s" %((etime-stime)/float(set_length)))

        RMSE =  numpy.sqrt(numpy.mean((np.array(results) - Ytestset)**2))
        print ("RMSE: %s" %(RMSE))

        corrcoeff = np.corrcoef(np.array(results), Ytestset)
        print ("Correlation Coefficient: %s" %(corrcoeff[0,1]))

    corrcos = list()
    print set_length 
 
    set_length = 65
    for num in range(30,set_length):
        print num
        results = list()
        learner = RandomForestLearner(k=num)
        learner.addEvidence(xpoints,ypoints)
        for point in Xtestset:
            reac = learner.query(point)
            results.append(reac)
        corrcos.append(np.corrcoef(np.array(results), Ytestset)[0,1])
       

    corr_array = np.array(zip(np.array(corrcos), range(30,set_length)))
    print corr_array
    print "Optimal K (by Correlation Coefficient: %s" %(corr_array[corr_array[:,0].argsort()][-1])

    plt.clf()
    plt.plot(range(30,set_length), np.array(corrcos))
    title = "%s.pdf" %(fname)
 
    savefig(title, format='pdf')
#    for result, true_value in zip(results, Ytestset):
#        rmse_sum += distances.append(numpy.sqrt(numpy.sum((entry-point)**2))) 
    
home = expanduser('~')
testgendata("%s/QSTK/Examples/KNN/data-classification-prob.csv" %(home))
testgendata("%s/QSTK/Examples/KNN/data-ripple-prob.csv" %(home))
