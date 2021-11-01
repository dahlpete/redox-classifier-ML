#!/usr/bin/env python3.6

"""
Created on Mon May 20 2019
Last Edited: June 13,2019

@author: alex chang & Jessica Freeze & Malika Uteuliyeva
modified by Peter Dahl for calculation of vertical displacement energies
"""

import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras.callbacks
from keras.callbacks import Callback
import sklearn.metrics as sklm
from sklearn import linear_model
import scipy
from sklearn.feature_selection import RFE
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV


#for normal data change to 11 and 13, 8 and 10 for lasso parameters
testlen = 0.10
#trainlen = int(round(0.75*nfrms))
npcs = 171
#Load the training data: e.g. 30/150 molecules
#for normal data remove lasparams
trainfile = np.loadtxt("pca_proj_impt_frames.txt", delimiter = " ")
labelfile = np.loadtxt("energy_gaps.txt", delimiter = "\t")
nfrms = np.shape(trainfile)[0]


trainlen = int(round(0.90*nfrms))

trainset = trainfile[0:trainlen,1:npcs+1]
trainLabels = labelfile[0:trainlen,1]
#Load the test data
#testfile = np.loadtxt("HammettNNTest.csv", delimiter = ",")
testset = trainfile[trainlen:,1:npcs+1]
testLabels = labelfile[trainlen:,1]

#First column of each trainset is the name of the molecule
mollength = nfrms-trainlen
moleculeID = np.reshape(labelfile[trainlen:,0], (nfrms-trainlen,1))
moleculeID = moleculeID.astype(int)
#collects the test results of individual runs
testIndividPreds = [[None]*3]*nfrms 
#collects the average test results
testAverage = [[None]*2]*nfrms

#print(np.shape(trainset),np.shape(testset),np.shape(trainLabels),np.shape(testLabels),np.shape(moleculeID))


def ml(trainset, testset, trainLabels, testLabels, moleculeID):
    global testAverage, testIndividPreds
    #training set with labels
    trainFeatures = trainset
    #trainLabels = trainset[:,-2:]
    #testing set with labels
    X = testset

    model = Sequential()
    model.add(Dense(25, input_dim = npcs, activation = 'tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(30, activation = 'relu'))
    model.add(Dense(15, activation = 'sigmoid'))
    model.add(Dense(25, activation = 'tanh'))
    model.add(Dense(10, activation = 'sigmoid'))
    model.add(Dense(1, activation = 'linear'))
    model.compile(loss = 'mean_absolute_error', optimizer = 'adam')
    #10% of training set used for validation; first column of trainFeatures was molecule name so it's excluded
    #trainFeatures without labels
    model.fit(trainFeatures, trainLabels, epochs = 220, batch_size = 5, validation_split = .15, verbose = 1)
    #predicts energies
    #X without labels
    testPrediction = model.predict(X)
    
    if (np.any(testAverage)==True):
        testAverage = np.append(testAverage, testPrediction,axis = 1)
    else:
        testAverage = testPrediction
        

    print(np.shape(moleculeID), np.shape(testPrediction))
    #add the molecule names
    testPrediction2 = np.concatenate([moleculeID, testPrediction], axis = 1)
    #the individual runs have the molecule names
    if (np.any(testIndividPreds)==True):
        testIndividPreds = np.append(testIndividPreds, testPrediction2,axis = 1)
    else:
        testIndividPreds = testPrediction2


i = 0
#train/test 30 times
n = 5
print('\n ML')
while i < n:
    #ml(trainset, trainfile[:,1:], trainLabels, testLabels, moleculeID)
    ml(trainset, testset, trainLabels, testLabels, moleculeID)
    i += 1

#print(testIndividPreds[0:10])

#testIndividPreds = testIndividPreds.reshape((n,mollength,3))

#the average of all runs is taken
#testAverage = testAverage.reshape((n,mollength,2))

mltestmean = np.mean(testAverage, axis = 1)
mltestmean = mltestmean.reshape((np.shape(mltestmean)[0],1))
#the names of the molecules are added on (note that same order is kept)
testLabels = testLabels.reshape((np.shape(testLabels)[0],1))
mltest = np.append(moleculeID, mltestmean, axis = 1)
mltest = np.append(mltest, testLabels, axis = 1)
#standard deviation
mlteststd = np.std(testAverage, axis = 1)
mlteststd = mlteststd.reshape((np.shape(mlteststd)[0],1))
mlteststd = np.append(moleculeID, mlteststd, axis = 1)

#prints molecules sorted by label
#mean square error
#print(testIndividPreds)
#print(mltest)
fileOUT = open('ml_results.txt','w')
print('frame	predicted VDE (eV)	computed VDE (eV)',file=fileOUT)
for i in range(len(mltest)):
	print('%s	      %.4f		    %.4f' % (int(mltest[i,0]),mltest[i,1],mltest[i,2]),file=fileOUT)

mse = np.sum(np.square(testLabels-mltestmean),axis=0)/(nfrms-trainlen)
mae = np.sum(testLabels-mltestmean)/(nfrms-trainlen)
print('\nmean squared error = %.4f' % mse)
print('\nmean absolute error = %.4f' % mae)
