#!/usr/bin/env python
# coding:utf-8
'''''
Created on 2014年11月24日
@author: zhaohf
'''
from sklearn import svm
from numpy import genfromtxt, savetxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
def main():
    dataset = pd.read_csv("train.csv").values[1:]
    test = pd.read_csv("test.csv").values[1:]
    scaler = StandardScaler().fit(test)
    test = scaler.transform(test)
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    svc = svm.SVC(probability=True)
    svc.fit(train, target)
    predicted_probs = [[index+1,x[1]] for index,x in enumerate(svc.predict_proba(test))]
    savetxt('svm_benchmark.csv',predicted_probs,delimiter=',',fmt='%d,%f',header='MoleculeId,PredictedProbability',comments='')

if __name__ == '__main__':
    main()