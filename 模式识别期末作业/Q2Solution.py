# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 23:21:57 2016

@author: king
"""

import scipy.io as sio  
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import xlrd
import numpy as np
from Kmeans3D import *
fname = "Q22.xlsx"
bk = xlrd.open_workbook(fname)
sh = bk.sheet_by_name("Sheet1")
nrows = sh.nrows
#获取列数
ncols = sh.ncols
centroids = zeros((nrows-1,ncols-1))  
for i in range(1,nrows):
    for j in range(1,ncols):
        centroids[i-1,j-1]= sh.cell_value(i,j)
        
k = 2  
centerinit=np.array([[1,1,1],[-1,1,-1]])
centerinit=np.array([[1,1,1],[-1,1,-1]])
la,clusterAssment = kmeans(centroids, k,centerinit)

x,y,z = centroids[:,0],centroids[:,1],centroids[:,2]

ax=plt.subplot(111,projection='3d') #创建一个三维的绘图工程
 # draw all samples  
mark = ['or', '+r'] 
for i in xrange(nrows-1):  
    markIndex = int(clusterAssment[i, 0])  
    if markIndex==1:
        ax.scatter(centroids[i, 0], centroids[i, 1],centroids[i,2], marker = '+', color = 'c', label='2', s = 50)
    else:
        ax.scatter(centroids[i, 0], centroids[i, 1],centroids[i,2], marker = 'o', color = 'r', label='3', s = 15)
#将数据点分成三部分画，在颜色上有区分度

mark = ['Dr', 'Db'] 
for i in range(k):  
        ax.scatter(la[i, 0], la[i, 1],la[i,2], mark[i])  
  
ax.set_zlabel('Z') #坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()