from sklearn.cluster import KMeans
import scipy.io as sio  
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import xlrd
from numpy import *
from Kmeans3D import *
fname = "Q2.xlsx"
bk = xlrd.open_workbook(fname)
sh = bk.sheet_by_name("Sheet1")
nrows = sh.nrows
#获取列数
ncols = sh.ncols
centroids = zeros((nrows,ncols))  
for i in range(nrows):
    for j in range(ncols):
        centroids[i,j]= sh.cell_value(i,j)
clf = KMeans(n_clusters=2)
s = clf.fit(centroids)
la=s.labels_
print s

    #20个中心点
print clf.cluster_centers_
    
    #每个样本所属的簇
print clf.labels_