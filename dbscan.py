#!/usr/bin/env python
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN 
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def readCoor(fname):
    """Read in coordinate data for clustering."""
    frame = []
    sim = []
    count = 0
    f = open(fname,'r')
    for line in f:
        li=line.strip()
        if not li.startswith("i") and not li.startswith("Step"):
            xi = li.split()
            if int(xi[0]) == 0 and count>2:
                sim.append(frame)
                frame = []
            frame.append([float(xi[1]),float(xi[2]),float(xi[3])])
            #print sim
        else:
            count = count+1
            continue
    f.close()
    coord = np.asarray(sim)
    return coord

def findcenters(c0,lab):
    """Find average of x,y,z coords for each member of cluster."""
    xavg = []
    yavg = []
    zavg = []
    cavg = []
    #print lab[np.where(lab > -1)]
    nC = len(np.unique(lab[np.where(lab>-1)]))
    #for d in range(3):
    for n in range(nC):
        ind = np.where(lab == n)
        xavg.append(np.average(c0[ind,0]))
        yavg.append(np.average(c0[ind,1]))
        zavg.append(np.average(c0[ind,2]))
    cavg.append(xavg)
    cavg.append(yavg)
    cavg.append(zavg)
    return cavg

def numClstr(label):
    n = len(np.unique(label))-1
    c = []
    for n in range(-1,n):
	    c.append([n, sum(1 for x in label if x==n)])
    return c

def plotClstrFrame(cFr,lab,cen):
    """Plot single frame of clusters."""
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    nClst = len(np.unique(lab[np.where(lab>-1)]))
    #for d in range(3):
    for n in range(nClst):
        indi = np.where(lab == n)
        ax.scatter(cFr[indi,0], cFr[indi, 1], cFr[indi, 2], c="red",s=50)
    ax.scatter(cen[0],cen[1],cen[2], c="cyan", s=200)
    plt.show()

#Run DBSCAN clustering algorithm for first dump
#for n in range(2,18):
    #scr = []
#	dbscan = DBSCAN(eps=1.5,min_samples=n).fit(coordFr)
#	labels = dbscan.labels_ 
#	if sum(labels) <> 0 and sum(labels) <> (len(labels)*-1):
#		scoreS = metrics.silhouette_score(coordFr,labels,metric='euclidean')
#		scoreCH =  metrics.calinski_harabaz_score(coordFr,labels)
#		neighb.append(n)
#		scr.append([n,scoreS, scoreCH])
#scr = np.asarray(scr)
#print(scr)
co = readCoor('../v5/gCoor_0-0-1.20-0.60-500000.dat')
#co = readCoor('../v5/gCoor_0-0-1.00-0.20-500000.dat')
cNumFr = []
cNumCore = []
notClstr = []
for fr in range(0,len(co)):
    coordFr = co[fr]
    #print coord[1000]
    neighb = []

    dbscan = DBSCAN(eps=1.5,min_samples=5).fit(coordFr)
    core = dbscan.core_sample_indices_
    labels = dbscan.labels_
    cavg = findcenters(coordFr,labels)
    num = numClstr(labels)
    numCore = numClstr(core)
    cNumFr.append(len(num)-1)
    cNumCore.append(len(numCore)-1)
    notClstr.append(num[0][1])
    #print fr,num,len(num)-1,numCore,len(numCore)-1
print "Cluster and not",cNumFr,notClstr
#print "Labels",labels,cNot
#print np.mean(cNumFr),np.std(cNumFr),np.mean(notClstr),np.std(notClstr)
h1,b1 = np.histogram(cNumFr)
h2,b2 = np.histogram(notClstr)
center = (b1[:-1] + b1[1:])/2
#print h1,h2
#plt.bar(center,h,align='center',width=1)
plt.hist(cNumFr,bins=b1)
plt.show()
plt.hist(notClstr,bins=b2)
plt.show()
#coordFr = coordFr
#score = DBSCAN(n_clusters=200, random_state=0).score(coord[:])
#print(kmeans.cluster_centers_)
#print(kmeans2.cluster_centers_)
#Plot figure of clusters
plotClstrFrame(coordFr,labels,cavg)
#print labels

