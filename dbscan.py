#!/usr/bin/env python
from math import sqrt
import numpy as np
import glob,sys
#from sklearn import metrics
from sklearn.cluster import DBSCAN 
#from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getFiles(tlist):
    flist = sorted(glob.glob("../v5/gCoor_0-0-{0}-*-500000.dat".format(tlist)))
    return flist

def boxDim(n,den):
    """Calculate the box length from density and num particles"""
    l = (n/den)**(1./3.)
    return l

def shiftCoord(c,bl):
    """Shift coordinates by box length so all are positive."""
    return [[i + bl/2. for i in ci] for ci in c]

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

def numOut(label):
    return sum(1 for x in label if x == -1)

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
temp = ["1.00"]
for tm in temp:
    fl = getFiles(tm)
    data = []
    for f in fl:
        #split name and record density and temperature
        print f
        fs = f.split("-")
        tempf = float(fs[2])
        densf = float(fs[3])
        #Read Coordinate data from file
        co = readCoor(f)
        nAtom = len(co[0])
        nFr = len(co)
        blen = boxDim(nAtom,densf)
        cNumFr = []
        cNumCore = []
        notClstr = []
        nCore = []
        nLabels = []
        #Loop over all frames, run dbscan, and record cluster info for frame
        for fr in range(0,len(co),10):
            coordFr = co[fr]
            cShift = shiftCoord(coordFr,blen)
            tree = cKDTree(cShift,boxsize=blen)
            sdm = tree.sparse_distance_matrix(tree,50.,output_type="dok_matrix")
            #dbscan = DBSCAN(eps=1.5,min_samples=5).fit(cShift)
            #core = dbscan.core_sample_indices_
            #labels = dbscan.labels_
            dbscan = DBSCAN(eps=1.5,min_samples=5,metric="precomputed").fit(sdm)
            core = dbscan.core_sample_indices_
            labels = dbscan.labels_
            nCore.append(sum(1 for c in core if c > -1))
            nLabels.append(sum(1 for l in labels if l > -1))
            #cavg = findcenters(coordFr,labels)
            num = numClstr(labels)
            numCore = numClstr(core)
            cNumFr.append(len(num)-1)
            cNumCore.append(len(numCore)-1)
            nOut = numOut(labels)
            notClstr.append(nOut)
            #print fr,num,len(num)-1,numCore,len(numCore)-1
        notClstrP = [float(n)/float(nAtom) for n in notClstr]
        #print nCore
        data.append([tempf,densf,np.mean(nCore),np.std(nCore,dtype=np.float32)/sqrt(len(co))])
        #plt.hist(cNumFr,bins=range(13),normed=1)
        #plt.show()
    np.savetxt("cluster.out",data,delimiter=",",fmt="%.5f")
    #plt.hist(notClstr,bins=range(800,950,10),normed=1)
    #plt.show()
    #Plot figure of clusters
    #plotClstrFrame(coordFr,labels,cavg)

