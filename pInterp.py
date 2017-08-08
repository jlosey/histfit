#!/usr/bin/env python
import glob
import numpy as np
from scipy.interpolate import interp2d
from scipy.interpolate import griddata 
import matplotlib.pylab as plt
import sys

if len(sys.argv) > 1:
	threshold = int(sys.argv[1]) 
else:
	threshold = 4
dataL = []
dlist = ["0.20","0.25","0.30","0.316","0.35","0.40","0.45","0.50","0.55","0.60","0.65","0.70","0.75"]
for d in dlist:
	flist = sorted(glob.glob("../v5/gClst_0-0-*-{0}-500000.dat".format(d)))
	#Loop through files in flist
	for fl in flist:
		n = []
		sCount = []
		pSn = []
		clstCount = []
		pClst = []
		fs = fl.split("-")
		temp = float(fs[2])
		dens = float(fs[3])
		f = open(fl,'r')
		#Loop through lines in file
		for line in f:
			li=line.strip()
			if not li.startswith("#") and not li.startswith("5000000"):
				n.append(int(li.split()[0]))
				sCount.append(int(li.split()[1]))
				pSn.append(float(li.split()[2]))
				clstCount.append(int(li.split()[3]))
				pClst.append(float(li.split()[4]))
		f.close()
		dataS = np.asarray(pSn)
		dataC = np.asarray(pClst)
		dataSL = dataS[threshold:]
		dataCL = dataC[threshold:]
		cSum = dataC.sum()
		sLSum = dataSL.sum()
		cLSum = dataCL.sum()
		dataL.append((temp,dens,sLSum,cLSum,cSum))
#Convert data list into numpy array
dataL = np.asarray(dataL)
ind = np.lexsort((dataL[:,0],dataL[:,1]))
dataL = dataL[ind]
points = (dataL[:,0],dataL[:,1])
#print dataL
dG = np.linspace(0.2,0.7,225)
tG = np.linspace(0.5,4.0,225)
T,D = np.meshgrid(tG,dG)
#Interpolate data
gClst = griddata(points,dataL[:,3],(T,D),method="cubic")
#iClst = interp2d(dataL[:,0],dataL[:,1],dataL[:,3],kind="linear")
#I = iClst(tG,dG)
#print I
#plt.contourf(tG,dG,I)

#Read in LJ vapour-liq data
tLJ = []
rvLJ = []
rlLJ = []
ljf = open("LJCoex.dat","r")
for line in ljf.readlines():
	li = line.strip()
	lis = li.split("\t")
	tLJ.append(float(lis[0]))
	rvLJ.append(float(lis[1]))
	rlLJ.append(float(lis[2]))
ljf.close()
#print tLJ,rvLJ
cLevels = [0.2,0.4,0.6,0.8]
CS = plt.contour(D,T,gClst,levels=cLevels)
plt.plot(rvLJ,tLJ,"-g",rlLJ,tLJ,"-k")
plt.clabel(CS,inline=1,fmt="%1.2f")
plt.ylabel("T*")
plt.xlabel("Density*")
plt.title("P[N>4], Interpolated")
plt.show()
