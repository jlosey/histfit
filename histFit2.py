#!/usr/bin/env python
import glob
import numpy as np
from scipy.stats import normaltest
from scipy.integrate import quad 
import matplotlib.pylab as plt
import sys

fig1=plt.figure(1,figsize=(10,6))
#fig2=plt.figure(2,figsize=(10,6))
#fig3=plt.figure(3,figsize=(10,6))
if len(sys.argv) > 1:
	threshold = int(sys.argv[1]) 
else:
	threshold = 4
denslabel = []
plabel = []
tlabel = []
tindex = [2,4,6,12,13,14,14,15,16,16,16,16,15,14,13,12,2]
dlist = ["0.02","0.05","0.10","0.15","0.20","0.25","0.30","0.316","0.35","0.40","0.45","0.50","0.55","0.60","0.65","0.70","0.75"]
#dlist = ["0.05", "0.10","0.15","0.20","0.25"]
for d,i in zip(dlist,tindex):
	flist = sorted(glob.glob("../v5/gClst_0-0-*-{0}-500000.dat".format(d)))
	#flist = sorted(glob.glob("../v4/gClst_5-0-*-{0}-250000.dat".format(d)))
	areaList = []
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

		for line in f:
			li=line.strip()
			if not li.startswith("#") and not li.startswith("5000000"):
				n.append(int(li.split()[0]))
				sCount.append(int(li.split()[1]))
				pSn.append(float(li.split()[2]))
				clstCount.append(int(li.split()[3]))
				pClst.append(float(li.split()[4]))
		f.close()

		#Cluster of neighbors with > 4 Stillinger nieghbors
		dataS = np.asarray(pSn)
		dataC = np.asarray(pClst)
		dataC2 = dataC[1:]
		dataSL = dataS[threshold:]
		dataCL = dataC[threshold:]
		indices = np.arange(dataC.size)
		cSum = dataC.sum
		cAvg = np.multiply(indices,dataC).sum() 
		cMax = dataC.argmax()
		c2Sum = dataC2.sum()
		sLSum = dataSL.sum()
		cLSum = dataCL.sum()
		areaList.append((temp,dens,sLSum,cLSum,cAvg,cMax))

	areaList = np.asarray(areaList)
	ind = np.lexsort((areaList[:,0],areaList[:,1]))
	areaList = areaList[ind]
	plabel.append(areaList[i,3])
	tlabel.append(areaList[i,0])
	#plt.plot(areaList[:,0],areaList[:,6])
	#ax = fig.add_subplot(1,2,1)
	#ax.set_xlabel("T")
	#ax.set_ylabel("P[n > {0}]".format(threshold))
	#ax.set_ylim([0,1])
	#plt.plot(areaList[:,0],areaList[:,2], '.-', label="{0}".format(dens))
	denslabel.append(areaList[0,1])
	ax1 = fig1.add_subplot(1,1,1)
	ax1.set_ylabel("T")
	ax1.set_xlabel("P[n > {0}]".format(threshold))
	ax1.set_xlim([0,1])
	ax1.plot(areaList[:,3],areaList[:,0], '+-', label="{0}".format(dens))
	
	#ax1.legend(areaList[0,1])
	#print areaList
	#ax2 = fig2.add_subplot(1,1,1)
	#ax2.set_xlabel("T")
	#ax2.set_ylabel("Average Neighbours, <N>")
	#ax2.plot(areaList[:,0],areaList[:,4], '+-', label="{0}".format(dens))
	#ax2.legend(areaList[:,1])
	#ax3 = fig3.add_subplot(1,1,1)
	#ax3.set_xlabel("T")
	#ax3.set_ylabel("Mode of Neighbours, Max(N)")
	#ax3.plot(areaList[:,0],areaList[:,5], '+-', label="{0}".format(dens))
	#ax2.legend(areaList[:,1])
yt = ax1.get_yticks()
yt = np.append(yt,1.312)
ytl = yt.tolist()
ytl[-1] = "$T_c$"
#ax3.plot((1.312,1.312),(0,1),'k--',linewidth=0.7)
#ax2.plot((1.312,1.312),(0,1),'k--',linewidth=0.7)
ax1.plot((0,1),(1.312,1.312),'k--',linewidth=0.7)
#ax1.legend(denslabel,loc="right")
plt.yticks(yt,ytl)
for d,p,t in zip(denslabel,plabel,tlabel):
	plt.annotate(d,xy=(p,t),xytext=(20,20),textcoords='offset points', ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc="white",alpha=0.5),arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
#plt.annotate("Test",xy=(0.5,2),xytext=(0,20),textcoords='offset points', ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.5', alpha=0.5),arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
#fig.xlabel("T")
#fig.ylabel("P[n > {0}]")
#plt.legend(loc="lower left")
plt.show()
#fig2.show()
#fig3.show()
raw_input()
#plt.savefig("P4.png")
#fig2.savfig("AvgN.png")
#fig3.savfig("MaxN.png")
