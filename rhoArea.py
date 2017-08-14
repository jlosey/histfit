#!/usr/bin/env python
import glob
import numpy as np
from scipy.stats import normaltest
from scipy.integrate import quad 
import matplotlib.pylab as plt
import sys

fig=plt.figure(1,figsize=(10,6))
count = 1
if len(sys.argv) > 1:
	threshold = int(sys.argv[1]) 
else:
	threshold = 4
tlist = ["0.50","0.60","0.70","0.80","0.90","1.00","1.10","1.20","1.30","1.312","1.50","2.00","4.00","4.50"]
tlabel = []
dindex = [9,5,3,2,2,2,3,4,7,7,4,6,5,-1]
print len(tlist),len(dindex)
#tlist = ["1.00","1.312","1.3365","2.00","4.00","6.00","8.00","10.00","12.00"]
#dlist = ["0.25","0.27","0.29","0.32","0.32","0.34","0.37","0.40"]
#dlist = ["0.20","0.32","0.40","0.60"]
#dlist = ["0.05", "0.10","0.15","0.20","0.25"]
for t,i in zip(tlist,dindex):
	flist = sorted(glob.glob("../v5/gClst_0-0-{0}-*-500000.dat".format(t)))
	#flist = sorted(glob.glob("../v4/gClst_0-0-*-{0}-*.dat".format(d)))
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

		#Stillinger fit
		dataS = np.asarray(pSn)

		#Cluster of neighbors with > 4 Stillinger nieghbors
		dataC = np.asarray(pClst)
		dataC2 = dataC[1:]
		dataSL = dataS[threshold:]
		dataCL = dataC[threshold:]
		indices = np.arange(dataC.size)
		cSum = dataC.sum
		cAvg = np.multiply(indices,dataC).sum() 
		cMax = dataC.argmax()
		#print temp,dens,cMax,cAvg
		#c2Sum = sum(dataC2)
		sSum = dataSL.sum()
		cLSum = dataCL.sum()
		areaList.append((temp,dens,sSum,cLSum,cAvg,cMax))

	areaList = np.asarray(areaList)
	ind = np.lexsort((areaList[:,0],areaList[:,1]))
	areaList = areaList[ind]
	#print areaList
	#ax = fig.add_subplot(1,2,1)
	#ax.plot(areaList[:,1],areaList[:,2], '.-', label="{0}".format(temp))
	#ax.set_ylim([0,1])
	#ax.set_xlabel("Density")
	#ax.set_ylabel("P[n>{0}]".format(threshold))
	ax1 = fig1.add_subplot(1,1,1)
	ax1.plot(areaList[:,3],areaList[:,1], '+-', label="{0}".format(temp))
	tlabel.append(areaList[0,0])
	ax1.set_xlim([0,1])
	ax1.set_ylabel(r"$\rho$*")
	ax1.set_xlabel("P[n>{0}]".format(threshold))
	#print i,areaList[i,0],areaList[i,3],areaList[i,1]
	ax1.annotate(areaList[0,0],
		xy=(areaList[i,3],areaList[i,1]),
		xytext=(20,20),
		textcoords='offset points', ha='right', va='bottom', 
		bbox=dict(boxstyle='round,pad=0.5', fc="white",alpha=0.5),
		arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
	)
	ax2 = fig2.add_subplot(1,1,1)
	ax2.plot(areaList[:,1],areaList[:,4], '+-', label="{0}".format(temp))
	#ax2.legend(areaList[:,1])
	ax2.set_xlim([0.2,1.0])
	ax2.set_xlabel("Density")
	ax2.set_ylabel("Average Neighbour, <N>".format(threshold))
	ax3 = fig3.add_subplot(1,1,1)
	ax3.plot(areaList[:,1],areaList[:,5], '+-', label="{0}".format(temp))
	ax3.set_xlim([0.2,1.0])
	#ax3.legend(areaList[:,1])
	ax3.set_xlabel("Density")
	ax3.set_ylabel("Mode Neighbour, Max(N)".format(threshold))
#plt.xlabel("Density")
#ax1.set_ylabel("Area Ratio")
#ax2.set_xlabel("T")
#plt.ylabel("Cumulative probability")
ax1.plot((0,1.1),(0.316,0.316),"k--")
#ax2.set_xlim((0,6))
#ax2.set_ylim((0,1.02))
#ax1.legend(loc="upper right")
#plt.legend(loc="best")
#ax1.legend(tlabel)
ax2.legend(tlabel)
ax3.legend(tlabel)
fig1.show()
#fig2.show()
#fig3.show()
#raw_input()
#fig1.savefig("IsothermNProb.png")
