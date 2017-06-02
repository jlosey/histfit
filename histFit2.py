#!/usr/bin/env python
import glob
import numpy as np
from scipy.stats import normaltest
from scipy.integrate import quad 
import matplotlib.pylab as plt
import sys

fig=plt.figure(1,figsize=(7,4))
count = 1
threshold = int(sys.argv[1]) 
dlist = ["0.20","0.316","0.40","0.60"]
#dlist = ["0.05", "0.10","0.15","0.20","0.25"]
for d in dlist:
	flist = sorted(glob.glob("../v4/gClst_0-0-*-{0}-*.dat".format(d)))
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
		X = np.linspace(0,20,200)
		xA = np.arange(0,21)
		nAr = np.asarray(n)
		nAr2 = nAr[1:]
		nArL = nAr[4:]

		#Stillinger fit
		dataS = np.asarray(pSn)
		maxDataS = dataS.max()
		avgS = np.sum(dataS*nAr)/np.sum(dataS)
		widthS = np.sqrt(np.abs(np.sum((nAr-avgS)**2*dataS)/np.sum(dataS)))
		fitS = lambda t : maxDataS*np.exp(-(t-avgS)**2/(2*widthS**2))

		#Cluster of neighbors with > 4 Stillinger nieghbors
		dataC = np.asarray(pClst)
		dataC2 = dataC[1:]
		dataSL = dataS[threshold:]
		dataCL = dataC[threshold:]
		maxDataC = dataC.max()
		maxDataC2 = dataC2.max()
		avgC = np.sum(dataC*nAr)/np.sum(dataC)
		avgC2 = np.sum(dataC2*nAr2)/np.sum(dataC2)
		#avgCL = np.sum(dataCL*nArL)/np.sum(dataCL)
		widthC = np.sqrt(np.abs(np.sum((nAr-avgC)**2*dataC)/np.sum(dataC)))
		widthC2 = np.sqrt(np.abs(np.sum((nAr2-avgC2)**2*dataC2)/np.sum(dataC2)))
		fitC = lambda t : maxDataC*np.exp(-(t-avgC)**2/(2*widthC**2))
		fitC2 = lambda t : maxDataC2*np.exp(-(t-avgC2)**2/(2*widthC2**2))
		#fitCL = lambda t : maxDataCL*np.exp(-(t-avgCL)**2/(2*widthCL**2))
		area,aerr = quad(fitC,0,20)
		area2,a2err = quad(fitC2,0,20)
		#areaL,aLerr = quad(fitCL,0,20)
		cSum = sum(dataC)
		c2Sum = sum(dataC2)
		sLSum = sum(dataSL)
		cLSum = sum(dataCL)
		areaList.append((temp,dens,sLSum,cLSum))

	areaList = np.asarray(areaList)
	ind = np.lexsort((areaList[:,0],areaList[:,1]))
	areaList = areaList[ind]
	#plt.plot(areaList[:,0],areaList[:,6])
	ax = fig.add_subplot(1,2,1)
	ax.set_xlabel("T")
	ax.set_ylabel("P[n > {0}]".format(threshold))
	ax.set_ylim([0,1])
	plt.plot(areaList[:,0],areaList[:,2], '.-', label="{0}".format(dens))
	ax = fig.add_subplot(1,2,2)
	ax.set_xlabel("T")
	ax.set_ylabel("P[n > {0}]".format(threshold))
	ax.set_ylim([0,1])
	plt.plot(areaList[:,0],areaList[:,3], '^-', label="{0}".format(dens))
print areaList
#fig.xlabel("T")
#fig.ylabel("P[n > {0}]")
plt.legend(loc="upper right")
plt.show()
#plt.savefig("AreaRatio.png")
