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
tlist = ["1.00","1.312","1.3365","2.00","4.00","6.00","8.00","10.00","12.00"]
#dlist = ["0.25","0.27","0.29","0.32","0.32","0.34","0.37","0.40"]
#dlist = ["0.20","0.32","0.40","0.60"]
#dlist = ["0.05", "0.10","0.15","0.20","0.25"]
for t in tlist:
	flist = sorted(glob.glob("../v4/gClst_0-0-{0}-*-500000.dat".format(t)))
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
		#maxDataC = dataC.max()
		#maxDataC2 = dataC2.max()
		#avgC = np.sum(dataC*nAr)/np.sum(dataC)
		#avgC2 = np.sum(dataC2*nAr2)/np.sum(dataC2)
		#avgCL = np.sum(dataCL*nArL)/np.sum(dataCL)
		#widthC = np.sqrt(np.abs(np.sum((nAr-avgC)**2*dataC)/np.sum(dataC)))
		#widthC2 = np.sqrt(np.abs(np.sum((nAr2-avgC2)**2*dataC2)/np.sum(dataC2)))
		#widthCL = np.sqrt(np.abs(np.sum((nArL-avgCL)**2*dataCL)/np.sum(dataCL)))
		#fitC = lambda t : maxDataC*np.exp(-(t-avgC)**2/(2*widthC**2))
		#fitC2 = lambda t : maxDataC2*np.exp(-(t-avgC2)**2/(2*widthC2**2))
		#fitCL = lambda t : maxDataCL*np.exp(-(t-avgCL)**2/(2*widthCL**2))
		#area,aerr = quad(fitC,0,20)
		#area2,a2err = quad(fitC2,0,20)
		#areaL,aLerr = quad(fitCL,0,20)
		cSum = sum(dataC)
		#c2Sum = sum(dataC2)
		sSum = sum(dataSL)
		cLSum = sum(dataCL)
		areaList.append((temp,dens,sSum,cLSum))
		#print("Temp = {0} Rho = {1}".format(temp, dens))
		#print("Area = {0}, area2 = {1}, simA = {2}".format(sum(fitC(X)),c2A,sum(dataC[1:])))
		#test = normaltest(data)
		#test2 = normaltest(data2-x)
		#print test,test2
		#width = np.sqrt(np.abs(np.sum((nAr2-x)**2*data2)/np.sum(data2)))

		#plt.bar(nAr-0.5,dataS,width=1)
		#ax.set_title("T={0} Dens={1}".format(temp,dens))
		#ax.set_xlim([0,20])
		#ax.set_xlabel("n")
		#ax.set_ylabel("P[n]")
		#ax.plot(nAr,dataS,'b*',label="Stillinger")
		#ax.plot(nAr,dataC,'g.', ms=10, label="Cluster Data")
		#ax.plot(X,fitC(X),'k--', label="Fit All")
		#ax.plot(X,fitC2(X),'r-', label="Fit n > 0")
		#if count==1:
			#ax.legend(loc="upper right")
		#count = count+1

	areaList = np.asarray(areaList)
	ind = np.lexsort((areaList[:,0],areaList[:,1]))
	areaList = areaList[ind]
	#print areaList
	#ax = fig.add_subplot(1,2,1)
	#ax.plot(areaList[:,1],areaList[:,2], '.-', label="{0}".format(temp))
	#ax.set_ylim([0,1])
	#ax.set_xlabel("Density")
	#ax.set_ylabel("P[n>{0}]".format(threshold))
	ax = fig.add_subplot(1,1,1)
	ax.plot(areaList[:,1],areaList[:,3], '^-', label="{0}".format(temp))
	ax.set_ylim([0,1])
	ax.set_xlabel("Density")
	ax.set_ylabel("P[n>{0}]".format(threshold))
#plt.xlabel("Density")
#ax1.set_ylabel("Area Ratio")
#ax2.set_xlabel("T")
#plt.ylabel("Cumulative probability")
#plt.plot((1.326,1.326),(0,1),"k--")
#ax2.set_xlim((0,6))
#ax2.set_ylim((0,1.02))
#ax1.legend(loc="upper right")
plt.legend()
plt.show()
#plt.savefig("AreaRatio.png")