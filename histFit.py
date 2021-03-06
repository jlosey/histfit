#!/usr/bin/env python
import glob
import numpy as np
from scipy.stats import normaltest
from scipy.integrate import quad 
import matplotlib.pylab as plt

flist = sorted(glob.glob("../v3/gClst_0-0-*-0.90-25000.dat"),reverse=True)
fig=plt.figure(1,figsize=(12,10))
areaList = []
count = 1
for fl in flist:
	n = []
	sCount = []
	pSn = []
	clstCount = []
	pClst = []
	temp = fl[16:20]
	dens = fl[21:25]
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

	#Stillinger fit
	dataS = np.asarray(pSn)
	maxDataS = dataS.max()
	avgS = np.sum(dataS*nAr)/np.sum(dataS)
	widthS = np.sqrt(np.abs(np.sum((nAr-avgS)**2*dataS)/np.sum(dataS)))
	fitS = lambda t : maxDataS*np.exp(-(t-avgS)**2/(2*widthS**2))

	#Cluster of neighbors with > 4 Stillinger nieghbors
	dataC = np.asarray(pClst)
	dataC2 = dataC[1:]
	maxDataC = dataC.max()
	maxDataC2 = dataC2.max()
	avgC = np.sum(dataC*nAr)/np.sum(dataC)
	avgC2 = np.sum(dataC2*nAr2)/np.sum(dataC2)
	widthC = np.sqrt(np.abs(np.sum((nAr-avgC)**2*dataC)/np.sum(dataC)))
	widthC2 = np.sqrt(np.abs(np.sum((nAr2-avgC2)**2*dataC2)/np.sum(dataC2)))
	fitC = lambda t : maxDataC*np.exp(-(t-avgC)**2/(2*widthC**2))
	fitC2 = lambda t : maxDataC2*np.exp(-(t-avgC2)**2/(2*widthC2**2))
	area,aerr = quad(fitC,0,20)
	area2,a2err = quad(fitC2,0,20)
	cSum = sum(dataC)
	c2Sum = sum(dataC2)
	areaList.append((temp,dens,area,aerr,area2,a2err,area/area2,cSum,c2Sum))
	#print("Temp = {0} Rho = {1}".format(temp, dens))
	#print("Area = {0}, area2 = {1}, simA = {2}".format(sum(fitC(X)),c2A,sum(dataC[1:])))
	#test = normaltest(data)
	#test2 = normaltest(data2-x)
	#print test,test2
	#width = np.sqrt(np.abs(np.sum((nAr2-x)**2*data2)/np.sum(data2)))

	#plt.bar(nAr-0.5,dataS,width=1)
	ax = fig.add_subplot(3,2,count)
	ax.set_title("T={0} Dens={1}".format(temp,dens))
	ax.set_xlim([0,20])
	ax.set_xlabel("n")
	ax.set_ylabel("P[n]")
	#ax.plot(nAr,dataS,'b*',label="Stillinger")
	ax.plot(nAr,dataC,'g.', ms=10, label="Cluster Data")
	ax.plot(X,fitC(X),'k--', label="Fit All")
	ax.plot(X,fitC2(X),'r-', label="Fit n > 0")
	if count==1:
		ax.legend(loc="upper right")
	count = count+1

areaList = np.asarray(areaList)
fig = plt.figure(2)
#plt.plot(areaList[:,0],areaList[:,6])
#plt.plot(areaList[:,0],areaList[:,7])
plt.plot(areaList[:,0],areaList[:,8])
plt.xlabel("T")
plt.ylabel("Area")
#plt.ylim([0,0.2])
plt.show()
#plt.savefig("%s-nearN.png"%temp)
