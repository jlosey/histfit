#!/usr/bin/env python
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pylab as plt
import sys

fig1=plt.figure(1,figsize=(10,6))
#fig2=plt.figure(2,figsize=(10,6))
#fig3=plt.figure(3,figsize=(10,6))
if len(sys.argv) > 1:
	ti = int(sys.argv[1]) 
else:
	ti = 5
denslabel = []
threshold = 4
plabel = []
tlabel = []
#tindex = [2,4,6,12,13,14,14,15,16,16,16,16,15,14,13,12,2]
#dlist = ["0.02","0.05","0.10","0.15","0.20","0.25","0.30","0.316","0.35","0.40","0.45","0.50","0.55","0.60","0.65","0.70","0.75"]
dlist = ["0.20","0.25","0.30","0.316","0.35","0.40"]
clist = ["r","g","b","c","m","y","k"]
alabel = [-7,-7,-6,-7,-7,-7]
for d,c,a in zip(dlist,clist,alabel):
	flist = sorted(glob.glob("../v5/gClst_0-0-*-{0}-500000.dat".format(d)))
	#flist = sorted(glob.glob("../v4/gClst_5-0-*-{0}-250000.dat".format(d)))
	tempL = []
	dL = []
	sCL = []
	sAvL = []
	cCL = []
	cAvL = []
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

		#Calculate values from .dat files
		dataS = np.asarray(pSn)
		dataC = np.asarray(pClst)
		dataSL = dataS[threshold:]
		dataCL = dataC[threshold:]
		indices = np.arange(dataC.size)
		sAvg = np.multiply(indices,dataS).sum() 
		cAvg = np.multiply(indices,dataC).sum() 
		cMax = dataC.argmax()
		sLSum = dataSL.sum()
		cLSum = dataCL.sum()
		#Record values in lists
		tempL.append(temp)
		sCL.append(sLSum)
		sAvL.append(sAvg)
		cCL.append(cLSum)
		cAvL.append(cAvg)

	#Sort lists by ascending percentage
	cCL = np.asarray(cCL)
	tempL = np.asarray(tempL)
	ind = np.argsort(cCL)
	cCLsort = cCL[ind]
	tLsort = tempL[ind]
	#sio.savemat("ccL-{0}.mat".format(dens),{"cCl":cCLsort})	
	#sio.savemat("temp-{0}.mat".format(dens),{"temp":tLsort})	
	#cCLsort.astype("float32").tofile("cCl-{0}.dat".format(dens))
	#tLsort.astype("float32").tofile("temp-{0}.dat".format(dens))
	#Fit sorted data to to 3rd degree polynomial. Skip n highest temps. 
	#n = 8
	fitC,resid,rank,vdm,rcond = np.polyfit(cCLsort[ti:],tLsort[ti:],3,full=True)
	p = np.poly1d(fitC)
	print p,resid
	#print tLsort[n:]
	pdCmplx = p.deriv().r
	pdReal = pdCmplx.real
	pd2Cmplx = p.deriv(m=2).r
	pd2Real = pdCmplx.real
	print pdReal,p(pdReal),pd2Real,p(pd2Real)
	#print dens,p.r, p(p.r),p.deriv
	pline = np.linspace(0,1,1000)
	dp = np.diff(p(pline))
	idx = np.abs(dp).argmin()
	dp2 = np.diff(p(pline),2)
	dpl = pline[0:dp.size] 
	dp2l = pline[0:dp2.size] 
	idx2 = np.abs(dp2).argmin()
	#print dens,pline[idx],p(pline[idx]),dp[idx],pline[idx2],p(pline[idx2]),dp2[idx2]
	ax = fig1.add_subplot(1,1,1)
	ax.set_xlim([0,1])
	ax.set_ylim([0,4.5])
	ax.plot(cCL,tempL,".",pline,p(pline),"--",pline[idx],p(pline[idx]),color=c,lw=0.6)
	#ax.plot(cCL,tempL,".",pline,p(pline),pline[idx],p(pline[idx]),"*k",pline[idx2],p(pline[idx2]),"s",ms=10)
	ax.plot(pdReal[0],p(pdReal[0]),"^",pd2Real[0],p(pd2Real[0]),"*",color=c,ms=8)
	ax.annotate(d,
		xy=(cCL[a],tempL[a]),
		xytext=(20,20),
		textcoords='offset points', ha='right', va='bottom', 
		bbox=dict(boxstyle='round,pad=0.5', fc="white",alpha=0.5),
		arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
	)
	#ax2 = fig1.add_subplot(1,2,2)
	#ax2.set_ylim([-0.5,0.2])
	#ax2.plot(dpl,dp,dpl[idx],dp[idx],"*",dp2l,dp2,"--")
fig1.show()
raw_input()
