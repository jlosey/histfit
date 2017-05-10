#!/usr/bin/env python
import numpy as np
import glob
from scipy.integrate import simps
import matplotlib.pylab as plt
fig = plt.figure(figsize=(10,12))
count = 1 
flist = sorted(glob.glob("../v3/gRDF_0-0-0.70-*-25000.dat"),reverse=True)
for fl in flist:
	f = open(fl,"r")
	dens = fl[20:24]
	temp = fl[15:19]
	bn = []
	hCount = []
	r = []
	gr = []
	for line in f:
		li=line.strip()
		if not li.startswith("BinNum"):
			bn.append(int(li.split()[0]))
			r.append(float(li.split()[1]))
			gr.append(float(li.split()[2]))
	f.close()
	rA = np.asarray(r)
	grA = np.asarray(gr)
	rMinInd = np.where((rA > 1.3)&(rA < 2.2))
	#rMinInd = np.where((rA > 1.1)&(rA < 1.5))
	minGR = grA[rMinInd].min()
	#minGR = grA[int(rMinInd)]
	minGRi = int(np.where(grA == minGR)[0])
	rAmin = float(rA[minGRi])
	rLow = rA[0:minGRi]
	r2 = np.multiply(rLow,rLow)
	grDenA = np.multiply(grA[0:minGRi],r2)
	i1 = 4*np.pi*float(dens)*simps(grDenA,rA[0:minGRi])
	print temp,dens,rAmin,minGR,i1 
	ax = fig.add_subplot(4,2,count)
	ax.set_title(r"T={0} Dens={1} N_c={2:.4g}".format(temp,dens,i1))
	ax.set_xlabel("r")
	ax.set_ylabel("g(r)")
	ax.set_xlim([0,4])
	ax.plot(rA,grA,rAmin,minGR,"r*",rA,rA/rA,"k--")
	ax.fill_between(rA[0:minGRi],grA[0:minGRi])
	count = count + 1
fig.tight_layout()
plt.savefig("{0}-Coordination-Min.png".format(temp))
#plt.show() 
