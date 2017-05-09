#!/usr/bin/env python
import numpy as np
import glob
from scipy.integrate import simps
 
flist = sorted(glob.glob("../v3/gRDF_0-0-1.60-*-25000.dat"),reverse=True)
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
	rMinInd = np.where((rA > 1.3)&(rA < 2))
	#print rMin
	minGr = grA[rMinInd].min()
	minRInd = np.where(grA == minGr)
	ind1 = np.where(rA < rA[minRInd])
	rAmin = float(rA[minRInd])
	grDenA = float(dens)*grA[ind1]
	i1 = 4*np.pi*simps(grDenA,rA[ind1])
	print temp,dens,rAmin,i1 
