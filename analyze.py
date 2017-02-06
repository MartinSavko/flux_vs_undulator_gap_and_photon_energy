#!/usr/bin/env python

import pickle
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import numpy
import glob

import re
import seaborn as sns

scans = glob.glob('undulator*_step_50*pck')

def get_gap(filename):
    return float(re.findall('.*_gap_(.*)_step.*pck', filename)[0])
 
def XYZ(x,y,z):
    '''Go through the results and return X, Y, Z matrices for 3d plots'''
    X, Y, Z = [numpy.array(l) for l in [x, y, z]]
    X, Y, Z = [numpy.reshape(l, self.nbsteps) for l in [X, Y, Z]]
    return X, Y, Z

scans.sort(key=get_gap)
ens = numpy.arange(4000, 20001, 50)
fluxes = []
gaps = []
for scan in scans:
    gap = get_gap(scan)
    r = numpy.array(pickle.load(open(scan)))
    energies = r[:, 0]
    flux = r[:, 1]
    if abs(energies[0] - 4000) > 2:
        flux = flux[::-1]
    fluxes.append(flux)
    gaps.append(gap)
    
plt.figure(1)
k=0
for gap in gaps:
    plt.plot(energies, fluxes[k], label='gap=%s' % gaps[k])
    k+=1
    
plt.xlabel('energy [eV]')
plt.ylabel('flux [ph/s]')
plt.title('photon flux vs energy vs undulator gap')
plt.legend()
plt.grid(True)

fluxes = numpy.array(fluxes)
#fluxes = fluxes[::-1,::-1]
ens, gaps = numpy.meshgrid(ens, gaps)

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(ens, gaps, fluxes, rstride=1, cstride=4) #, azimuth=-111, elevation=54)
ax.set_xlabel('energy [eV]')
ax.set_ylabel('gap [mm]')
ax.set_zlabel('flux [ph/s]')
#ax.set_azimuth(-111)
#ax.set_elevation(54)
ax.view_init(54, -111)
plt.show()
    
        
