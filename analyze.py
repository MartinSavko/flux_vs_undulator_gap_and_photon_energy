#!/usr/bin/env python

import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import numpy as np
import glob

import re
import seaborn as sns

def get_gap(filename):
    return float(re.findall('.*_gap_(.*)_step.*pkl', filename)[0])
 
def XYZ(x,y,z):
    '''Go through the results and return X, Y, Z matrices for 3d plots'''
    X, Y, Z = [np.array(l) for l in [x, y, z]]
    X, Y, Z = [np.reshape(l, self.nbsteps) for l in [X, Y, Z]]
    return X, Y, Z

def get_ring_current(filename):
    return float(re.findall('.*_scan_(.*)mA_.*pkl', filename)[0])

def get_slit_opening(filename):
    return re.findall('.*_ps_(.*)_gap_.*pkl', filename)[0]

def main():
    import optparse
    import os
    parser = optparse.OptionParser()
    parser.add_option('-d', '--directory', default='scans/ps_4.0x4.0', type=str, help='Directory with the scan results')
    parser.add_option('-t', '--template', default='undulator*_step_50*pkl', type=str, help='glob template to identify the result files')
    options, args = parser.parse_args()
    
    scans = glob.glob(os.path.join(options.directory, options.template))
    
    scans.sort(key=get_gap)
    print 'scans'
    print scans
    
    ens = np.arange(4000, 20001, 50)
    fluxes = []
    gaps = []
    plt.figure(1)
    for scan in scans:
        gap = get_gap(scan)
        r = np.array(pickle.load(open(scan)))
        energies = r[:, 0]
        flux = r[:, 1]
        plt.plot(energies, flux, label='gap=%s' % gap)
        if abs(energies[0] - 4000) > 2:
            flux = flux[::-1]
        fluxes.append(flux)
        gaps.append(gap)
        
    #k=0
    #for gap in gaps:
        #plt.plot(ens, fluxes[k][::-1], label='gap=%s' % gaps[k])
        #k+=1
        
    plt.xlabel('energy [eV]')
    plt.ylabel('flux [ph/s]')
    plt.title('Photon flux vs energy vs undulator gap; ring current %d mA, PS %s mm' % (get_ring_current(scans[0]), get_slit_opening(scans[0])))
    plt.legend()
    plt.grid(True)

    fluxes = np.array(fluxes)
    ens, gaps = np.meshgrid(ens, gaps)

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(ens, gaps, fluxes, rstride=1, cstride=2)
    ax.set_xlabel('energy [eV]')
    ax.set_ylabel('gap [mm]')
    ax.set_zlabel('flux [ph/s]')
    plt.title('Photon flux vs energy vs undulator gap; ring current %d mA, PS %s mm' % (get_ring_current(scans[0]), get_slit_opening(scans[0])))
    ax.view_init(54, -111)
    plt.show()
        
if __name__ == '__main__':
    main()
        
