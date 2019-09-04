import os
import sys
import subprocess
import time
import signal
import ROOT
import numpy as np
from root_numpy import root2array, rec2array
import pickle

from multiprocessing import Process
from multiprocessing import Pool

def readout_to_numpy_arrays(infilename, treename, outpath, outname, unwanted_tags, unwanted_exact_tags):
    infile = ROOT.TFile.Open(infilename)

    myoutpath = outpath
    create_path(myoutpath)

    print 'creating numpy arrays for input sample %s' % (outname)
    # Get AnalysisTree
    entries = infile.AnalysisTree.GetEntriesFast()
    # print entries
    tree = infile.Get(treename)
    leaves = tree.GetListOfLeaves()
    variables = []
    eventweights = ['eventweight']
    for leaf in leaves:
        write = True
        for tag in unwanted_tags:
            if tag in leaf.GetName(): write = False
        for tag in unwanted_exact_tags:
            if tag == leaf.GetName(): write = False
        if write: variables.append(leaf.GetName())
    print variables
    print "len(variables): ",len(variables)

    chunksize = 200000
    maxidx = int(entries/float(chunksize)) + 1
    if entries % chunksize == 0: maxidx -= 1
    print entries, chunksize, maxidx
    for i in range(maxidx):
        mymatrix = root2array(filenames=infilename, treename=treename, branches=variables, start=i*chunksize, stop=(i+1)*chunksize)
        mymatrix = rec2array(mymatrix)
        myweights = root2array(filenames=infilename, treename=treename, branches=eventweights, start=i*chunksize, stop=(i+1)*chunksize)
        myweights = rec2array(myweights)

        thisoutname = myoutpath + outname + '_' + str(i) + '.npy'
        thisoutname_weights = myoutpath + 'Weights_' + outname + '_' + str(i) + '.npy'
        np.save(thisoutname, mymatrix)
        np.save(thisoutname_weights, myweights)
        percent = float(i+1)/float(maxidx) * 100.
        sys.stdout.write( '{0:d} of {1:d} ({2:4.2f} %) jobs done.\r'.format(i+1, maxidx, percent))
        if not i == maxidx-1: sys.stdout.flush()


    with open(myoutpath + 'variable_names.pkl', 'w') as f:
        pickle.dump(variables, f)


def create_path(path):
    if os.path.isdir(path):
        print 'path "%s" already exists, not creating it.' % (path)
    else:
        os.makedirs(path)
        print 'Created path "%s"' % (path)
