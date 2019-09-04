import os
import sys
import subprocess
from multiprocessing import Process
import ROOT

from functions_ml import *
from constants   import *



class ModuleRunner:

    def __init__(self, path, outpath):
        self.path = path
        self.outpath = outpath
   
    def MakeOutDir(self, filename):
        outdir = out_dir(self.path, filename)
        dirs_to_create = []
        if os.path.exists(outdir):
            print 'Outdir %s already exists. Not creating it.' % outdir
        while not os.path.exists(outdir):
            dirs_to_create[:0] = [outdir]
            split_dir = outdir.split('/')
            outdir = ''
            for i in range(len(split_dir)-2):
                outdir += '/'
                outdir = outdir+split_dir[i+1]
        for directory in dirs_to_create:
            os.mkdir(directory)
            print 'created directory %s' % directory

   
    def ReadoutMLVariables(self, procnames, unwanted_tags, unwanted_exact_tags,syst_var):
        inpath = fullsel_path
        fullinpath = inpath + '/'+syst_var+'/uhh2.AnalysisModuleRunner.'
        for proc in procnames:
            filename = fullinpath
            if not proc == 'DATA':
                filename += 'MC.' + proc + '.root'
            else:
                filename += 'DATA.' + proc + '.root'
            outpath = self.outpath + '/MLInput/'+syst_var+'/'
            procoutname = proc
            if proc == 'RSGluon':
                procoutname = 'RSGluon_All'
            readout_to_numpy_arrays(infilename=filename, treename='AnalysisTree', outpath=outpath, outname=procoutname, unwanted_tags=unwanted_tags, unwanted_exact_tags=unwanted_exact_tags)

   
