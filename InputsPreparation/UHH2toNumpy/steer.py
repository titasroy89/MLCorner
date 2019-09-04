#!/usr/bin/env python

import subprocess
import os
import time

from python.ModuleRunner import *
from python.constants import *


"""This is macro to steer Root to Numpy conversion. """


#paths are set in constants.py
ModuleRunner = ModuleRunner(path_MLDIR, outpath)

# ---- Macro for ML inputs preparation ----
#names of the process, e.g part after uhh2.AnalysisModuleRunner. in the input file name
procnames = ['TTbar', 'QCD_Mu', 'ST', 'DYJets', 'WJets', 'RSGluon', 'RSGluon_M1000', 'RSGluon_M2000', 'RSGluon_M3000', 'RSGluon_M4000', 'RSGluon_M5000', 'RSGluon_M6000']
#name of branches to be skipped in conversion
## exact names
unwanted_exact_tags = ['n_ele', 'n_mu']
## partial names to exclude common set of variables at once
unwanted_tags = ['weight_', 'jet4', 'jet5', 'jet6', 'jet7', 'jet8', 'jet9', 'jet10', 'N_bJets_', '_zprime', 'eventweight', 
                  '_ele1', 'tau32', 'tau21','e_jet','px_','py_','e_ak8','met_p','pz_']

syst_vars = ['NOMINAL','JEC_up','JEC_down','JER_down','JER_up']
for syst_var in syst_vars:
    print'--- Convert inputs for ',syst_var,' variation ---'
    ModuleRunner.ReadoutMLVariables(procnames=procnames,unwanted_tags=unwanted_tags, unwanted_exact_tags=unwanted_exact_tags,syst_var=syst_var)

