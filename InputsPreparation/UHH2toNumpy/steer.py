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
procnames = ['TTbar','QCD_Ele', 'ST', 'DYJets', 'WJets', 'Diboson', 'Zprime']
#procnames = ['Diboson']
#name of branches to be skipped in conversion
## exact names
unwanted_exact_tags = ['genInfo','slimmedMETsPuppi','slimmedMETs_GenMET','slimmedElectronsUSER_']
## partial names to exclude common set of variables at once
unwanted_tags = ['weight_','jetsAk8CHSSubstructure_SoftDropCHS','jetsAk8PuppiSubstructure_SoftDropPuppi','genjetsAk8SubstructureSoftDrop','m_btag_MassDecorrelatedDeepBoosted','Top','m_btag_Deep','rho', 'run', 'jetsAk4Puppi.m_charge','year','m_uncorr','bits','luminosityBlock','Flavour','Fraction','m_btaginfo','alpha','tags.tagdata','m_jetArea','m_chargedMultiplicity','isRealData','trk','nef','Class','Daughters','Station','EoverP','fbrem','m_PU_pT_hat_max','m_qScale', '_zprime', 'm_photonEnergyFraction','m_binningValues','m_electronMultiplicity', 'jet4', 'jet5', 'jet6', 'jet7', 'jet8', 'jet9', 'jet10','tau32', 'tau21','px_','py_','met_p','pz_','m_dEtaInSeed','FlightDistance','m_Track','_error','m_pdf_','m_source_candidates','m_shifted','GenJets','GenParticles','MINIIso','m_VertexChi2','gsfTrack','m_full','beta','m_photonIso','Vertex','Error','mva','originalXWGTU','m_SecondaryVertex','.m_btag_BoostedDouble','m_btag_MassIndependent','m_tune','parton','m_pileup','beamspot','volatility','trigger','m_sim','pdgId','m_HFEMPuppiMultiplicity','m_JEC','offlineSlimmedPrimaryVertices','prefiringWeight','Multiplicity','_groomed','track','Ecal','ecal','index','Compatibility','m_cef','keys','m_muf','m_Nclusters','Significance','Pixel','rawCHS','genparticles_indices','sigma','Hits','m_chf','AEff','m_pu','veto','m_tau','over','Photon','m_nhf','sumPU','ecal','m_d','chi2','Chi2','Nu_','genjet_index']

syst_vars = ['NOMINAL']
for syst_var in syst_vars:
    print'--- Convert inputs for ',syst_var,' variation ---'
    ModuleRunner.ReadoutMLVariables(procnames=procnames,unwanted_tags=unwanted_tags, unwanted_exact_tags=unwanted_exact_tags,syst_var=syst_var)

