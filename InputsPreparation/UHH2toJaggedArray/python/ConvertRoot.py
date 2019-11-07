import uproot
import numpy as np


#Calculate dR between 2 jets
def dR(eta1,phi1,eta2,phi2):
 dR = ((eta2-eta1)**2+(phi2-phi2)**2)**0.5
 return dR

#Convert Root to JaggedArray and store only variables needed for 4-vector calculation 
#Advantage of JaggedArray: all entries per vector of objects, e.g jets, are stored
#The example developed with T* + T*-> (t+gluon) + (t+gluon) reconstruction in mind
def ConvertRootToArray(output_path='input_csv_Array_wTTbarsemilepReco_exactW', input_path='input_root_wTTbarsemilepReco_exactW/', name='MC_TstarTstarToTgluonTgluon_M-1500_Run2016v3'):
    print ("Skim input:",name)
    file_name=input_path+'/uhh2.AnalysisModuleRunner.MC.'+name+'.root'
    file = uproot.open(file_name) #File after UHH2 AnalysisModule with TTree usually called "AnalysisTree"
    ana_tree = file['AnalysisTree']

    objects_ak8jets=['jetsAk8PuppiSubstructure_SoftDropPuppi'] #Name of TBranch
    objects_ak4jets=['jetsAk4Puppi'] #Name of TBranch
    objects_leptons=['slimmedElectronsUSER', 'slimmedMuonsUSER'] #Name of TBranch


    vars_jets = ['m_pt','m_eta','m_phi','m_energy'] #Name of sub-branches for jets
    vars_leptons = ['m_pt','m_eta','m_phi','m_energy'] #Name of sub-branches for leptons
    
    jetsak4_array_all=[] #list of all varaibles we would like to store in skimmed pd DataFrame
    for jetname in objects_ak4jets:
        for varname in vars_jets:
            jetsak4_array_all.append(jetname+'.'+varname)

    df_ak4jets= ana_tree.arrays(jetsak4_array_all,outputtype=tuple)

    jetsak8_array_all=[] #list of all varaibles we would like to store in skimmed pd DataFrame
    for jetname in objects_ak8jets:
        for varname in vars_jets:
            jetsak8_array_all.append(jetname+'.'+varname)

    df_ak8jets= ana_tree.arrays(jetsak8_array_all,outputtype=tuple)
  
    df_ak4jets_overlapped = []
    for ievent in range(len(df_ak8jets[:][0])):
        df_ak4jets_overlapped_i = []
        #if(len(df_ak4jets[:][1][ievent])<1 or len(df_ak8jets[:][1][ievent])<1): continue
        for iak8jets in range(len(df_ak8jets[:][0][ievent])):
            for iak4jets in range(len(df_ak4jets[:][0][ievent])):
                dR_ak8ak4 =  dR(df_ak8jets[1][ievent][iak8jets],df_ak8jets[2][ievent][iak8jets],df_ak4jets[1][ievent][iak4jets],df_ak4jets[2][ievent][iak4jets])
                if(dR_ak8ak4<0.4):
                    df_ak4jets_overlapped_i.append(iak4jets)
        df_ak4jets_overlapped.append([ievent,df_ak4jets_overlapped_i]) #ak4jets overlaped with ak8 jets

    leptons_array_all=[] #list of all varaibles we would like to store in skimmed pd DataFrame
    for lepname in objects_leptons:
        for varname in vars_leptons:
            leptons_array_all.append(lepname+'.'+varname)

    leptons_array_all=[] #list of all varaibles we would like to store in skimmed pd DataFrame
    for lepname in objects_leptons:
        for varname in vars_leptons:
            leptons_array_all.append(lepname+'.'+varname)

    df_leptons= ana_tree.arrays(leptons_array_all,outputtype=tuple)

    df_MET = pd.DataFrame({'neutrino_v4_fPt':ana_tree['TTbarReconstruction_best']['m_neutrino_v4.fCoordinates.fPt'].array(), 
                           'neutrino_v4_fEta':ana_tree['TTbarReconstruction_best']['m_neutrino_v4.fCoordinates.fEta'].array(), 
                           'neutrino_v4_fPhi':ana_tree['TTbarReconstruction_best']['m_neutrino_v4.fCoordinates.fPhi'].array(), 
                           'neutrino_v4_fE':ana_tree['TTbarReconstruction_best']['m_neutrino_v4.fCoordinates.fE'].array(), 
                       })
    df_MET_array = df_MET.to_numpy()
    print("N events:",len(df_MET_array))
    #store all arrays in compressed file
    np.savez(output_path+'/'+name+'_MET_leptons_jets.npz',MET=df_MET_array,LEP=df_leptons,AK4JETS=df_ak4jets,AK8JETS=df_ak8jets,JETSOverlap=df_ak4jets_overlapped)
