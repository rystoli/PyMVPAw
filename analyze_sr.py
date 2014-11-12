#version: 11_Nov_14

##########TO DO
# clean up directory refs, and print outs
# add bit to automatically detect data hdf5s and run datadict2nifti()


#run this from facecat/analysis
######### NEEED TO FINISH DSMS #############
from slRSA_models_fc import *

slRSAdir = '/home/freeman_lab/fMRI/Stolier/SRMTfMRI/analysis'
studydir = '/home/freeman_lab/fMRI/Stolier/SRMTfMRI'
dataloc = '/home/freeman_lab/fMRI/Stolier/SRMTfMRI/prep/standard_13.11/srmtfmri_allsubjs_masked.gzipped.hdf5'
sexdata = '/home/freeman_lab/fMRI/Stolier/SRMTfMRI/prep/standard_13.11/srmtfmri_sex.gzipped.hdf5'
racedata = '/home/freeman_lab/fMRI/Stolier/SRMTfMRI/prep/standard_13.11/srmtfmri_race.gzipped.hdf5'
colordata = '/home/freeman_lab/fMRI/Stolier/SRMTfMRI/prep/standard_13.11/srmtfmri_color.gzipped.hdf5'



##################################################################
# apriori analyses
##################################################################
print('beginning apriori analyses')
slRSA_m_nSs(sexdata,dsms['apriori']['apriori'],['oF','er'],h5=1,h5out='srsex_nSs_apriori.hdf5')
slRSA_m_nSs(racedata,dsms['apriori']['apriori'],['oF','er'],h5=1,h5out='srrace_nSs_apriori.hdf5')
slRSA_m_nSs(colordata,dsms['apriori']['apriori'],['oF','er'],h5=1,h5out='srcolor_nSs_apriori.hdf5')

##################################################################
# avgIdio analyses 
##################################################################
print('beginning avgIdio analyses')
slRSA_m_nSs(sexdata,dsms['sexIdioAvg'],['oF','er'],cmetric='spearman',h5=1,h5out='srsex_nSs_idioAvg_sp.hdf5')
slRSA_m_nSs(racedata,dsms['raceIdioAvg'],['oF','er'],cmetric='spearman',h5=1,h5out='srrace_nSs_idioAvg_sp.hdf5')
slRSA_m_nSs(colordata,dsms['colorIdioAvg'],['oF','er'],cmetric='spearman',h5=1,h5out='srcolor_nSs_idioAvg_sp.hdf5')


##################################################################
# xSs 
##################################################################

dcm_map=slRSA_xSs(sexdata,['oF','er'],h5=1,h5out='srsex_xSs.hdf5')
for i,ds in enumerate(dcm_map.samples):
    slRSA2nifti(ds,remap,'%s_sex_xSs.nii.gz' % (i))
dcm_map=slRSA_xSs(racedata,['oF','er'],h5=1,h5out='srrace_xSs.hdf5')
for i,ds in enumerate(dcm_map.samples):
    slRSA2nifti(ds,remap,'%s_race_xSs.nii.gz' % (i))
dcm_map=slRSA_xSs(colordata,['oF','er'],h5=1,h5out='srcolor_xSs.hdf5')
for i,ds in enumerate(dcm_map.samples):
    slRSA2nifti(ds,remap,'%s_color_xSs.nii.gz' % (i))

# STILL? not sure if below is necessary due to datadict2niftis
# to move these to niftis, need to split dset up to many dsets by samples
for i,ds in enumerate(dcm_map.samples):
    slRSA2nifti(ds,remap,'%s_xSs.nii.gz' % (i))


##################################################################
#idio analyses
##################################################################
print('beginning idio analyses')
idio_sex = {}
for s,dm in dsms['sexIdio'].iteritems():
    print('Running slRSA on MT for %s' % (s))
    idio_sex[s] = slRSA_m_1Ss(sexdata[s],dm,['oF','er'],cmetric='spearman')[1]
h5save('sexIdio_sp.hdf5',idio_sex,compression=9)
idio_race = {}
for s,dm in dsms['raceIdio'].iteritems():
    print('Running slRSA on MT for %s' % (s))
    idio_race[s] = slRSA_m_1Ss(racedata[s],dm,['oF','er'],cmetric='spearman')[1]
h5save('raceIdio_sp.hdf5',idio_race,compression=9)
idio_color = {}
for s,dm in dsms['colorIdio'].iteritems():
    print('Running slRSA on MT for %s' % (s))
    idio_color[s] = slRSA_m_1Ss(colordata[s],dm,['oF','er'],cmetric='spearman')[1]
h5save('colorIdio_sp.hdf5',idio_color,compression=9)

print('idio slRSA complete, with hdf5s saved - thank R--n!')





















##################################################################
#next round analyses 1_26 - temp area for those not yet done
##################################################################
data=h5load(dataloc)
othnew = {
'emo_avg': dsms['oth']['emo_avg'],
'emo_avgC': dsms['oth']['emo_avgC'],
'sex_avg': dsms['oth']['sex_avg'],
'sex_avgC': dsms['oth']['sex_avgC'],
'idioAvg_12C': dsms['oth']['idioAvg_12C']
}
print('beginning non-idio analyses')
data = h5load(dataloc)
for dsm in othnew:
    slRSA_m_nSs(data,othnew[dsm],['oF','er'],h5=1,h5out='fc_slRSAnSs_%s.hdf5' % (dsm))

##################################################################
# 7x7
##################################################################
data=h5load(dataloc)
data7x7={}
for subj in data:
    data7x7[subj]=(overlap_mgs(data[subj],['sex','emo','race'],['oF','er']))

#################################################################
#xSs analysis - 
print('beginning xSs analyses')
dcm_map=slRSA_xSs(data7x7,['oF','er'],h5=1,h5out='fc_slRSA_xSs_7x7.hdf5')
print('xSs done')

##################################################################
#idio analyses
##################################################################
print('beginning idio analyses')
data=h5load(dataloc)
data7x7={}
for subj in data:
    data7x7[subj]=(overlap_mgs(data[subj],['sex','emo','race'],['oF','er']))

del(data7x7['299']) #missing idio DSM
idiodict7, idiodict12 = {}, {}
for dsm in dsms['idio_7']:
    if dsm[:3] in data.keys(): #only looks at subj in data
        if dsm[-1] == '7': 
            print('Running slRSA on MT for %s' % (dsm))
            idiodict7[dsm[:3]] = slRSA_m_1Ss(data7x7[dsm[:3]],dsms['idio_7'][dsm],['oF','er'])[1]
        else: 
            print('Running slRSA on MT for %s' % (dsm))
            idiodict12[dsm[:3]] = slRSA_m_1Ss(data7x7[dsm[:3]],dsms['idio_12'][dsm],['oF','er'])[1]
    else: pass
h5save('idioMT_7.hdf5',idiodict7,compression=9)
#h5save('idioMT_12.hdf5',idiodict12,compression=9)
print('idio slRSA compleexte, with hdf5s saved - thank R--n!')

##################################################################
# oth analyses = NEED TO MAKE bottom-up dsms 7x7
##################################################################
print('beginning non-idio analyses')
data = h5load(dataloc)
for dsm in dsms['oth']:
    slRSA_m_nSs(data,dsms['oth'][dsm],['oF','er'],h5=1,h5out='fc_slRSAnSs_%s.hdf5' % (dsm))

#data=h5load('datamini.hdf5') # for testing
#partials...
slRSA_m_nSs(data,dsms['oth']['idioAvg_12C'],['oF','er'],partial_dsm=dsms['oth']['orth_12'],h5=1,h5out='fc_slRSAnSs_pOrth12_%s.hdf5' % ('idioAvg_12C'))
slRSA_m_nSs(data,dsms['oth']['orth_12'],['oF','er'],partial_dsm=dsms['oth']['idioAvg_12C'],h5=1,h5out='fc_slRSAnSs_pIdioavg12c_%s.hdf5' % ('orth_12'))
#allcat control... is structure more fine than general cats?
slRSA_m_nSs(data,dsms['oth']['idioAvg_12C'],['oF','er'],partial_dsm=dsms['oth']['allcat_control'],h5=1,h5out='fc_slRSAnSs_MT12c_pAllcatcontrol.hdf5')
slRSA_m_nSs(data,dsms['oth']['orth_12'],['oF','er'],partial_dsm=dsms['oth']['allcat_control'],h5=1,h5out='fc_slRSAnSs_FG12_pAllcatcontrol.hdf5')
slRSA_m_nSs(data,dsms['oth']['allcat_control'],['oF','er'],partial_dsm=dsms['oth']['orth_12'],h5=1,h5out='fc_slRSAnSs_Allcat_pFG12.hdf5')
slRSA_m_nSs(data,dsms['oth']['allcat_control'],['oF','er'],partial_dsm=dsms['oth']['idioAvg_12C'],h5=1,h5out='fc_slRSAnSs_Allcat_pMT12.hdf5')
slRSA_m_nSs(data,dsms['oth']['idioAvg_12C'],['oF','er'],partial_dsm=dsms['oth']['topdownonlycontrol1'],h5=1,h5out='fc_slRSAnSs_mt12c_ptopdown.hdf5')
slRSA_m_nSs(data,dsms['oth']['topdownonlycontrol1'],['oF','er'],partial_dsm=dsms['oth']['idioAvg_12C'],h5=1,h5out='fc_slRSAnSs_topdown_pmt12c.hdf5')


#redo spearman main models:
slRSA_m_nSs(data,dsms['oth']['idioAvg_12C'],['oF','er'],cmetric='spearman',h5=1,h5out='fc_slRSAnSs_%s_spearman.hdf5' % ('MT12c'))
slRSA_m_nSs(data,dsms['oth']['orth_12'],['oF','er'],cmetric='spearman',h5=1,h5out='fc_slRSAnSs_%s_spearman.hdf5' % ('FG12'))
slRSA_m_nSs(data,dsms['oth']['idioAvg_12C'],['oF','er'],cmetric='spearman',partial_dsm=dsms['oth']['orth_12'],h5=1,h5out='fc_slRSAnSs_%s_pFG12_spearman.hdf5' % ('MT12c'))






###########################
#searchlight classificaiton
from slClassification import *
for i in data:
    data[i] = omit_targets(data[i],['er','oF'])
    data[i].sa['targets'] = data[i].sa.sex #depends on analysis
    #data[i].sa['targets'] = data[i].sa.emo
    #data[i].sa['targets'] = data[i].sa.race
slrs = slSVM_nSs(data)
#now have below done in function 'chance_dev'
#for i in slrs:
#    slrs[i].samples = slrs[i].samples[0]
#    slrs[i].samples -= .5 #chance to ttest against 0
datadict2nifti(slrs,remap,'slSVM_ofcrfg_sex','ofcrfg_slSVM_')

