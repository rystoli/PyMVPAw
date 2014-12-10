#version: 11_Nov_14

from wrapit_pymvpa import *

###########
# This readies the environment for analysis, importing modules and data etc.


####################################
homedir = '/home/freeman_lab/fMRI/Stolier/brc'
#make mask that includes everyone
#maskpath = 'masks/avg_mask_d3_BEST.nii.gz'
dsmspath= 'analysis/DMs_brc.hdf5'
#remap per subject
#remappath = 'analysis/data_remap_159.nii.gz'

#load data
datafiles = {'pv2': 'prep_passview/brc_allsubjs_pv2.gzipped.hdf5', 'pv3': 'prep_passview/brc_allsubjs_pv3.gzipped.hdf5', 'pvm': 'prep_passview/brc_allsubjs_pvm.gzipped.hdf5', 'tc2': 'prep_3choice/brc_allsubjs_tc2.gzipped.hdf5', 'tc3': 'prep_3choice/brc_allsubjs_tc3.gzipped.hdf5', 'tcm': 'prep_3choice/brc_allsubjs_tcm.gzipped.hdf5'}

#load remap ds
#remap = fmri_dataset(os.path.join(homedir,remappath),mask=os.path.join(homedir,maskpath))
#remap per subj, for native analysis.... specifies one ds fo rhtis specific study
#remap_native = dict((s,fmri_dataset(os.path.join(homedir,remap_nativepath,s,'%s_betasliced_01.nii.gz' % s))) for s in sexdata.keys())
#print('Remap dataset loaded from %s, with shape:' % (os.path.join(homedir,remappath)),remap.shape)




dsms = dsms_refresh(homedir,dsmspath)

dataf = raw_input('\nPlease specify your dataset from: %s\n\n  >>' % ('   '.join(datafiles.keys())))
data = h5load(os.path.join(homedir,datafiles[dataf]))

for s,ds in data.iteritems():
   print(ds.a.subjID, ds.a.task, ds.UC, ds.UT, ds.shape)

print('\nAnalysis materials successfully loaded')
