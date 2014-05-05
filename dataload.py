from pymvpanalyzer.py import *

###########
# This readies the environment for analysis, importing modules and data etc.


####################################
homedir = '/home/freeman_lab/fMRI/Stolier/SRMTfMRI'
dpath = 'prep/standard_13.11/'
maskpath = 'masks/avg_mask_d3_BEST.nii.gz'
dsmspath= 'analysis/dsms_sr.hdf5'
remappath = 'analysis/data_remap_159.nii.gz'

#load data
sexdata = h5load(os.path.join(homedir,dpath,'srmtfmri_sex.gzipped.hdf5'))
racedata = h5load(os.path.join(homedir,dpath,'srmtfmri_race.gzipped.hdf5'))
colordata = h5load(os.path.join(homedir,dpath,'srmtfmri_color.gzipped.hdf5'))
print('\nLoading data from %s ...' % (os.path.join(homedir,dpath)))
print('Sex task:')
for ds in sexdata:
    print('Subj %s ; shape:' % (ds), sexdata[ds].shape)
print('Race task:')
for ds in racedata:
    print('Subj %s ; shape:' % (ds), racedata[ds].shape)
print('Color task:')
for ds in colordata:
    print('Subj %s ; shape:' % (ds), colordata[ds].shape)

#load remap ds
remap = fmri_dataset(os.path.join(homedir,remappath),mask=os.path.join(homedir,maskpath))
print('Remap dataset loaded from %s, with shape:' % (os.path.join(homedir,remappath)),remap.shape)

dsms = dsms_refresh()

print('\nAnalysis materials successfully loaded')
