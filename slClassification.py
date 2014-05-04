from mvpa2.suite import *
import os
import pylab as pylab
import numpy as np

# to do
# make classifier an argument
# make omit optional....
#chance_level = 1.0 - (1.0 / len(ds.uniquetargets))
# need to set targets beforehand if swithcing what they are


##########################
# General functions
##########################

#remap slRSA results array to native space and save to nifti
def slRSA2nifti(ds,remap,outfile):
    '''
    No return; converts slRSA output and saves nifti file to working directory

    ds=array of slRSA results
    remap: dataset to remap to
    outfile: string outfile name including extension .nii.gz
    '''

    nimg = map2nifti(data=ds,dataset=remap)
    nimg.to_filename(outfile)

def datadict2nifti(datadict,remap,outdir,outprefix=''):
    '''

    No return; runs slRSA2nifti on dictionary of slRSA data files (1subbrick), saving each file based upon its dict key

    datadict: dictionary of pymvpa datasets
    remap: dataset to remap to
    outdir: target directory to save nifti output
    outprefix: optional; prefix string to outfile name

    TO DO: make call afni for 3ddtest...
    '''

    os.mkdir(outdir) #make directory to store data
    for key,ds in datadict.iteritems():
        print('Writing nifti for subject: %s' % (key))
        slRSA2nifti(ds,remap,os.path.join(outdir,'%s%s.nii.gz' % (outprefix,key)))
        print('NIfTI successfully saved: %s' % (os.path.join(outdir,'%s%s.nii.gz' % (outprefix,key))))

def omit_targets(ds,omit):
    '''
    Returns ds with specified targets omitted

    ds: pymvpa dataset with targets
    omit: list of targets to be omitted
    '''

    for om in omit:
        ds = ds[ds.sa.targets != om]
    return ds

###############################################
# Runs slClass for 1 subject # make to take kNN and halfpartiitoner, nfoldpartitioner, different clf
###############################################

def slSVM_1Ss(ds, omit=[], radius=3):
    '''

    Executes slSVM on single subjects and returns ?avg accuracy per voxel?

    ds: pymvpa dsets for 1 subj
    omit: list of targets omitted from pymvpa datasets
    radius: sl radius, default 3
    '''        

    if __debug__:
        debug.active += ["SLC"]

    #dataprep
    remapper = ds.copy()
    inv_mask = ds.samples.std(axis=0)>0
    sfs = StaticFeatureSelection(slicearg=inv_mask)
    sfs.train(remapper)
    ds = remove_invariant_features(ds)
    ds = omit_targets(ds,omit)
    print('Target |%s| omitted from analysis' % (omit))

    print('Beginning sl classification analysis...')
    #clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
    clf = LinearCSVMC()
    #cv=CrossValidation(clf,HalfPartitioner(), enable_ca=['stats'], errorfx=lambda p, t: np.mean(p == t))
    cv=CrossValidation(clf,NFoldPartitioner(), enable_ca=['stats'], errorfx=lambda p, t: np.mean(p == t))
    sl = sphere_searchlight(cv, radius=radius, postproc=mean_sample())
    slr = sl(ds)
    return sfs.reverse(slr).samples - (1.0/len(ds.UT))
   

##############################################
# Runs group level slRSA with defined model
###############################################

def slSVM_nSs(data, omit=[], radius=3, h5 = 0, h5out = 'slSVM_nSs.hdf5'):
    '''

    Executes slSVM per subject in datadict (keys=subjIDs), returns ?avg accuracy per voxel?

    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    omit: list of targets omitted from pymvpa datasets
    radius: sl radius, default 3
    h5: 1 if you want to save hdf5 as well
    h5out: hdf5 outfilename
    '''        
    
    print('slSVM initiated with...\n Ss: %s\nomitting: %s\nradius: %s\nh5: %s\nh5out: %s' % (data.keys(),omit,radius,h5,h5out))

    ### slRSA per subject ###
    slrs={} #dictionary to hold slSVM reuslts per subj
    print('Beginning group level searchlight on %s Ss...' % (len(data)))
    for subjid,ds in data.iteritems():
        print('\Running slSVM for subject %s' % (subjid))
        subj_data = slSVM_1Ss(ds,omit)
        slrs[subjid] = subj_data
    print('slSVM complete for all subjects')

    if h5==1:
        h5save(h5out,slrs,compression=9)
        return slrs
    else: return slrs


####################################
#  Setup - will delete this later, but good for texting
####################################
homedir = '/home/freeman_lab/fMRI/Stolier/facecat'
#dpath = 'analysis/OFC_rFF_masked.hdf5'
#maskpath = 'masks/OFCrFG.nii.gz'
#remappath = 'analysis/ROI_slSVM_remap.nii.gz'
dpath = 'prep/standard_13.11/facecat_allsubjs_masked.gzipped.hdf5'
maskpath = 'masks/avg_mask_d3_BEST.nii.gz'
remappath = 'analysis/data_remap_159.nii.gz'

#load data
data = h5load(os.path.join(homedir,dpath))
print('\nLoading data from %s ...' % (os.path.join(homedir,dpath)))
for ds in data:
    print('Subj %s ; shape:' % (ds), data[ds].shape)

#load remap ds
remap = fmri_dataset(os.path.join(homedir,remappath),mask=os.path.join(homedir,maskpath))
print('Remap dataset loaded from %s, with shape:' % (os.path.join(homedir,remappath)),remap.shape)

print('\nAnalysis materials successfully loaded')
