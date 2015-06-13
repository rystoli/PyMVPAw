#version: 11_Nov_14

from mvpa2.suite import *
import os
import pylab as pylab
import numpy as np
import rsa as rsa
from scipy.spatial.distance import pdist, squareform
import pcorr #for partial rsa
from dset_manage import *

##################################################
# currently in progress of making this module with slRSA functions...

# TO DO: 

#*****   allow slRSA functions kwargs to use overlap_mgs instead of default mgs; issue may be that overlap_msg() requires omit submitted directly to it? nope, not issue just need to include?

#move dataloading etc. to end in __main__; make dsms refresh take arguments

# allow people to not use 'omit' argument

##################################################

def countDig(dig,arr):
    '''Counts value 'dig' in 1D iterable arr'''
    count = 0
    for i in arr:
        if i==dig: count+=1
    return count

##################################################
# data/dsm prep
##################################################

def dsms_refresh(homedir,dsmspath):
    '''
    Refreshes hdf5 file with dsms from srmtfmri_dsms.py B
    '''
    import srmtfmri_dsms
    dsms_fc={'apriori': srmtfmri_dsms.apriori, 'sexIdio': srmtfmri_dsms.sexIdio, 'raceIdio': srmtfmri_dsms.raceIdio, 'colorIdio': srmtfmri_dsms.colorIdio, 'sexIdioAvg': srmtfmri_dsms.idioAvg_sr['sex'], 'raceIdioAvg': srmtfmri_dsms.idioAvg_sr['race'], 'colorIdioAvg': srmtfmri_dsms.idioAvg_sr['color']}
    h5save('dsms_sr.hdf5',dsms_fc,compression=9)
    return h5load(os.path.join(homedir,dsmspath))

def overlap_mgs(ds,mgslist,omit):
    '''
    Returns pyMVPA dataset where samples are mean neural patterns per target category, where target categories can overlap - e.g., male=whitemale+blackmale, white=whitemale+whitefemale, etc.
    
    ds: single pymvpa dataset with targets already defined, and additional sample attributes identifying higher level target group memberships (eg, where 'target'=whitemale, 'sex'=male & 'race'=white
    mgslist: list of sample attribute names for higher level target groups (those which overlap
    omit: list of targets to be omitted from analyses
    
    NOTE: assumes mgs puts things in alphabetical order...
    '''

    #omissions
    for om in omit:
        ds = ds[ds.sa.targets != om] # cut out omits
        print('Target |%s| omitted from analysis' % (om))

    #create mean neural pattern per new target
    premerge = {} # list of mgs results to be merged into new ds
    for mgs in mgslist:
        premerge[mgs] = mean_group_sample([mgs])(ds)
        premerge[mgs].sa['targets'] = np.unique(premerge[mgs].sa[mgs]) #set correct targets

    nds = vstack(premerge.values()) #new dataset
    return mean_group_sample(['targets'])(nds)

def clust2mask(clusts_infile, clustkeys, savenifti = False, outprefix = ''):
    '''
    Returns dict of cluster maks arrays per specified clusters/indices via afni cluster output; can also save these as separate nifti files to be used with fmri_dataset

    clusts_infile = name of nifti file from afni where each cluster is saved as unique integer
    clustkeys: dict of clusters to be separated out and saved, where keys are indices and values are names of ROIs (used in filenaming)
    savenifti: default False, saves each clustmask as separate nifti file
    outprefix: prefix for outfile name
    '''

    clusts = fmri_dataset(clusts_infile)
    clustmasks = {}
    for i,clust in clustkeys.iteritems():
        clustmasks[clust] = clusts.samples[0] == i
        if savenifti == True: slRSA2nifti(clusts.samples[0]==i,clusts,'%s%s.nii.gz' % (outprefix,clust))
    return clustmasks

###############################################
# Data output functions
###############################################

def ndsmROI(ds,cmaskfile,omit,cmaskmaskfile=None,dsm2csv=False,csvname=None,mgslist=None):
    '''
    Returns neural DSM of ROI specified via mask file
    
    ds: pymvpa dataset
    cmaskfile: nifti filename of ROI mask
    omit: targets to be omitted from DSM
    cmaskmaskfile: mask to be applied to ROI mask to make same size as neural ds; default None
    dsm2csv: saves dsm as csv if True; default False
    csvname: outfile name for csv; defualt None
    mgslist: if want to use overlap_mgs, set list here
    '''

    cmask = fmri_dataset(cmaskfile,mask=cmaskmaskfile)
    ds_masked = ds
    ds_masked.samples *= cmask.samples
    ds_masked = remove_invariant_features(ds_masked)
    for om in omit:
        ds_masked = ds_masked[ds_masked.sa.targets != om] # cut out omits
        print('Target |%s| omitted from analysis' % (om))
    if mgslist == None: ds_maskedUT = mean_group_sample(['targets'])(ds_masked) #make UT ds
    elif mgslist != None:
        print('overlap used instead')
        ds_maskedUT = overlap_mgs(ds_masked,mgslist,omit)
    print('shape:',ds_maskedUT.shape)
    ndsm = squareform(pdist(ds_maskedUT.samples,'correlation'))
    if dsm2csv == True: np.savetxt(csvname, ndsm, delimiter = ",")
    return ndsm

def roi2ndsm_nSs(data, cmaskfiles, omit, mask, h5=0, h5name = 'ndsms_persubj.hdf5',mgslist=None):
    '''
    Runs ndsmROI on data dictionary of subjects, for each ROI maskfile specified

    data: datadict, keys subjids, values pymvpa dsets
    cmaskfiles: list of cluster mask ROI file names +.nii.gz
    omit: targets omitted
    mask: mask filename +.nii.gz to equate datasets
    h5: 1 saves full dict of dsms per ROI per subj as hdf5, default 0
    h5name: name for hdf5
    mgslist: if want to use overlap_mgs, set list here

    to do: make call of ndsmROI more flexible
    '''

    data = h5load(os.path.join(homedir,dpath))
    ndsms_persubj = {}
    for ds in data:
        ndsms_persubj[ds]={}
        print('Starting ndsms for subj: %s' % (ds))
        for cmfile in cmaskfiles:
            temp_ds = data[ds].copy()
            ndsms_persubj[ds][cmfile] = ndsmROI(temp_ds,cmfile,['omnF','pro'],mask,dsm2csv=True,csvname='%s_%s.csv' % (ds,cmfile),mgslist=mgslist)
            print('ndsm added to dict and saved as %s_%s.csv' % (ds,cmfile))
    if h5==1: h5save('ndsms_persubj.hdf5',ndsms_persubj,compression=9)
    return ndsms_persubj

def avg_ndsms(ndsms,cmaskfiles,h5=0,h5fname='avg_nDSMs.hdf5',savecsv=0):
    '''
    Returns dict of average nDSM per ROI in output of roi2ndsm_nSs - cmaskfiles - list of cmaksfiles
    '''
    avg_ndsms = dict([(cmask,np.mean([ndsms[subj][cmask] for subj in ndsms],axis=0)) for cmask in cmaskfiles])
    if h5==1: h5save(h5fname,avg_ndsms,compression=9)
    for dsm in avg_ndsms:
        if savecsv == 1: np.savetxt('%s_avg_ndsm.csv' % (dsm), avg_ndsms[dsm], delimiter = ",")
    return avg_ndsms

###############################################
# Runs slRSA with defined model for 1 subject
###############################################

def slRSA_m_1Ss(ds, model, omit, partial_dsm = None, radius=3, cmetric='pearson'):
    '''one subject

    Executes slRSA on single subjects and returns tuple of arrays of 1-p's [0], and fisher Z transformed r's [1]

    ds: pymvpa dsets for 1 subj
    model: model DSM to be correlated with neural DSMs per searchlight center
    partial_dsm: model DSM to be partialled out of model-neural DSM correlation
    omit: list of targets omitted from pymvpa datasets
    radius: sl radius, default 3
    cmetric: default pearson, other optin 'spearman'
    '''        

    if __debug__:
        debug.active += ["SLC"]

    for om in omit:
        ds = ds[ds.sa.targets != om] # cut out omits
        print('Target |%s| omitted from analysis' % (om))
    ds = mean_group_sample(['targets'])(ds) #make UT ds
    print('Mean group sample computed at size:',ds.shape,'...with UT:',ds.UT)

    print('Beginning slRSA analysis...')
    if partial_dsm == None: tdcm = rsa.TargetDissimilarityCorrelationMeasure(squareform(model), comparison_metric=cmetric)
    elif partial_dsm != None: tdcm = rsa.TargetDissimilarityCorrelationMeasure(squareform(model), comparison_metric=cmetric, partial_dsm = squareform(partial_dsm))
    sl = sphere_searchlight(tdcm,radius=radius)
    slmap = sl(ds)
    if partial_dsm == None:
        print('slRSA complete with map of shape:',slmap.shape,'...p max/min:',slmap.samples[0].max(),slmap.samples[0].min(),'...r max/min',slmap.samples[1].max(),slmap.samples[1].min())
        return 1-slmap.samples[1],np.arctanh(slmap.samples[0])
    else:
        print('slRSA complete with map of shape:',slmap.shape,'...r max/min:',slmap.samples[0].max(),slmap.samples[0].min())
        return 1-slmap.samples[1],np.arctanh(slmap.samples[0])
    

###############################################
# Runs group level slRSA with defined model
###############################################

def slRSA_m_nSs(data, model, omit, radius=3, partial_dsm = None, cmetric = 'pearson', h5 = 0, h5out = 'slRSA_m_nSs.hdf5'):
    '''

    Executes slRSA per subject in datadict (keys=subjIDs), returns dict of avg map fisher Z transformed r's and 1-p's per voxel arrays

    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    model: model DSM to be correlated with neural DSMs per searchlight center
    partial_dsm: model DSM to be partialled out of model-neural DSM correlation
    omit: list of targets omitted from pymvpa datasets
    radius: sl radius, default 3
    cmetric: default pearson, other optin 'spearman'
    h5: 1 if you want to save hdf5 as well
    h5out: hdf5 outfilename
    '''        
    
    print('Model slRSA initiated with...\n Ss: %s\nmodel shape: %s\nomitting: %s\nradius: %s\nh5: %s\nh5out: %s' % (data.keys(),model.shape,omit,radius,h5,h5out))

    ### slRSA per subject ###
    slr={'p': {}, 'r':{}} #dictionary to hold fzt r's and 1-p's
    print('Beginning group level searchlight on %s Ss...' % (len(data)))
    for subjid,ds in data.iteritems():
        print('\nPreparing slRSA for subject %s' % (subjid))
        subj_data = slRSA_m_1Ss(ds,model,omit,partial_dsm=partial_dsm,cmetric=cmetric)
        if partial_dsm == None: slr['p'][subjid],slr['r'][subjid] = subj_data[0],subj_data[1]
        else: slr['r'][subjid] = subj_data
    print('slRSA complete for all subjects')

    if h5==1:
        h5save(h5out,slr,compression=9)
        return slr
    else: return slr
    
###############################################
# idiosyncratic slRSA: DM per subject
###############################################

def idio_slRSA(datadict,idioDMs,omit=[],partial_dsm=None,cmetric='spearman',h5name='out.hdf5'):
    '''
    
    Returns avg map of corrs per subject, using unique DM per subject, specified in dict of DMs
    
    datadict: dict of subject data, subject as keys
    idioDMs: dict of DM per subject, subject as keys
    omit: list of targets to omit
    partial_dsm: DM to partial out
    cmetric: yaknow
    h5name: yaknow
    '''
    
    idio = {}
    for s,dm in idioDMs.iteritems():
        idio[s] = slRSA_m_1Ss(datadict[s],dm,omit,partial_dsm=partial_dsm,cmetric=cmetric)
    h5save(h5name,idio,compression=9)
    return idio
    
###############################################
# across subjects RSA
###############################################

def slRSA_xSs(data,omit,measure='DCM',radius=3,h5=0,h5out='slRSA_xSs.hdf5'):
    '''
    
    Returns avg map of xSs correlations of neural DSMs per searchlight sphere center
    - uses either Dissimilarity Consistency Measure or RSMMeasure, recommended DCM, as of now RSM seems unusual

    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    omit: list of targets omitted from pymvpa datasets
    radius: sl radius, default 3
    measure: specify if measure is 'DCM' or 'RSM'
    h5: 1 saves hdf5 of output as well 
    h5out: hdf5 outfilename
    
    TO DO: should not return average, but full ds of pairs? - need to split ds's into different keys in datadict
    '''   

    print('%s xSs slRSA initiated with...\n Ss: %s\nomitting: %s\nradius: %s\nh5: %s\nh5out: %s' % (measure,data.keys(),omit,radius,h5,h5out))

    if __debug__:
        debug.active += ["SLC"]
    
    for i in data:
        data[i] = mean_group_sample(['targets'])(data[i]) 
    print('Dataset targets averaged with shapes:',[ds.shape for ds in data.values()])

    #omissions
    print('Omitting targets: %s from data' % (omit))
    for i in data:
        for om in omit:
            data[i] = data[i][data[i].sa.targets != om]

    if measure=='DCM':
        group_data = None
        for s in data.keys():
             ds = data[s]
             ds.sa['chunks'] = [s]*len(ds)
             if group_data is None: group_data = ds
             else: group_data.append(ds)
        print('Group dataset ready including Ss: %s\nBeginning slRSA:' % (np.unique(group_data.chunks)))
        dcm = rsa.DissimilarityConsistencyMeasure()
        sl_dcm = sphere_searchlight(dcm,radius=radius)
        slmap_dcm = sl_dcm(group_data)
        print('Analysis complete with shape:',slmap_dcm.shape)
        if h5 == 1:
            h5save(h5out,slmap_dcm,compression=9)
            return slmap_dcm
        else: return slmap_dcm

################################
# Classification


# to do
# make omit optional.... #Is this not already done?
#chance_level = 1.0 - (1.0 / len(ds.uniquetargets))
# need to set targets beforehand if swithcing what they are

###############################################
# Runs slClass for 1 subject # make to take kNN and halfpartiitoner, nfoldpartitioner, different clf
###############################################

def slClass_1Ss(ds, omit=[], radius=3, clf = LinearCSVMC(), part = NFoldPartitioner(), scale_result=False):
    '''

    Executes slClass on single subjects and returns ?avg accuracy per voxel?

    ds:             pymvpa dsets for 1 subj
    omit:           list of targets omitted from pymvpa datasets
    radius:         sl radius, default 3
    clf:            specify classifier
    part:           specify partitioner
    scale_result:   if True, scales the voxels that guess at chance to be 0
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
    cv=CrossValidation(clf, part, enable_ca=['stats'], errorfx=lambda p, t: np.mean(p == t))
    sl = sphere_searchlight(cv, radius=radius, postproc=mean_sample())
    slr = sl(ds)
    if scale_result:
        return sfs.reverse(slr).samples - (1.0/len(ds.UT))
    else:
        return sfs.reverse(slr).samples
   

##############################################
# Runs group level slClass
###############################################

def slClass_nSs(data, omit=[], radius=3, clf = LinearCSVMC(), part = NFoldPartitioner(), h5 = 0, h5out = 'slSVM_nSs.hdf5'):
    '''

    Executes slClass per subject in datadict (keys=subjIDs), returns ?avg accuracy per voxel?

    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    omit: list of targets omitted from pymvpa datasets
    radius: sl radius, default 3
    clf: specify classifier
    part: specify partitioner
    h5: 1 if you want to save hdf5 as well
    h5out: hdf5 outfilename
    '''        
    
    print('slClass initiated with...\n Ss: %s\nomitting: %s\nradius: %s\nh5: %s\nh5out: %s' % (data.keys(),omit,radius,h5,h5out))

    ### slClass per subject ###
    slrs={} #dictionary to hold slSVM reuslts per subj
    print('Beginning group level searchlight on %s Ss...' % (len(data)))
    for subjid,ds in data.iteritems():
        print('\Running slClass for subject %s' % (subjid))
        subj_data = slClass_1Ss(ds,omit,radius,clf,part)
        slrs[subjid] = subj_data
    print('slClass complete for all subjects')

    if h5==1:
        h5save(h5out,slrs,compression=9)
        return slrs
    else: return slrs


#############################################
# Runs SampleBySampleSimilarityCorrelation through searchlight
#############################################

def slSxS_1Ss(ds, comparison_sample, sample_covariable, omit = [], radius = 3, h5 = 0, h5out = 'slSxS.hdf5'):
    '''

    Executes searchlight SampleBySampleSimilarityCorrelation, returns corr coef (and optional p value) per voxel

    
    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    comparison_sample:  Name of the sample (sa.targets target name) to be compared with each individual sample. Mean of this will be used. 
    sample_covariable:  Name of the variable (sample attribute) with a value for each sample. The distance of each sample with the comparison_sample will be correlated with this variable.
    omit: list of targets omitted from pymvpa datasets; VERY IMPORTANT TO GET THIS RIGHT, should omit typically all targets besides the target of interest, and comparison_sample.
    radius: sl radius, default 3
    h5: 1 if you want to save hdf5 as well
    h5out: hdf5 outfilename
    
    TO DO: probably better way to keep wanted targets in dset and omit others with having to specify omits...
    '''    
   
    if __debug__:
        debug.active += ["SLC"]

    for om in omit:
        ds = ds[ds.sa.targets != om] # cut out omits
        print('Target |%s| omitted from analysis' % (om))
 
    print('Beginning slSxS analysis...')
    SxS = rsa.SampleBySampleSimilarityCorrelation(comparison_sample,sample_covariable)
    sl = sphere_searchlight(SxS,radius=radius)
    slmap = sl(ds)

    print('slSxS complte with map of shape:',slmap.shape,'...p max/min:',slmap.samples[0].max(),slmap.samples[0].min(),'...r max/min',slmap.samples[1].max(),slmap.samples[1].min())
    
    #change slmap to right format
    slmap.samples[0],slmap.samples[1]=np.arctanh(slmap.samples[0]),1-slmap.samples[1]    
    h5save(h5out,slmap,compression=9)
    print('h5 saved as:',h5out)

    return slmap    
#    return slmap #not transforming to Zr or 1-p    
#    if h5==1:
#        h5save(h5out,np.array(1-slmap.samples[1],np.arctanh(slmap.samples[0])),compression=9)
#        return np.array(1-slmap.samples[1],np.arctanh(slmap.samples[0]))
#    else: return np.array(1-slmap.samples[1],np.arctanh(slmap.samples[0]))


##############################################
# SxS group 
###############################################

def slSxS_nSs(data, targs_comps, sample_covariable, omit=[], radius=3, h5 = 0, h5out = 'slSxS_nSs.hdf5'):
    '''

    Executes searchlight SampleBySampleSimilarityCorrelation, returns corr coef (and optional p value) per voxel

    ***assumes anything not in targs_comps is omitted***

    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    comparison_sample:  Dict of SxS targets (keys) and comparison targets (values)
    sample_covariable:  Name of the variable (sample attribute) with a value for each sample. The distance of each sample with the comparison_sample will be correlated with this variable.
    omit: list of targets omitted from pymvpa datasets; VERY IMPORTANT TO GET THIS RIGHT, should omit typically all targets besides the target of interest, and comparison_sample.
    radius: sl radius, default 3
    h5: 1 if you want to save hdf5 as well
    h5out: hdf5 outfilename
    '''        
    
    print('slSxS initiated with...\n Ss: %s\ncomparison sample: %s\nsample covariable: %s\nomitting: %s\nradius: %s\nh5: %s\nh5out: %s' % (data.keys(),targs_comps,sample_covariable,omit,radius,h5,h5out))

    ### slSxS per subject ###
    slrs={} #dictionary to hold reuslts per subj
    print('Beginning group level searchlight on %s Ss...' % (len(data)))
    for subjid,ds in data.iteritems():
        print('\Running slSxS for subject %s' % (subjid))
        subj_data = slSxS_1Ss(ds,targs_comps,sample_covariable,omit,radius)
        slrs[subjid] = subj_data
    print('slSxS complete for all subjects')

    if h5==1:
        h5save(h5out,slrs,compression=9)
        return slrs
    else: return slrs


###############################################
# BDSM 
###############################################

def slBDSM_xSs(data,xSs_behav,targ_comp,radius=3,h5=0,h5out='bdsm_xSs.hdf5'):
    '''
    
    Returns correlation of subject-level behav sim with subject-level neural sim between two targs

    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    xSs_behav: Dictionary of behavioral value between subjects to be
               correlated with intrasubject neural similarity (subjects are keys)
    targ_comp: List of targets whose similarity is correlated with xSs_behav
    radius: sl radius, default 3
    h5: 1 saves hdf5 of output as well 
    h5out: hdf5 outfilename
    
    '''   

    print('xSs BDSM initiated with...\n Ss: %s \n targ_comp: %s\nradius: %s\nh5: %s\nh5out: %s' % (data.keys(),targ_comp,radius,h5,h5out))

    if __debug__:
        debug.active += ["SLC"]
    
    for i in data:
        data[i] = mean_group_sample(['targets'])(data[i]) 
    print('Dataset targets averaged with shapes:',[ds.shape for ds in data.values()])

    group_data = None
    for s in data.keys():
         ds = data[s]
         ds.sa['chunks'] = [s]*len(ds)
         if group_data is None: group_data = ds
         else: group_data.append(ds)
    print('Group dataset ready including Ss: %s\nBeginning slBDSM:' % (np.unique(group_data.chunks)))
    bdsm = rsa_adv.xss_BehavioralDissimilarity(xSs_behav,targ_comp)
    sl_bdsm = sphere_searchlight(bdsm,radius=radius)
    slmap_bdsm = sl_bdsm(group_data)
    print('Analysis complete with shape:',slmap_bdsm.shape)
    if h5 == 1:
        h5save(h5out,slmap_bdsm,compression=9)
        return slmap_bdsm
    else: return slmap_bdsm

##############################################
# BDSM ROI
###############################################

def roiBDSM_xSs(data, xSs_behav, targ_comp, roi_mask_nii_path, h5 = 0,h5out = 'bdsm_roi.hdf5'):
    '''
    
    Returns correlation of subject-level behav sim with subject-level neural sim between two targs

    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    xSs_behav: Dictionary of behavioral value between subjects to be
               correlated with intrasubject neural similarity (subjects are keys)
    targ_comp: List of targets whose similarity is correlated with xSs_behav
    roi_mask_nii_path: Nifti file location of binary  mask for ROI
    h5: 1 saves hdf5 of output as well 
    h5out: hdf5 outfilename
    
    '''

    print('xSs BDSM initiated with...\n Ss: %s \n targ_comp: %s\nroi_mask: %s\nh5: %s\nh5out: %s' % (data.keys(),targ_comp,roi_mask_nii_path,h5,h5out))

    for i in data:
        data[i] = mean_group_sample(['targets'])(data[i])
    print('Dataset targets averaged with shapes:',[ds.shape for ds in data.values()])

    group_data = None
    for s in data.keys():
         ds = data[s]
         ds.sa['chunks'] = [s]*len(ds)
         if group_data is None: group_data = ds
         else: group_data.append(ds)
    print('Group dataset ready including Ss: %s\nBeginning slBDSM:' % (np.unique(group_data.chunks)))
    group_data_m = mask_dset(group_data,roi_mask_nii_path)
    print('Group dataset masked, to size: %s' % (str(group_data_m.shape)))

    bdsm = rsa_adv.xss_BehavioralDissimilarity(xSs_behav,targ_comp)
    roi_bdsm = bdsm(group_data_m)
    bdsmr = roi_bdsm.samples[0][0]
    print('Analysis complete with r: %s' % (str(bdsmr)))

    if h5 == 1:
        h5save(h5out,bdsmr,compression=9)
        return bdsmr
    else: return bdsmr


###############################################
# BDSM ROI double
###############################################

def roiBDSM_xSs_d(data,xSs_behav1,targ_comp1,xSs_behav2,targ_comp2,roi_mask_nii_path,h5=0,h5out='bdsm_xSs.hdf5'):
    '''
    
    Returns correlation of subject-level behav sim with subject-level neural sim between two targs

    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    xSs_behav: Dictionary of behavioral value between subjects to be
               correlated with intrasubject neural similarity (subjects are keys)
    targ_comp: List of targets whose similarity is correlated with xSs_behav
    roi_mask_nii_path: path to nifti mask file for ROI
    h5: 1 saves hdf5 of output as well 
    h5out: hdf5 outfilename
    
    '''

    print('xSs BDSM initiated with...\n Ss: %s \n targ_comp1: %s\n targ_comp2: %s\n mask_roi: %s\nh5: %s\nh5out: %s' % (data.keys(),targ_comp1,targ_comp2,roi_mask_nii_path,h5,h5out))

    for i in data:
        data[i] = mean_group_sample(['targets'])(data[i])
    print('Dataset targets averaged with shapes:',[ds.shape for ds in data.values()])

    group_data = None
    for s in data.keys():
         ds = data[s]
         ds.sa['chunks'] = [s]*len(ds)
         if group_data is None: group_data = ds
         else: group_data.append(ds)
    print('Group dataset ready including Ss: %s\nBeginning slBDSM:' % (np.unique(group_data.chunks)))

    group_data_m = mask_dset(group_data,roi_mask_nii_path)
    print('Group dataset masked, to size: %s' % (str(group_data_m.shape)))

    bdsm = rsa_adv.xss_BehavioralDissimilarity_double(xSs_behav1,targ_comp1,xSs_behav2,targ_comp2)
    roi_bdsm = bdsm(group_data_m)
    bdsmr = roi_bdsm.samples[0][0]
    print('Analysis complete with r: %s' % (str(bdsmr)))

    if h5 == 1:
        h5save(h5out,bdsmr,compression=9)
        return bdsmr
    else: return bdsmr


#########################################
# load subj data
#########################################


def load_subj_data(study_dir, subj_list, file_suffix='.nii.gz', attr_filename=None, remove_invariants=False, hdf5_filename=None, mask=None):
    ''' Loads in subject files and stores them in a data_dict.

        Function returns [ {data}, and [Sample Attributes] ]
        Keys    in the data_dict are the    subject filenames
        Values  in the data_dict are the    pymvpa datasets

        Keyword arguments:
        study_dir           -- Study directory (should contain fMRI and attr files)
        subj_list           -- List of subject IDs
        file_suffix         -- What to add to subject IDs to complete filename
        attr_filename       -- Filename for Sample Attributes within study dir
        remove_invariants   -- Remove invariant features
        hdf5_filename       -- If not none, saves hdf5 output with this name
        mask                -- Specify an ROI mask for the data
    '''


    data_dict = {}
    flags = {}
    if attr_filename != None:
        print( "Loading Sample Attributes file" )
        attr_filepath   =   os.path.join( study_dir, attr_filename )
        attr=SampleAttributes( os.path.join( study_dir, attr_filename ) )
        flags['targets'] = attr.targets
        flags['chunks']  = attr.chunks
        print( "Done\n" )

    if mask != None:
        flags['mask']    = mask

    for subj in subj_list:
        subj_filename = ''.join(( subj, file_suffix) )
        subj_filepath = os.path.join( study_dir, subj_filename  )
        print( 'loading subject file: %s'   %   subj_filename   )
        print( 'from: %s\n'                 %   study_dir       )


        data_dict[subj] = fmri_dataset( subj_filepath, **flags )

        if remove_invariants:
            print( 'Removing invariant features' )
            data_dict[subj] = data_dict[subj].remove_invariant_features()
            print( 'Done\n' )

    print( 'Subject data successfully loaded\n' )

    if hdf5_filename != None:
        hdf5_filename = os.path.join( study_dir, hdf5_filename )
        print( 'Saving hdf5 file: %s' % hdf5_filename )
        h5save( hdf5_filename, data_dict, compression=9 )
        print( 'Done\n' )

    return data_dict


#############################################
# Functions for classificaiton of ROIs
#############################################

def stdev_percat(cataccs):
    '''
	Returns dict of stdev per category b/w subjects
	
	cataccs: input is list of ACC per cat per subject
	'''
    accpercat,stdpercat={},{}
    for i in cataccs[0].keys():
        accpercat[i] = [cataccs[j][i] for j in range(len(cataccs))]
    for i in accpercat.keys():
        stdpercat[i] = np.std(accpercat[i])
    return stdpercat

def acc_percat(c_sum):
    '''
	Returns dict of ACC per category
	
	c_sum: summed confusion matrix across subject
	'''
    labels = c_sum.labels
    #pulls in all cat rates
    TPA,PA = c_sum.stats["TP"],c_sum.stats["P"]
    ACC = {}
    for i in range(len(labels)):
        TP,P = float(TPA[i]),float(PA[i])
        ACC[labels[i]]=(TP/P)
    return ACC

#classification 
def classify(data):
	'''
	Returns results from classification of dataset by all targets in dataset
	
	data: dictionary of data, keys are subject Ids, values their datasets
	
	***NEED TO MAKE ARGS FOR CLASSIFIERS ETC***
	'''
    clf = LinearCSVMC([-1])
    #clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
    confusions=[]
    cataccs=[]
    for ds in data.itervalues():
        cv = CrossValidation(clf,NFoldPartitioner(), enable_ca=['stats'])
        errors = cv(ds)
        confusions.append(cv.ca.stats)
        cataccs.append(acc_percat(cv.ca.stats))
    c_sum = pylab.sum(confusions)
    ACC=acc_percat(c_sum)
    stdev=stdev_percat(cataccs)
    return {'c_sum': c_sum, 'ACC': ACC, 'stdev': stdev}

def class_plotter(classdict,pout):
	'''
	Saves plots as .png files in current directory for both average confusion matrix and results histrograms
	
	classdict: output of classify function above
	pout: output name for picture
	'''
    c_sum,ACC,stdev=classdict['c_sum'],classdict['ACC'],classdict['stdev']
    #make confusion matrices
    c_sum.plot()
    pylab.plt.savefig('%s_confusion.png' % (pout))
    pylab.clf() #clears confusion matrix
    #make histograms
    X,ymax = np.arange(len(ACC)),1
    pylab.bar(X, ACC.values(), align='center', width=0.5, yerr=stdev.values(), ecolor="red")
    pylab.xticks(X, ACC.keys())
    pylab.ylim(0, ymax)
    pylab.plt.savefig('%s_hist.png' % (pout),format='png')
    pylab.clf() #clears hist
    pylab.close()

def normalize(data,method = 'featureByChunk'):
	'''
	Returns dataset normalized
	
	data = dictionary with data per subject, key is subject ID
	method = specific normalization method: ['featureByChunk','featureZ','featureMD','sampleZ','sampleMD']
	'''
    if method == 'featureByChunk': 
		for s in data:
			zscore(data[s])
	elif method == 'featureZ':
		for s in data:
			data[s].samples = (data[s].samples - np.mean(data[s],axis=0))/np.std(data[s],axis=0)
	elif method == 'featureMD':
		for s in data:
			data[s].samples = data[s].samples - np.mean(data[s],axis=0)
	elif method == 'sampleMD':
		for s in data:
			for i in range(len(data[s])):
				data[s].samples[i] = (data[s][i].samples - np.mean(data[s][data[s].sa.targets == data[s][i].targets]))
	elif method == 'sampleZ':
		for s in data:
			for i in range(len(data[s])):
				data[s].samples[i] = (data[s][i].samples - np.mean(data[s][data[s].sa.targets == data[s][i].targets]))/np.std(data[s][data[s].sa.targets == data[s][i].targets])
	return data
