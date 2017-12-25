#version 8/19/15
from importer import *
#searchlight wrappers


    


###############################################
# Runs slRSA with defined model for 1 subject
###############################################

def slRSA_m_1Ss(ds, model, partial_dsm = None, control_dsms = None, resid = False, radius=3, cmetric='pearson',status_print=1):
    '''

    Executes slRSA on single subjects and returns tuple of arrays of 1-p's [0], and fisher Z transformed r's [1]

    ds: pymvpa dsets for 1 subj
    model: model DSM to be correlated with neural DSMs per searchlight center
    partial_dsm: DSM to be partialled out of model-neural DSM correlation
    control_dsms: list of DSMs to be controlled for in multiple regression 
                  which returns r of model DM predictor (converted from beta)                  
    resid: Default False. Set to True to return residual to searchlight center
    radius: sl radius, default 3
    cmetric: default pearson, other optin 'spearman'
    status_print: if 1, prints status of searchlight (progress)
    '''        

    if partial_dsm != None and control_dsms != None: raise NameError('Only set partial_dsm (partial model control) OR control_dsms (multiple regression model controls)')

    if status_print == 1:
        if __debug__: debug.active += ["SLC"]
    else: pass

    ds = mean_group_sample(['targets'])(ds) #make UT ds
    print('Mean group sample computed at size:',ds.shape,'...with UT:',ds.UT)

    print('Beginning slRSA analysis...')
    if partial_dsm == None and control_dsms == None: tdcm = rsa_pymvpaw.TargetDissimilarityCorrelationMeasure_Partial(squareform(model), comparison_metric=cmetric)
    elif partial_dsm != None and control_dsms == None: tdcm = rsa_pymvpaw.TargetDissimilarityCorrelationMeasure_Partial(squareform(model), comparison_metric=cmetric, partial_dsm = squareform(partial_dsm))
    elif partial_dsm == None and control_dsms != None: tdcm = rsa_pymvpaw.TargetDissimilarityCorrelationMeasure_Regression(squareform(model), comparison_metric=cmetric, control_dsms = [squareform(dm) for dm in control_dsms], resid = resid)
    sl = sphere_searchlight(tdcm,radius=radius)
    slmap = sl(ds)
    if partial_dsm == None and control_dsms == None:
        print('slRSA complete with map of shape:',slmap.shape,'...p max/min:',slmap.samples[0].max(),slmap.samples[0].min(),'...r max/min',slmap.samples[1].max(),slmap.samples[1].min())
        return 1-slmap.samples[1],np.arctanh(slmap.samples[0])
    elif partial_dsm != None and control_dsms == None:
        print('slRSA complete with map of shape:',slmap.shape,'...r max/min:',slmap.samples[0].max(),slmap.samples[0].min())
        return np.arctanh(slmap.samples[0])
    elif partial_dsm == None and control_dsms != None:
        print('slRSA complete with map of shape:',slmap.shape,'...r max/min:',slmap.samples[0].max(),slmap.samples[0].min())
        return slmap.samples[0]
    

###############################################
# Runs group level slRSA with defined model
###############################################

def slRSA_m_nSs(data, model, radius=3, partial_dsm = None, control_dsms = None, resid = False, cmetric = 'pearson', h5 = 0, h5out = 'slRSA_m_nSs.hdf5',status_print=1):
    '''

    Executes slRSA per subject in datadict (keys=subjIDs), returns dict of avg map fisher Z transformed r's and 1-p's per voxel arrays

    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    model: model DSM to be correlated with neural DSMs per searchlight center
    partial_dsm: model DSM to be partialled out of model-neural DSM correlation
    control_dsms: list of DSMs to be controlled for in multiple regression 
                  which returns r of model DM predictor (converted from beta)                  
    resid: Default False. Set to True to return residual to searchlight center
    radius: sl radius, default 3
    cmetric: default pearson, other optin 'spearman'
    h5: 1 if you want to save hdf5 as well
    h5out: hdf5 outfilename
    '''        
    
    print('Model slRSA initiated with...\n Ss: %s\nmodel shape: %s\nradius: %s\nh5: %s\nh5out: %s' % (data.keys(),model.shape,radius,h5,h5out))

    ### slRSA per subject ###
    slr={'p': {}, 'r':{}} #dictionary to hold fzt r's and 1-p's
    print('Beginning group level searchlight on %s Ss...' % (len(data)))
    for subjid,ds in data.iteritems():
        print('\nPreparing slRSA for subject %s' % (subjid))
        subj_data = slRSA_m_1Ss(ds,model,partial_dsm=partial_dsm,control_dsms=control_dsms,resid=resid,cmetric=cmetric,status_print=status_print)
        if partial_dsm == None and control_dsms == None: slr['p'][subjid],slr['r'][subjid] = subj_data[0],subj_data[1]
        else: slr['r'][subjid] = subj_data
    print('slRSA complete for all subjects')

    if h5==1:
        h5save(h5out,slr,compression=9)
        return slr
    else: return slr
    

###############################################
# Pairsim RSA 
###############################################

def sl_pairsimRSA_1Ss(ds, pairs_dsm, radius=3, cmetric='spearman',status_print=1):
    '''
    Runs standard RSA between a specified model and neural data, 
    but allows specification of exactly which target-pairs are included
    i.e., specification of exactly which DM cells are kept in the analysis

    ds: pymvpa dsets for 1 subj
    pairs_dsm : Dictionary of target pairs separated by '-' (keys) and
                corresponding predicted model dissimilarity values (values)
    cmetric: spearman or pearson or eucldiean
    status_print: if 1, prints status of searchlight (progress)
    '''        

    if status_print == 1:
        if __debug__: debug.active += ["SLC"]
    else: pass
    ds = mean_group_sample(['targets'])(ds) #make UT ds
    print('Mean group sample computed at size:',ds.shape,'...with UT:',ds.UT)

    print('Beginning slRSA analysis...')
    tdcm = rsa_pymvpaw.Pairsim_RSA(pairs_dsm,comparison_metric=cmetric)    
    sl = sphere_searchlight(tdcm,radius=radius)
    slmap = sl(ds)
    return slmap.samples[0]
    

def sl_pairsimRSA_nSs(data, pairs_dsm, radius=3, cmetric = 'pearson', h5 = 0, h5out = 'slRSApairsim_m_nSs.hdf5',status_print=1):
    '''
    Runs standard RSA between a specified model and neural data, 
    but allows specification of exactly which target-pairs are included
    i.e., specification of exactly which DM cells are kept in the analysis

    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    pairs_dsm : Dictionary of target pairs separated by '-' (keys) and
                corresponding predicted model dissimilarity values (values)
    cmetric: pearson or spearman or euclidean
    h5: 1 if you want to save hdf5 as well
    h5out: hdf5 outfilename
    '''        
    
    ### slRSA per subject ###
    slr= {} 
    print('Beginning group level searchlight on %s Ss...' % (len(data)))
    for subjid,ds in data.iteritems():
        print('\nPreparing slRSA for subject %s' % (subjid))
        subj_data = sl_pairsimRSA_1Ss(ds,pairs_dsm,radius=radius,cmetric=cmetric,status_print=status_print)
        slr[subjid] = subj_data
    print('slPairsim complete for all subjects')

    if h5==1:
        h5save(h5out,slr,compression=9)
        return slr
    else: return slr


###############################################
# across subjects RSA
###############################################

def slRSA_xSs(data,measure='DCM',radius=3,h5=0,h5out='slRSA_xSs.hdf5',status_print=1):
    '''
    
    Returns avg map of xSs correlations of neural DSMs per searchlight sphere center
    - uses either Dissimilarity Consistency Measure or RSMMeasure, recommended DCM, as of now RSM seems unusual

    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    radius: sl radius, default 3
    measure: specify if measure is 'DCM' or 'RSM'
    h5: 1 saves hdf5 of output as well 
    h5out: hdf5 outfilename
    status_print: if 1, prints status of searchlight (progress)
    
    TO DO: should not return average, but full ds of pairs? - need to split ds's into different keys in datadict
    '''   

    print('%s xSs slRSA initiated with...\n Ss: %s\nradius: %s\nh5: %s\nh5out: %s' % (measure,data.keys(),radius,h5,h5out))

    if status_print == 1:
        if __debug__: debug.active += ["SLC"]
    else: pass
 
    for i in data:
        data[i] = mean_group_sample(['targets'])(data[i]) 
    print('Dataset targets averaged with shapes:',[ds.shape for ds in data.values()])

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


###############################################
# Runs slClass for 1 subject 
###############################################

def slClass_1Ss(ds, radius=3, clf = LinearCSVMC(), part = NFoldPartitioner(), partmean = 1,status_print=1):
    '''

    Executes slClass on single subjects and returns ?avg accuracy per voxel?

    ds: pymvpa dsets for 1 subj
    radius: sl radius, default 3
    clf: specify classifier
    part: specify partitioner
    partmean: 1 if collapse results across results per fold
    status_print: if 1, prints status of searchlight (progress)
    '''        

    if status_print == 1:
        if __debug__: debug.active += ["SLC"]
    else: pass

    #dataprep
    remapper = ds.copy()
    inv_mask = ds.samples.std(axis=0)>0
    sfs = StaticFeatureSelection(slicearg=inv_mask)
    sfs.train(remapper)
    ds = remove_invariant_features(ds)

    print('Beginning sl classification analysis...')
    cv=CrossValidation(clf, part, enable_ca=['stats'], errorfx=lambda p, t: np.mean(p == t))
    if partmean == 1: sl = sphere_searchlight(cv, radius=radius, postproc=mean_sample())
    elif partmean == 0: sl = sphere_searchlight(cv, radius=radius)
    slr = sl(ds)
    return sfs.reverse(slr).samples - (1.0/len(ds.UT))
   

##############################################
# Runs group level slRSA with defined model
###############################################

def slClass_nSs(data, radius=3, clf = LinearCSVMC(), part = NFoldPartitioner(), partmean = 1, h5 = 0, h5out = 'slSVM_nSs.hdf5',status_print=1):
    '''

    Executes slClass per subject in datadict (keys=subjIDs), returns ?avg accuracy per voxel?

    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    omit: list of targets omitted from pymvpa datasets
    radius: sl radius, default 3
    clf: specify classifier
    part: specify partitioner
    partmean: 1 if collapse results across results per fold
    h5: 1 if you want to save hdf5 as well
    h5out: hdf5 outfilename
    '''        
    
    print('slClass initiated with...\n Ss: %s\nradius: %s\nh5: %s\nh5out: %s' % (data.keys(),radius,h5,h5out))

    ### slClass per subject ###
    slrs={} #dictionary to hold slSVM reuslts per subj
    print('Beginning group level searchlight on %s Ss...' % (len(data)))
    for subjid,ds in data.iteritems():
        print('\Running slClass for subject %s' % (subjid))
        subj_data = slClass_1Ss(ds,radius,clf=clf,part=part,partmean=partmean,status_print=status_print)
        slrs[subjid] = subj_data
    print('slClass complete for all subjects')

    if h5==1:
        h5save(h5out,slrs,compression=9)
        return slrs
    else: return slrs


#############################################
# Runs SampleBySampleSimilarityCorrelation through searchlight
#############################################

def slSxS_1Ss(ds, targs_comps, sample_covariable, omit = [], radius = 3, h5 = 0, h5out = 'slSxS_1Ss.nii.gz',status_print=1):
    '''

    Executes searchlight SampleBySampleSimilarityCorrelation, returns corr coef (and optional p value) per voxel

    
    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    targs_comps: dict of trial by trial targets (keys) and their comparison targets (values) - **assumes non-interest targets omitted***
    sample_covariable:  Name of the variable (sample attribute) with a value for each sample. The distance of each sample with the comparison_sample will be correlated with this variable.
    omit: list of targets omitted from pymvpa datasets; VERY IMPORTANT TO GET THIS RIGHT, should omit typically all targets besides the target of interest, and comparison_sample.
    radius: sl radius, default 3
    h5: 1 if you want to save hdf5 as well
    h5out: hdf5 outfilename
    status_print: if 1, prints status of searchlight (progress)
    
    TO DO: probably better way to keep wanted targets in dset and omit others with having to specify omits...
    '''    
   
    if status_print == 1:
        if __debug__: debug.active += ["SLC"]
    else: pass
 
    print('MAKE SURE you have omitted targets not specified in the analysis!')
    for om in omit:
        ds = ds[ds.sa.targets != om] # cut out omits
        print('Target |%s| omitted from analysis' % (om))
 
    print('Beginning slSxS analysis...')
    SxS = rsa_pymvpaw.SampleBySampleSimilarityCorrelation(targs_comps,sample_covariable)
    sl = sphere_searchlight(SxS,radius=radius)
    slmap = sl(ds)

    print('slSxS complte with map of shape:',slmap.shape,'...p max/min:',slmap.samples[0].max(),slmap.samples[0].min(),'...r max/min',slmap.samples[1].max(),slmap.samples[1].min())
    
    #change slmap to right format
    slmap.samples[0],slmap.samples[1]=np.arctanh(slmap.samples[0]),1-slmap.samples[1]
    h5save(h5out,slmap,compression=9)
    print('h5 saved as:',h5out)

    return slmap    


##############################################
# SxS group 
###############################################

def slSxS_nSs(data, targs_comps, sample_covariable, omit=[], radius=3, h5 = 0, h5out = 'slSxS_nSs.hdf5',status_print=1):
    '''

    Executes searchlight SampleBySampleSimilarityCorrelation, returns corr coef (and optional p value) per voxel

    ***assumes anything not in targs_comps is omitted***

    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    targs_comps: dict of trial by trial targets (keys) and their comparison targets (values) - **assumes non-interest targets omitted***
    sample_covariable:  Name of the variable (sample attribute) with a value for each sample. The distance of each sample with the comparison_sample will be correlated with this variable.
    omit: list of targets omitted from pymvpa datasets; VERY IMPORTANT TO GET THIS RIGHT, should omit typically all targets besides the target of interest, and comparison_sample.
    radius: sl radius, default 3
    h5: 1 if want h5 per subj 
    h5out: h outfilename suffix
    '''        
    
    print('slSxS initiated with...\n Ss: %s\ncomparison sample: %s\nsample covariable: %s\nomitting: %s\nradius: %s\nh5: %s\nh5out: %s' % (data.keys(),targs_comps,sample_covariable,omit,radius,h5,h5out))

    ### slSxS per subject ###
    slrs={} #dictionary to hold reuslts per subj
    print('Beginning group level searchlight on %s Ss...' % (len(data)))
    for subjid,ds in data.iteritems():
        print('\Running slSxS for subject %s' % (subjid))
        subj_data = slSxS_1Ss(ds,targs_comps,sample_covariable,omit,radius,h5,subjid+h5out,status_print=status_print)
        slrs[subjid] = subj_data
    print('slSxS complete for all subjects')

    if h5==1:
        h5save(h5out,slrs,compression=9)
        return slrs
    else: return slrs

##############################################
# BDSM 
###############################################

def slBDSM_xSs(data,xSs_behav,targ_comp,radius=3,h5=0,h5out='bdsm_xSs.hdf5',status_print=1):
    '''
    
    Returns correlation of subject-level behav sim with subject-level neural sim between two targs

    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    xSs_behav: Dictionary of behavioral value between subjects to be
               correlated with intrasubject neural similarity (subjects are keys)
    targ_comp: List of targets whose similarity is correlated with xSs_behav
    radius: sl radius, default 3
    h5: 1 saves hdf5 of output as well 
    h5out: hdf5 outfilename
    status_print: if 1, prints status of searchlight (progress)
    
    '''   

    print('xSs BDSM initiated with...\n Ss: %s \n targ_comp: %s\nradius: %s\nh5: %s\nh5out: %s' % (data.keys(),targ_comp,radius,h5,h5out))

    if status_print == 1:
        if __debug__: debug.active += ["SLC"]
    else: pass
    
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
    bdsm = rsa_pymvpaw.xss_BehavioralDissimilarity(xSs_behav,targ_comp)
    sl_bdsm = sphere_searchlight(bdsm,radius=radius)
    slmap_bdsm = sl_bdsm(group_data)
    print('Analysis complete with shape:',slmap_bdsm.shape)
    if h5 == 1:
        h5save(h5out,slmap_bdsm,compression=9)
        return slmap_bdsm
    else: return slmap_bdsm

###############################################
# BDSM double
###############################################

def slBDSM_xSs_d(data,xSs_behav1,targ_comp1,xSs_behav2,targ_comp2,radius=3,h5=0,h5out='bdsm_xSs.hdf5',status_print=1):
    '''
    
    Returns correlation of subject-level behav sim with subject-level neural sim between two targs

    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    xSs_behav: Dictionary of behavioral value between subjects to be
               correlated with intrasubject neural similarity (subjects are keys)
    targ_comp: List of targets whose similarity is correlated with xSs_behav
    radius: sl radius, default 3
    h5: 1 saves hdf5 of output as well 
    h5out: hdf5 outfilename
    status_print: if 1, prints status of searchlight (progress)
    
    '''   

    print('xSs BDSM initiated with...\n Ss: %s \n targ_comp1: %s\n targ_comp2: %s\n radius: %s\nh5: %s\nh5out: %s' % (data.keys(),targ_comp1,targ_comp2,radius,h5,h5out))

    if status_print == 1:
        if __debug__: debug.active += ["SLC"]
    else: pass
 
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
    bdsm = rsa_pymvpaw.xss_BehavioralDissimilarity_double(xSs_behav1,targ_comp1,xSs_behav2,targ_comp2)
    sl_bdsm = sphere_searchlight(bdsm,radius=radius)
    slmap_bdsm = sl_bdsm(group_data)
    print('Analysis complete with shape:',slmap_bdsm.shape)
    if h5 == 1:
        h5save(h5out,slmap_bdsm,compression=9)
        return slmap_bdsm
    else: return slmap_bdsm



#######################################################
# Pairsim
#######################################################

def sl_pairsim_1Ss(ds, pairs, radius=3, pairwise_metric='correlation',status_print=1):
    '''
    Gets pairwise (dis)similarity between specified target pairs in a searchlight

    ds: pymvpa dsets for 1 subj
    pairs: pairs to get neural sim of
    cmetric: spearman or eucldiean

    Returns: dict per voxel with pairs as keys, (dis)sim as values
    status_print: if 1, prints status of searchlight (progress)
    '''        

    if status_print == 1:
        if __debug__: debug.active += ["SLC"]
    else: pass

    ds = mean_group_sample(['targets'])(ds) #make UT ds
    print('Mean group sample computed at size:',ds.shape,'...with UT:',ds.UT)
    print('Beginning slPairsim analysis...')
    psm = rsa_pymvpaw.Pairsim(pairs,pairwise_metric=pairwise_metric)    
    sl = sphere_searchlight(psm,radius=radius)
    slmap = sl(ds)
    slmaps = dict([(k,np.array([i[k] for i in slmap.samples[0]]).flatten()) for k in slmap.samples[0][0]])
    return slmaps
    
def sl_pairsim_nSs(data, pairs, radius=3, pairwise_metric = 'correlation', h5 = 0, h5out = 'slRSA_m_nSs.hdf5',status_print=1):
    '''
    Runs sl_pairsim_1Ss (pariwise dissim per specified target pairs) per subject

    data: dictionary of pymvpa dsets per subj, indices being subjIDs
    pairs: bla
    pairwise_metric: metric for neural dis(sim)
    h5: 1 if you want to save hdf5 as well
    h5out: hdf5 outfilename

    Returns: datadict of pairwise sim per pair per subject
    '''        
    
    ### slRSA per subject ###
    slr= {} 
    print('Beginning group level searchlight on %s Ss...' % (len(data)))
    for subjid,ds in data.iteritems():
        print('\nPreparing slRSA for subject %s' % (subjid))
        subj_data = sl_pairsim_1Ss(ds,pairs,radius=radius,pairwise_metric=pairwise_metric,status_print=status_print)
        slr[subjid] = subj_data
    print('slPairsim complete for all subjects')

    slrall = dict([(s+'-'+p[0]+'-'+p[1],slr[s][p[0]+'-'+p[1]]) for p in pairs for s in slr])

    if h5==1:
        h5save(h5out,slrall,compression=9)
        return slrall
    else: return slrall


