#version 12/26/17
 
from importer import *

###############################################
# Runs slClass for 1 subject 
###############################################

def slClassPermTest_1Ss(ds, perm_count = 100, radius = 3, clf = LinearCSVMC(), part = NFoldPartitioner(), status_print = 1, h5 = 1, h5out = 'slClassPermTest_nulldist.hdf5'):
    '''
    Executes slClass on single subjects randomly permuted by target dsets  
    ...and returns avg accuracy per voxel / null per voxel dset

    ds: pymvpa dsets for 1 subj
    perm_count: number of permutations
    radius: sl radius, default 3
    clf: specify classifier
    part: specify partitioner
    status_print: if 1, prints status of searchlight (progress)
    h5: save output as hdf5
    h5out: filename for hdf5 output

    *based on Selzer et al. 2013
    https://lists.alioth.debian.org/pipermail/pkg-exppsy-pymvpa/2015q3/003202.html
    - recommended 100
   
    TO DO:
    - need to find way to force same shuffle order across subject
    - ideal algorithm will maintain chunks
    
    Probably best to run in parallel across subjects due to time.
    IMPORTANT: may lose dataset and feature attributes etc in output
    '''        

    if status_print == 1:
        if __debug__:
            debug.active += ["STATMC"]
            debug.active += ["SLC"]
    else: pass

    #Permutation setup
    permutator = AttributePermutator('targets', count=perm_count, limit='chunks')
    distr_est = MCNullDist(permutator, tail='right',enable_ca=['dist_samples'])
    cv = CrossValidation(clf,part,errorfx=lambda p, t: np.mean(p == t),enable_ca=['stats'], postproc=mean_sample())
    sl = sphere_searchlight(cv, radius=radius, space='voxel_indices',null_dist=distr_est,enable_ca=['roi_sizes'])

    #run sl
    sl_map = sl(ds)

    #return
    if h5 == 1:
        h5ssave(h5out,distr_est.ca.dist_samples,compression=9)
        return distr_est.ca.dist_samples
    else: return distr_est.ca.dist_samples


def Perm_GroupClusterThreshold( mean_map, perms, NN = 1, feature_thresh_prob = .005, n_bootstrap = 100000, fwe_rate = .05, h5 = 1, h5out = 'slClassPermTest.hdf5'):
    '''
    Executes GroupClusterThreshold to get back clusters corrected for multiple comparisons
    *based on Selzer et al. 2013

    mean_map: pymvpa dsets of group result map to be corrected (accuracy map)
    perms:    pymvpa dset with all perms per subject, each as a sample, with subject number in sa.chunks (make sure input here has proper feature (fa) and dataset (a) attributes, perhaps take them from mean_map
    NN:       Nearest neighbor clustering method - determines what voxels are contiguous / count as a cluster, 1: touch sides, 3: sides,edges,corners *need to add more options or make more flexible
    feature_thresh_prob,n_bootstrap, fwe_rate: see pymvpa doc for GroupClusterThreshold()
    h5:   save result to hdf5
    h5out:    filename for hdf5 output
    '''        

    if NN == 1: 
        clthr = GroupClusterThreshold(feature_thresh_prob=feature_thresh_prob,n_bootstrap=n_bootstrap,fwe_rate=fwe_rate)
    elif NN == 3:
        clthr = gct_pymvpaw.GroupClusterThreshold_NN3(feature_thresh_prob=feature_thresh_prob,n_bootstrap=n_bootstrap,fwe_rate=fwe_rate)
        
    print('Beginning to bootstrap... dont hold your breath here (has taken close to an hour for an example I did with 1600 samples in perms)')
    clthr.train(perms)
    print('Null distribution and cluster measurements complete, applying to group result map')
    res = clthr(mean_map)
    print('Correction complete... see res.a for stats table etc., res.fa.clusters_fwe_thresh for a mask of clusters that survived - see doc')
    if h5 == 1:
        h5save(h5out, res, compression=9)
        return res
    else: return res
    


