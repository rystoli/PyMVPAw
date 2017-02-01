#version 2/1/17 apocalypse day 13

##############
# IN PROGRESS, IGNORE!
 
from importer import *


###############################################
# Runs slClass for 1 subject 
###############################################

def slClassPermTest_1Ss(ds, perm_count = 100, radius=3, clf = LinearCSVMC(), part = NFoldPartitioner(), status_print=1):
    '''

    Executes slClass on single subjects and returns ?avg accuracy per voxel?

    ds: pymvpa dsets for 1 subj
    perm_count: number of permutations
    radius: sl radius, default 3
    clf: specify classifier
    part: specify partitioner
    status_print: if 1, prints status of searchlight (progress)

    *based on Selzer et al. 2013
    https://lists.alioth.debian.org/pipermail/pkg-exppsy-pymvpa/2015q3/003202.html
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
    return distr_est.ca.dist_samples


###################################
# possible implementation group level
###################################

#z = []
#for i,s in enumerate(['586','587','584']):
#    res = slClassPermTest_1Ss(mask_dset(data[s],'../masks/ACC.nii.gz'),perm_count=3)
#    res = Dataset(np.transpose(res.samples[0]),fa=ds.fa)
#    res.sa['chunks'] = [i+1 for j in range(len(res))]
#    z.append(res)
#perms = vstack(z) 
#clthr = GroupClusterThreshold()
#clthr.train(perms)
#res = clthr(mean_map)
#res.fa.clusters_fwe_thresh


###################################
#OTHER CODE
###################################

#    repeater = Repeater(count=5)
#    permutator = AttributePermutator('targets', limit={'partitions': 1}, count=1)
#    null_cv = CrossValidation(clf, ChainNode([part, permutator], space=part.get_space()), errorfx=lambda p, t: np.mean(p == t))
#    distr_est = MCNullDist(repeater, tail='right', measure=null_cv, enable_ca=['dist_samples'])
#    cv = CrossValidation(clf, part, errorfx=lambda p, t: np.mean(p == t), null_dist=distr_est, enable_ca=['stats'])
#
#    sl = sphere_searchlight(cv, radius=radius, postproc=mean_sample(),null_dist=distr_est)
#
#
#
#
#    print('Beginning sl classification analysis...')
#
#    sl = sphere_searchlight(cv, radius=radius, postproc=mean_sample(),null_dist=distr_est)
#    sl = sphere_searchlight(cv, radius=radius, postproc=mean_sample())
#    slr = sl(ds)
#    return sfs.reverse(slr).samples - (1.0/len(ds.UT))
#   
#    res = cv(ds)
#    nd = cv.null_dist.ca.dist_samples.samples[:,0,:]
#    acc_nd = np.mean(nd,axis=0)

