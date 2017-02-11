
##########################################################################################
##########################################################################################

#NOTES
#h5save measurements.label input to see if i can get structarray to work


##########################################################################################
##########################################################################################

#import perm data
from rymvpa import *
import glob
remap = fmri_dataset('remap_424.nii.gz',mask='../masks/ventral_stream_HO_mask.nii.gz')
permsL = []
for i,s in enumerate(glob.glob('526_results/nii/corrections/4*sex*')):
    res = h5load(s)
    res = Dataset(np.transpose(res.samples[0]),fa=remap.fa)
    res.sa['chunks'] = [i+1 for j in range(len(res))]
    permsL.append(res)
perms = vstack(permsL)
#basic instantiation, NN = 1?
clthr = GroupClusterThreshold(feature_thresh_prob=.005)
clthr.train(perms)
#mean map = ...
mean_map = fmri_dataset('526_results/nii/slSVM_2sex_526/t_slSVM_sex.nii.gz',mask='../masks/ventral_stream_HO_mask.nii.gz')[0]
mean_map.samples = mean_map.samples + .5
resClthr = clthr(mean_map)
resClthr.fa.clusters_fwe_thresh
resClthr.a.clusterstats
#some way to save map.... i forget, can use it as mask for data


##########################################################################################
##########################################################################################

#test new NN argument - after
from rymvpa import *
import glob
remap = fmri_dataset('remap_424.nii.gz',mask='../masks/ventral_stream_HO_mask.nii.gz')
permsL = []
for i,s in enumerate(glob.glob('526_results/nii/corrections/4*race*')):
    res = h5load(s)
    res = Dataset(np.transpose(res.samples[0]),fa=remap.fa,a=remap.a)
    res.sa['chunks'] = [i+1 for j in range(len(res))]
    permsL.append(res)
perms = vstack(permsL)
perms.a = remap.a #bc vstack loses a
#use edited groupclusterthr
cd ~/RyMVPA/rymvpa/
import group_clusterthr_rymvpa
clthr = group_clusterthr_rymvpa.GroupClusterThreshold_NN3(feature_thresh_prob=.005)
#need it to debug so measurements.label works with new structure array
clthr.train(perms)
h5save('clthr005nn3_race.hdf5',clthr,compression=9)

##########################################################################################
##########################################################################################

#to specify cluster size.... ugly solution
#once trained, and boolean result map ready
import statsmodels.stats.multitest as smm
from mvpa2.algorithms.group_clusterthr import _transform_to_pvals as ttp
nc = clthr.train(perms)
dsc = get_cluster_sizes(dsboolean)
dsc.update((SIZEPREFERED,1))
cluster_probs_raw = ttp(dsc,nc._null_cluster_sizes.astype('float'))
rej, probs_corr = smm.multipletests(cluster_probs_raw,alpha=nc.params.fwe_rate,method=nc.params.multicomp_correction)[:2]
