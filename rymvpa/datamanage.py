#version 8/19/15
from importer import *

##################################################

def countDig(dig,arr):
    '''Counts value 'dig' in 1D iterable arr'''
    count = 0
    for i in arr:
        if i==dig: count+=1
    return count

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
    
    
def sl2nifti(ds,remap,outfile):
    '''
    No return; converts sl output and saves nifti file to working directory

    ds=array of sl results
    remap: dataset to remap to
    outfile: string outfile name including extension .nii.gz
    '''

    nimg = map2nifti(data=ds,dataset=remap)
    nimg.to_filename(outfile)

def datadict2nifti(datadict,remap,outdir,outprefix=''):
    '''

    No return; runs sl2nifti on dictionary of sl data files (1subbrick), saving each file based upon its dict key

    datadict: dictionary of pymvpa datasets
    remap: dataset to remap to, or dictionary per subj in datadict
    outdir: target directory to save nifti output
    outprefix: optional; prefix string to outfile name

    TO DO: make call afni for 3ddtest...
    '''

    os.mkdir(outdir) #make directory to store data
    for key,ds in datadict.iteritems():
        print('Writing nifti for subject: %s' % (key))
	if (type(remap) == dict): thisRemap=remap[key]
        else: thisRemap = remap
        sl2nifti(ds,thisRemap,os.path.join(outdir,'%s%s.nii.gz' % (outprefix,key)))
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

def omit_targets_data(data,omit):
    '''
    Returns data with specified targets omitted

    data: dictionary containing pymvpa datasets with targets
    omit: list of targets to be omitted
    '''
    
    for key in data:
	ds= data[key]
	data[key]= omit_targets(ds,omit)
    return data

def select_targets(ds, select):
    '''
    Returns ds with specified targets selected

    ds: pymvpa dataset with targets
    select: list of targets to be selected
    '''     

    omit= [x for x in ds.sa.targest if not (x in select)]
    return omit_targets(ds, omit)

def select_targets_data(data, select):
    '''
    Returns data with specified targets selected

    data: dictionary containing pymvpa dataset with targets
    select: list of targets to be selected
    '''
    
    for key in data:
	ds= data[key]
	data[key]= select_targets(ds, select)
    return data
    
def mask_dset(dset, mask):
    '''
    Returns masked dataset

    ds: pymvpa dataset
    mask: binary [0,1] mask file in nii format
    '''

    mask = datasets.mri._load_anyimg(mask)[0]
    flatmask = ds.a.mapper.forward1(mask)
    return ds[:, flatmask != 0]

def sa2csv(dset, salist):
    '''
    Saves csv with SA specified
    NEED TO FINALIZE AND MAKE DICT ONE
    '''
    ds = np.asarray([[j.a.subjID,j.sa.chunks[0],j.sa.targets[0],j.sa.time_indices[0],j.sa.MD[0],j.sa.ACC[0]] for i,j in enumerate(colordata[s])])
    np.savetxt("sa_dset.csv", ds, delimiter=",", fmt='%s')
    t = {}
    for s in colordata:
        t[s] = np.asarray([[j.a.subjID,j.sa.chunks[0],j.sa.targets[0],j.sa.time_indices[0],j.sa.MD[0],j.sa.ACC[0]] for i,j in enumerate(colordata[s])])
    np.savetxt("color_txt.csv", np.vstack(t.values()), delimiter=",", fmt='%s')


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


