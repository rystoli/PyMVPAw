import os
from mvpa2.suite import map2nifti

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

