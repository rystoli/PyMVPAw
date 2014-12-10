#version 11_Nov_14
from mvpa2.suite import *
import os
import pylab as pylab
import numpy as np


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
        slRSA2nifti(ds,thisRemap,os.path.join(outdir,'%s%s.nii.gz' % (outprefix,key)))
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
