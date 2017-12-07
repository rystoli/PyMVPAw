#version 8/19/15
"""New RyMVPA measures of consistency between dissimilarity matrices across chunks."""

__docformat__ = 'restructuredtext'

import numpy as np
from mvpa2.measures.base import Measure
from mvpa2.datasets.base import Dataset
from mvpa2.mappers.fx import mean_group_sample
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, pearsonr
import statsmodels.api as sm
from collections import OrderedDict

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#         Partial correlation function for partial RSA... prob easier way
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

def mean(X):
    """
    returns mean of vector X.
    """
    return(float(sum(X))/ len(X))
 
def svar(X, xbar = None):
    """
    returns the sample variance of vector X.
    xbar is sample mean of X.
    """ 
    if xbar is None: #fools had mean instead of xbar
       xbar = mean(X)
    S = sum([(x - xbar)**2 for x in X])
    return S / (len(X) - 1)
 
def corr(X,Y, xbar= None, xvar = None, ybar = None, yvar= None):
    """
    Computes correlation coefficient between X and Y.
    returns None on error.
    """
    n = len(X)
    if n != len(Y):
       return 'size mismatch X/Y:',len(X),len(Y)
    if xbar is None: xbar = mean(X)
    if ybar is None: ybar = mean(Y)
    if xvar is None: xvar = svar(X)
    if yvar is None: yvar = svar(Y)
 
    S = sum([(X[i] - xbar)* (Y[i] - ybar) for i in range(len(X))])
    return S/((n-1)* np.sqrt(xvar* yvar))

def pcf3(X,Y,Z):
    """
    Returns a dict of the partial correlation coefficients
    r_XY|z , r_XZ|y, r_YZ|x 
    """
    xbar = mean(X)
    ybar = mean(Y)
    zbar = mean(Z)
    xvar = svar(X)
    yvar = svar(Y)
    zvar = svar(Z)
    # computes pairwise simple correlations.
    rxy  = corr(X,Y, xbar=xbar, xvar= xvar, ybar = ybar, yvar = yvar)
    rxz  = corr(X,Z, xbar=xbar, xvar= xvar, ybar = zbar, yvar = zvar)
    ryz  = corr(Y,Z, xbar=ybar, xvar= yvar, ybar = zbar, yvar = zvar)
    rxy_z = (rxy - (rxz*ryz)) / np.sqrt((1 -rxz**2)*(1-ryz**2))
    rxz_y = (rxz - (rxy*ryz)) / np.sqrt((1-rxy**2) *(1-ryz**2))
    ryz_x = (ryz - (rxy*rxz)) / np.sqrt((1-rxy**2) *(1-rxz**2))
    return {'rxy_z': rxy_z, 'rxz_y': rxz_y, 'ryz_x': ryz_x}


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
# FANCY RSA FUNCTIONS
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

class TargetDissimilarityCorrelationMeasure_Regression(Measure):
    """
    Target dissimilarity correlation `Measure`. Computes the correlation between
    the dissimilarity matrix defined over the pairwise distances between the
    samples of dataset and the target dissimilarity matrix.
    """
    
    is_trained = True
    """Indicate that this measure is always trained."""

    def __init__(self, target_dsm, control_dsms = None, resid = False,  
                    pairwise_metric='correlation', comparison_metric='pearson', 
                    center_data = False, corrcoef_only = False, **kwargs):
        """
        Initialize

        Parameters
        ----------
        dataset :           Dataset with N samples such that corresponding dissimilarity
                            matrix has N*(N-1)/2 unique pairwise distances
        target_dsm :        numpy array, length N*(N-1)/2. Target dissimilarity matrix
                            this is the predictor who's results get mapped back
        control_dsms:       list of numpy arrays, length N*(N-1)/2. DMs to be controlled for
                            Default: 'None'  
                            controlled for when getting results of target_dsm back
        resid:              Set to True to return residuals to searchlight center for 
                            smoothing estimation, default to False
        pairwise_metric :   To be used by pdist to calculate dataset DSM
                            Default: 'correlation', 
                            see scipy.spatial.distance.pdist for other metric options.
        comparison_metric : To be used for comparing dataset dsm with target dsm
                            Default: 'pearson'. Options: 'pearson' or 'spearman'
        center_data :       Center data by subtracting mean column values from
                            columns prior to calculating dataset dsm. 
                            Default: False
        corrcoef_only :     If true, return only the correlation coefficient
                            (rho), otherwise return rho and probability, p. 
                            Default: False
        Returns
        -------
        Dataset :           Dataset contains the correlation coefficient (rho) only or
                            rho plus p, when corrcoef_only is set to false.
        """
        # init base classes first
        Measure.__init__(self, **kwargs)
        if comparison_metric not in ['spearman','pearson']:
            raise Exception("comparison_metric %s is not in "
                            "['spearman','pearson']" % comparison_metric)
        self.target_dsm = target_dsm
        if comparison_metric == 'spearman':
            self.target_dsm = rankdata(target_dsm)
        self.pairwise_metric = pairwise_metric
        self.comparison_metric = comparison_metric
        self.center_data = center_data
        self.corrcoef_only = corrcoef_only
        self.control_dsms = control_dsms
        if comparison_metric == 'spearman' and control_dsms != None:
            self.control_dsms = [rankdata(dm) for dm in control_dsms]
        self.resid = resid

    def _call(self,dataset):
        data = dataset.samples
        if self.center_data:
            data = data - np.mean(data,0)
        dsm = pdist(data,self.pairwise_metric)
        if self.comparison_metric=='spearman':
            dsm = rankdata(dsm)
        if self.control_dsms == None:
            rho, p = pearsonr(dsm,self.target_dsm)
            if self.corrcoef_only:
                return Dataset(np.array([rho,]))
            else: 
                return Dataset(np.array([rho,p]))
        elif self.control_dsms != None:
            X = sm.add_constant(np.column_stack([self.target_dsm]+self.control_dsms))
            res = sm.OLS(endog=dsm,exog=X).fit()
            if self.resid == True: b = np.sum(res.resid**2)
            elif self.resid == False: b = res.params[1]
            return Dataset(np.array([b]))

class TargetDissimilarityCorrelationMeasure_Partial(Measure):
    """
    Target dissimilarity correlation `Measure`. Computes the correlation between
    the dissimilarity matrix defined over the pairwise distances between the
    samples of dataset and the target dissimilarity matrix.
    """
    
    is_trained = True
    """Indicate that this measure is always trained."""

    def __init__(self, target_dsm, partial_dsm = None, pairwise_metric='correlation', 
                    comparison_metric='pearson', center_data = False, 
                    **kwargs):
        """
        Initialize

        Parameters
        ----------
        dataset :           Dataset with N samples such that corresponding dissimilarity
                            matrix has N*(N-1)/2 unique pairwise distances
        target_dsm :        numpy array, length N*(N-1)/2. Target dissimilarity matrix
        partial_dsm:        numpy array, length N*(N-1)/2. DSM to be partialled out
                            Default: 'None'  
        pairwise_metric :   To be used by pdist to calculate dataset DSM
                            Default: 'correlation', 
                            see scipy.spatial.distance.pdist for other metric options.
        comparison_metric : To be used for comparing dataset dsm with target dsm
                            Default: 'pearson'. Options: 'pearson' or 'spearman'
        center_data :       Center data by subtracting mean column values from
                            columns prior to calculating dataset dsm. 
                            Default: False
        Returns
        -------
        Dataset :           Dataset contains the correlation coefficient (rho) only or
                            rho plus p, when corrcoef_only is set to false.
        """
        # init base classes first
        Measure.__init__(self, **kwargs)
        if comparison_metric not in ['spearman','pearson']:
            raise Exception("comparison_metric %s is not in "
                            "['spearman','pearson']" % comparison_metric)
        self.target_dsm = target_dsm
        if comparison_metric == 'spearman':
            self.target_dsm = rankdata(target_dsm)
        self.pairwise_metric = pairwise_metric
        self.comparison_metric = comparison_metric
        self.center_data = center_data
        self.partial_dsm = partial_dsm
        if comparison_metric == 'spearman' and partial_dsm != None:
            self.partial_dsm = rankdata(partial_dsm)

    def _call(self,dataset):
        data = dataset.samples
        if self.center_data:
            data = data - np.mean(data,0)
        dsm = pdist(data,self.pairwise_metric)
        if self.comparison_metric=='spearman':
            dsm = rankdata(dsm)
        if self.partial_dsm == None:
            rho, p = pearsonr(dsm,self.target_dsm)
            return Dataset(np.array([rho]))
        elif self.partial_dsm != None:
            rp = pcf3(dsm,self.target_dsm,self.partial_dsm)
            return Dataset(np.array([rp['rxy_z']]))


class xss_BehavioralDissimilarity(Measure):
    """
    Between Subjects Behavioral Dissimilarity Measure is a method that caculates the neural
    similarity between two conditions per subject, then correlates another subject-level
    variable with the neural similarity of the two conditions between subjects. I.e.,
    This looks at whether an individual difference predicts neural similarity of representations. 

    """
    is_trained = True
    """Indicate that this measure is always trained."""

    def __init__(self, xSs_behav, targ_comp, comparison_metric='pearson', chunks_attr='chunks', **kwargs):
        """Initialize

        Parameters
        ----------
        xSs_behav:          Dictionary of behavioral value between subjects to be
                            correlated with intrasubject neural similarity (subjects are keys)
        targ_comp:          List of targets whose similarity is correlated with xSs_behav
        chunks_attr :       Chunks attribute to use for chunking dataset. Can be any
                            samples attribute specified in the dataset.sa dict.
                            (Default: 'chunks')
        comparison_metric:  Distance measure for behavioral to neural comparison.
                            'pearson' (default) or 'spearman'
        center_data :       boolean. (Optional. Default = False) If True then center 
                            each column of the data matrix by subtracing the column 
                            mean from each element  (by chunk if chunks_attr 
                            specified). This is recommended especially when using 
                            pairwise_metric = 'correlation'.  

        Returns
        -------
        Dataset:    Contains an array of the pairwise correlations between the
                    DSMs defined for each chunk of the dataset. Length of array
                    will be N(N-1)/2 for N chunks.

        To Do:
        Another metric for consistency metric could be the "Rv" coefficient...  (ac)
        """
        # init base classes first
        Measure.__init__(self, **kwargs)

        self.xSs_behav = xSs_behav
        self.targ_comp = targ_comp
        self.chunks_attr = chunks_attr
        self.comparison_metric = comparison_metric

    def _call(self, dataset):
        """Computes the aslmap_dcm = sl_dcm(group_data)verage correlation in similarity structure across chunks."""
        
        chunks_attr = self.chunks_attr
        nchunks = len(np.unique(dataset.sa[chunks_attr]))
        if nchunks < 2:
            raise StandardError("This measure calculates similarity consistency across "
                                "chunks and is not meaningful for datasets with only "
                                "one chunk:")

        #calc neur sim b/w targ_comp targets per subject
        neur_sim={}
        for s in np.unique(dataset.sa[chunks_attr]):
            ds_s = dataset[dataset.sa.chunks == s]
            neur_sim[s] = 1 - np.corrcoef(ds_s[ds_s.sa.targets == self.targ_comp[0]],ds_s[ds_s.sa.targets == self.targ_comp[1]])[0][1]            
        #create dsets where cols are neural sim and mt sim for correlations
        behav_neur = np.array([[self.xSs_behav[s],neur_sim[s]] for s in neur_sim])
        #correlate behav with neur sim b/w subjects
        if self.comparison_metric == 'spearman':
            xSs_corr = pearsonr(rankdata(behav_neur[:,0]),rankdata(behav_neur[:,1])) 
        xSs_corr = pearsonr(behav_neur[:,0],behav_neur[:,1])
        
        #returns fish z transformed r coeff ; could change to be p value if wanted...
        return Dataset(np.array([np.arctanh(xSs_corr[0])])) 


class xss_BehavioralDissimilarity_double(Measure):
    """
    Between Subjects Behavioral Dissimilarity Measure is a method that caculates the neural
    similarity between two conditions per subject, then correlates another subject-level
    variable with the neural similarity of the two conditions between subjects. I.e.,
    This looks at whether an individual difference predicts neural similarity of representations. 

    """
    is_trained = True
    """Indicate that this measure is always trained."""

    def __init__(self, xSs_behav1, targ_comp1, xSs_behav2, targ_comp2, comparison_metric='pearson', chunks_attr='chunks', **kwargs):
        """Initialize 

           Parameters
        ----------
        xSs_behav:          Dictionary of behavioral value between subjects to be
                            correlated with intrasubject neural similarity (subjects are keys)
        targ_comp:          List of targets whose similarity is correlated with xSs_behav
        chunks_attr :       Chunks attribute to use for chunking dataset. Can be any
                            samples attribute specified in the dataset.sa dict.
                            (Default: 'chunks')
        comparison_metric:  Distance measure for behavioral to neural comparison.
                            'pearson' (default) or 'spearman'
        center_data :       boolean. (Optional. Default = False) If True then center 
                            each column of the data matrix by subtracing the column 
                            mean from each element  (by chunk if chunks_attr 
                            specified). This is recommended especially when using 
                            pairwise_metric = 'correlation'.  

        Returns
        -------
        Dataset:    Contains an array of the pairwise correlations between the
                    DSMs defined for each chunk of the dataset. Length of array
                    will be N(N-1)/2 for N chunks.

        To Do:
        Another metric for consistency metric could be the "Rv" coefficient...  (ac)
        """
        # init base classes first
        Measure.__init__(self, **kwargs)

        self.xSs_behav1 = xSs_behav1
        self.targ_comp1 = targ_comp1
        self.xSs_behav2 = xSs_behav2
        self.targ_comp2 = targ_comp2
        self.chunks_attr = chunks_attr
        self.comparison_metric = comparison_metric

    def _call(self, dataset):
        """Computes the aslmap_dcm = sl_dcm(group_data)verage correlation in similarity structure across chunks."""
        
        chunks_attr = self.chunks_attr
        nchunks = len(np.unique(dataset.sa[chunks_attr]))
        if nchunks < 2:
            raise StandardError("This measure calculates similarity consistency across "
                                "chunks and is not meaningful for datasets with only "
                                "one chunk:")

        #calc neur sim b/w targ_comp targets per subject
        neur_sim={}
        for s in np.unique(dataset.sa[chunks_attr]):
            ds_s = dataset[dataset.sa.chunks == s]
            neur_sim[s+'1'] = 1 - np.corrcoef(ds_s[ds_s.sa.targets == self.targ_comp1[0]],ds_s[ds_s.sa.targets == self.targ_comp1[1]])[0][1]            
            neur_sim[s+'2'] = 1 - np.corrcoef(ds_s[ds_s.sa.targets == self.targ_comp2[0]],ds_s[ds_s.sa.targets == self.targ_comp2[1]])[0][1]            

        #combine xSs_behavs
        xSs_behav = {}
        for s in self.xSs_behav1:
            xSs_behav[s+'1'] = self.xSs_behav1[s]
        for s in self.xSs_behav2:
            xSs_behav[s+'2'] = self.xSs_behav2[s]

        #create dsets where cols are neural sim and mt sim for correlations
        behav_neur = np.array([[xSs_behav[s],neur_sim[s]] for s in neur_sim])
        #correlate behav with neur sim b/w subjects
        if self.comparison_metric == 'spearman':
            xSs_corr = pearsonr(rankdata(behav_neur[:,0]),rankdata(behav_neur[:,1])) 
        xSs_corr = pearsonr(behav_neur[:,0],behav_neur[:,1])
        
        #returns fish z transformed r coeff ; could change to be p value if wanted...
        return Dataset(np.array([np.arctanh(xSs_corr[0])])) 





class SampleBySampleSimilarityCorrelation(Measure):
    """
    Sample by sample similarity correlation `Measure`. Computes the dissimilarity of each designated sample with another specified sample(s), then returns the correlations of that distance with any variable with a value for each sample. E.g., one could have samples be betas per trial, and measure if reaction time predicts similarity between one condition and the average representation of another condition. 
    **importantly, assumes anything not used in analysis is removed from dataset (targets)
    """
    
    is_trained = True
    """Indicate that this measure is always trained."""

    def __init__(self, targs_comps, sample_covariable, pairwise_metric='correlation', 
                    comparison_metric='pearson', center_data = False, 
                    corrcoef_only = False, **kwargs):
        """
        Initialize

        Parameters
        ----------
        dataset :           Dataset with N samples such that corresponding dissimilarity
                            matrix has N*(N-1)/2 unique pairwise distances
        targs_comps:        Dict of trial by trial targets (keys) and their comparison targets 
                            (values) - ***this measure assumes other omitted first***
        sample_covariable:  Name of the variable (sample attribute) with a value for each sample. 
                            The distance of each sample with the comparison_sample will be 
                            correlated with this variable.
        pairwise_metric :   To be used by pdist to calculate dataset DSM
                            Default: 'correlation', 
                            see scipy.spatial.distance.pdist for other metric options.
        comparison_metric : To be used for comparing dataset dsm with target dsm
                            Default: 'pearson'. Options: 'pearson' or 'spearman'
        center_data :       Center data by subtracting mean column values from
                            columns prior to calculating dataset dsm. 
                            Default: False
        corrcoef_only :     If true, return only the correlation coefficient
                            (rho), otherwise return rho and probability, p. 
                            Default: False 
        Returns
        -------
        Dataset :           Dataset contains the correlation coefficient (rho) only or
                            rho plus p, when corrcoef_only is set to false.

        -------
        TO DO:              Should this be done as repeated measures ANCOVA instead?
                            Does not currently handle rho comparison of samples, or rho 
                            corr with covariable
                            Should use mean_group_sample in wrapper function to get comparison_sample
                            Maybe have omit inside this method?
        """
        # init base classes first
        Measure.__init__(self, **kwargs)
        if comparison_metric not in ['spearman','pearson']:
            raise Exception("comparison_metric %s is not in "
                            "['spearman','pearson']" % comparison_metric)
        self.targs_comps = targs_comps
        self.sample_covariable = sample_covariable
        #if comparison_metric == 'spearman':
        #    self.target_dsm = rankdata(target_dsm)
        self.pairwise_metric = pairwise_metric
        self.comparison_metric = comparison_metric
        self.center_data = center_data
        self.corrcoef_only = corrcoef_only

    def _call(self,dataset):
        data = dataset.samples
        if self.center_data:
            data = data - np.mean(data,0)

        #compute comparison sample
        comp_samps = mean_group_sample(['targets'])(dataset)
        #omit all samples from comparison_sample target conditions
        for om in self.targs_comps.values():
            dataset = dataset[dataset.sa.targets != om] 

        #calculate sample attribute of distance between sample and comparison_sample (corr coef and p value)
        dataset.sa['sample_comp_dist_r'] = [pearsonr(s.samples[0],comp_samps[comp_samps.sa.targets == self.targs_comps[s.sa.targets[0]]].samples[0])[0] for s in dataset]
        dataset.sa['sample_comp_dist_p'] = [pearsonr(s.samples[0],comp_samps[comp_samps.sa.targets == self.targs_comps[s.sa.targets[0]]].samples[0])[1] for s in dataset]
        #calculate final correlations
        rho, p = pearsonr(dataset.sa['sample_comp_dist_r'],dataset.sa[self.sample_covariable])
        if self.corrcoef_only:
            return Dataset(np.array([rho,]))
        else:
            return Dataset(np.array([rho,p]))


class Pairsim(Measure):
    """
    Returns (dis)similarity value of specified targets (or multiple pairs) 
    """
    
    is_trained = True
    """Indicate that this measure is always trained."""

    def __init__(self, pairs, pairwise_metric='correlation', **kwargs):
        """
        Initialize

        Parameters
        ----------
        dataset :           Dataset with N samples such that corresponding dissimilarity
                            matrix has N*(N-1)/2 unique pairwise distances
                            Make sure is in alphabetical order!
        pairs :             list of lists (pairs) of target names
        pairwise_metric :   To be used by pdist to calculate dataset DSM
                            Default: 'correlation', 
                            see scipy.spatial.distance.pdist for other metric options.
        Returns
        -------
        Dataset :           Dataset contains the sim value 

        -------
        """
        # init base classes first
        Measure.__init__(self, **kwargs)
        self.pairs = pairs
        self.pairwise_metric = pairwise_metric
        print 'init'

    def _call(self,dataset):

        # Get neural sim b/w pairs of targets
        if self.pairwise_metric == 'correlation':
            pairsim = dict((pair[0]+'-'+pair[1],pdist([dataset[dataset.sa.targets == pair[0]].samples[0], dataset[dataset.sa.targets == pair[1]].samples[0]],metric=self.pairwise_metric)) for pair in self.pairs)
        else: pairsim = dict((pair[0]+'-'+pair[1],pdist([dataset[dataset.sa.targets == pair[0]].samples[0], dataset[dataset.sa.targets == pair[1]].samples[0]],metric=self.pairwise_metric)) for pair in self.pairs)
        return Dataset(np.array([pairsim,]))


class Pairsim_RSA(Measure):
    """
    Runs RSA but only included specified cells from the DM
    """
    
    is_trained = True
    """Indicate that this measure is always trained."""

    def __init__(self, pairs_dsm, pairwise_metric='correlation', 
                    comparison_metric='pearson', center_data = False, 
                    corrcoef_only = False, **kwargs):
        """
        Initialize

        Parameters
        ----------
        dataset :           Dataset with N samples such that corresponding dissimilarity
                            matrix has N*(N-1)/2 unique pairwise distances
        pairs_dsm :         Dictionary of target pairs separated by '-' (keys) and
                            corresponding predicted model dissimilarity values (values)
        pairwise_metric :   To be used by pdist to calculate dataset DSM
                            Default: 'correlation', 
                            see scipy.spatial.distance.pdist for other metric options.
        comparison_metric : To be used for comparing dataset dsm with target dsm
                            Default: 'pearson'. Options: 'pearson' or 'spearman'
        center_data :       Center data by subtracting mean column values from
                            columns prior to calculating dataset dsm. 
                            Default: False
        corrcoef_only :     If true, return only the correlation coefficient
                            (rho), otherwise return rho and probability, p. 
                            Default: False 
        Returns
        -------
        Dataset :           Dataset contains the correlation coefficient (rho) only or
                            rho plus p, when corrcoef_only is set to false.

        -------
        TO DO:             Add partial correlation and multiple regression RSA 
                            
        """
        # init base classes first
        Measure.__init__(self, **kwargs)
        if comparison_metric not in ['spearman','pearson','euclidean']:
            raise Exception("comparison_metric %s is not in "
                            "['spearman','pearson','euclidean']" % comparison_metric)
        self.pairs_dsm = pairs_dsm
        self.pairwise_metric = pairwise_metric
        self.comparison_metric = comparison_metric
        self.center_data = center_data
        self.corrcoef_only = corrcoef_only
        self.pairs = [i.split('-') for i in self.pairs_dsm.keys()]

    def _call(self,dataset):
        print 'hii'
        print dataset.UT
        #compute comparison sample
        ds = mean_group_sample(['targets'])(dataset)
        # Get neural dissim b/w pairs of targets
        pairsim = dict((pair[0]+'-'+pair[1],(1 - pearsonr(ds[ds.sa.targets == pair[0]].samples[0], ds[ds.sa.targets == pair[1]].samples[0])[0])) for pair in self.pairs)

        #Order DMs...
        pairs_dsm_o = OrderedDict(sorted(self.pairs_dsm.items())).values()
        pairsim_o = OrderedDict(sorted(pairsim.items())).values()

        #RSA
        if self.comparison_metric == 'spearman':
            res = np.arctanh(pearsonr(rankdata(pairs_dsm_o),rankdata(pairsim_o))[0])
        elif self.comparison_metric == 'pearson':
            res = np.arctanh(pearsonr(pairs_dsm_o,pairsim_o)[0])
        elif self.comparison_metric == 'euclidean':
            res = pdist(np.vstack([self.pairs_dsm,pairsim_vals]))
            res = np.round((-1 * res) + 2) #why?
        return Dataset(np.array([res,]))


