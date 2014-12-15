#version: 11_Nov_14

import numpy as np
#tdsm = squareform(dsms['oth']['idioAvg_12C'])
#pdsm = squareform(dsms['oth']['orth_12'])
#ndsm = np.random.random(66)

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
    print xvar
    yvar = svar(Y)
    print yvar
    zvar = svar(Z)
    print zvar
    # computes pairwise simple correlations.
    rxy  = corr(X,Y, xbar=xbar, xvar= xvar, ybar = ybar, yvar = yvar)
    rxz  = corr(X,Z, xbar=xbar, xvar= xvar, ybar = zbar, yvar = zvar)
    ryz  = corr(Y,Z, xbar=ybar, xvar= yvar, ybar = zbar, yvar = zvar)
    rxy_z = (rxy - (rxz*ryz)) / np.sqrt((1 -rxz**2)*(1-ryz**2))
    rxz_y = (rxz - (rxy*ryz)) / np.sqrt((1-rxy**2) *(1-ryz**2))
    ryz_x = (ryz - (rxy*rxz)) / np.sqrt((1-rxy**2) *(1-rxz**2))
    return {'rxy_z': rxy_z, 'rxz_y': rxz_y, 'ryz_x': ryz_x}


