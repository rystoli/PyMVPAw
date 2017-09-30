#version 8/19/15
#imports modules needed by other scripts
from mvpa2.suite import *
import os
import pylab as pylab
import numpy as np
import rsa as rsa
import rsa_rymvpa as rsa_rymvpa
import group_clusterthr_rymvpa as gct_rymvpa
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import nltools as nl
import nilearn.plotting as pl
from nilearn import image
from nltools.plotting import plotBrain
