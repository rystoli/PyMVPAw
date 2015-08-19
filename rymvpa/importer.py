#version 8/19/15
#imports modules needed by other scripts
from mvpa2.suite import *
import os
import pylab as pylab
import numpy as np
import rsa as rsa
import rsa_rymvpa as rsa_adv
from scipy.spatial.distance import pdist, squareform
