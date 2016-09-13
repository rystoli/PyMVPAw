#version 8/19/15
"""
rymvpa is a Python wrapper for pymvpa to enable faster 
access to some pymvpa analyses, with sane defaults.
rymvpa is written entirely in Python and requires 
only a working copy of pymvpa.

Git it at:
https://github.com/rystoli/RyMVPA
"""

__version__ = "0.0.1"

import sys
if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    raise ImportError("Python Version 2.6 or above is required for rymvpa.")
elif sys.version_info[0] == 3:
    raise ImportError("Python Version 3 not supported :-X")
else:
    pass

del sys

from datamanage import *
from roi_wraps import *
from searchlight_wraps import *
from partition_rymvpa import *


#############################################################
#TO DO:
#############################################################

#for integration still need to get everything to load proper, make omit optional...

#*****   allow slRSA functions kwargs to use overlap_mgs instead of default mgs; issue may be that overlap_msg() requires omit submitted directly to it? nope, not issue just need to include?

#RSA ROI analyses

#classification, train on one target, test on others
#more flexible classification output


