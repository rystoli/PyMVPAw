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

from rymvpa_searchlights import *
from rymvpa_rois import *
from rymvpa_datamanage import *

#for integration still need to get everything to load proper, make omit optional...
#how do we handle imports?



##################################################
# currently in progress of making this module with slRSA functions...

# TO DO: 

#*****   allow slRSA functions kwargs to use overlap_mgs instead of default mgs; issue may be that overlap_msg() requires omit submitted directly to it? nope, not issue just need to include?

#move dataloading etc. to end in __main__; make dsms refresh take arguments

# allow people to not use 'omit' argument

