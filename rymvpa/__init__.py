"""
rymvpa is a Python wrapper for pymvpa to enable faster 
access to some pymvpa analyses, with sane defaults.
rymvpa is written entirely in Python and requires 
only a working copy of pymvpa.

Git it at:
https://github.com/rystoli/pymvpa_rs
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

from wrapit_pymvpa import *
