#!/usr/bin/env python

from distutils.core import setup

setup(name='rymvpa',
        version='0.0.2',
        description='PyMVPA wrapper',
        author='Ryan Stolier',
        author_email='',
        url='https://github.com/rystoli/RyMVPA',
        packages=['rymvpa']
        )


#install
#pip install [this directory] 
#update
#pip install [this directory] -U

#update with git
#git clone [address found on github]
#git add -A #adds including new stuff, deletes old (see other options like -u)
#git commit -am "notes"
#git push origin master

#run on HPC or in case where can not pip install
# - first, git clone RyMVPA 
# - then, instead of 'from rymvpa import *' use:
#>import sys
#>sys.path.append('/home/rms620/RyMVPA/rymvpa/') #clone path
#>from __init__ import *

