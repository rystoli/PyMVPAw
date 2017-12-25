PyMVPAw
=========

Wrappers &amp; additions to [PyMVPA](http://www.pymvpa.org/) (I swear we will pull request one day)

More description [here](https://rystoli.github.io/#two).

## Purpose
These are wrappers and additional methods for PyMVPA as used by our lab. Wrappers reduce various analyses and output to single functions, and provide an assortment of useful tools for managing pymvpa datasets. Additional methods at this point are primarily unique methods of representational similarity analysis.  PyMVPAw</a> is an unnecessarily eponymous wrapper for <a href="http://www.pymvpa.org/">PyMVPA</a>. PyMVPA is a wonderful module for MVPA, multi-variate pattern analysis, of data - especially, in my case, fMRI data. In PyMVPAw, many of PyMVPA's pattern analysis tools are available in single function commands to make the magic of PyMVPA less verbose. For instance, you can run an entire searchlight multiple regression Representational Similarity Analysis with a single line of code.
little this:

```
slRSA_m_nSs( data, target_DM, [control_DM_1 ... control_DM_k] )
```

It also comes with many additional methods and tools you may find useful:
* Perform basic MVPA Classification and RSA analyses in single lines of code (within ROIs, searchlights, for single or all subjects)
* Perform RSA in a multiple regression (controlling for and including additional models), only assessing certain similarity values (of the DM), or compare similarities of specific condition-pairs directly
* See if between-subject individual differences or trial-by-trial covariates relate to the neural similarity of certain conditions
* Train your classifier on certain targets, and test on others

*We use this system by first initializing a python environment with this module imported, but importantly, data loaded as a dictionary where keys are subject IDs and values subject PyMVPA datasets ('datadicts'). Many of these functions are made to operate on these dictionaries and analyze all subjects at once.*

## Installation

Requirements are:
* [PyMVPA base and dependencies](http://www.pymvpa.org/download.html) ([See this guide for easy installation on Mac and Windows](https://rystoli.github.io/blog/9_27_17.html)).
* Also make sure [nilearn](http://nilearn.github.io/) and [nltools](http://neurolearn.readthedocs.io/en/latest/) are installed too.

1. Download or clone this repository: https://github.com/rystoli/PyMVPAw
2. Use [pip](https://packaging.python.org/tutorials/installing-packages/)!
```
pip install [path to PyMVPAw directory]
```
3. Import as a whole in Python
```
from rymvpa import *
```

## Contents

* importer.py - imports necessary modules for others to run
* rsa_rymvpa.py - our additional RSA methods (eg, do individual differences predict similarity of patterns between two conditions? and more)
* datamanage.py - etc. functions for handling datasets (e.g., saving, masking, printing attributes to CSV, making ROI masks, and more)
* searchlight_wraps.py - wrapper functions performing searchlights with various methods (classification, RSA, and our additional RSA methods)
* roi_wraps.py - wrapper functions performing various methods on specified ROI via ROI mask (classificaiton, RSA, our additional RSA methods)

## Use
See documentation nested in functions for now. Also, see tutorials.

--------
Thank to Zach Ingbretsen for plenty contribution and help!
