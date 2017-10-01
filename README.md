RyMVPA
=========

Wrappers &amp; additions to [PyMVPA](http://www.pymvpa.org/) (I swear we will pull request one day)

More description [here](https://rystoli.github.io/#two).

## Purpose
These are wrappers and additional methods for PyMVPA as used by our lab. Wrappers reduce various analyses and output to single functions, and provide an assortment of useful tools for managing pymvpa datasets. Additional methods at this point are primarily unique methods of representational similarity analysis.  

*We use this system by first initializing a python environment with this module imported, but importantly, data loaded as a dictionary where keys are subject IDs and values subject PyMVPA datasets ('datadicts'). Many of these functions are made to operate on these dictionaries and analyze all subjects at once.*

## Installation

Requirements are:
* [PyMVPA base and dependencies](http://www.pymvpa.org/download.html) ([See this guide for easy installation on Mac and Windows](http://www.pymvpa.org/download.html)).
* For plotting functions, also make sure [nilearn](http://nilearn.github.io/) and [nltools](http://neurolearn.readthedocs.io/en/latest/) are installed too. (See tutorials directory for plotting tips).

1. Download or clone this repository: https://github.com/rystoli/RyMVPA
2. Use [pip](https://packaging.python.org/tutorials/installing-packages/)!
```
pip install [path to RyMVPA directory]
```
3. Import as a whole in Python
```
from rymvpa import *
```

## Contents

* importer.py - imports necessary modules for others to run
* rsa_rymvpa - our additional RSA methods (eg, do individual differences predict similarity of patterns between two conditions? and more)
* datamanage.py - etc. functions for handling datasets (e.g., saving, masking, printing attributes to CSV, making ROI masks, and more)
* searchlight_wraps.py - wrapper functions performing searchlights with various methods (classification, RSA, and our additional RSA methods)
* roi_wraps.py - wrapper functions performing various methods on specified ROI via ROI mask (classificaiton, RSA, our additional RSA methods)

## Use
See documentation nested in functions for now. Also, see tutorials.

--------
Thanks Zach!
