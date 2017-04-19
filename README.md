RyMVPA
=========

Wrappers &amp; additions to PyMVPA http://www.pymvpa.org/ (I swear we will pull request one day)

More description at: https://rystoli.github.io/#two

##Purpose
These are wrappers and additional methods for PyMVPA as used by our lab. Wrappers reduce various analyses and output to single functions, and provide an assortment of useful tools for managing pymvpa datasets. Additional methods at this point are primarily unique methods of representational similarity analysis.  

*We use this system by first initializing a python environment with this module imported, but importantly, data loaded as a dictionary where keys are subject IDs and values subject PyMVPA datasets ('datadicts'). Many of these functions are made to operate on these dictionaries and analyze all subjects at once. 
##Contents

###Files
* importer.py - imports necessary modules for others to run
* rsa_rymvpa - our additional RSA methods (eg, do individual differences predict similarity of patterns between two conditions? and more)
* datamanage.py - etc. functions for handling datasets (e.g., saving, masking, printing attributes to CSV, making ROI masks, and more)
* searchlight_wraps.py - wrapper functions performing searchlights with various methods (classification, RSA, and our additional RSA methods)
* roi_wraps.py - wrapper functions performing various methods on specified ROI via ROI mask (classificaiton, RSA, our additional RSA methods)

###Functions
See documentation nested in functions for now. 

Thanks Zach!
