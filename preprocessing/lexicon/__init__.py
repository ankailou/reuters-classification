""" __init__.py
    -----------
    @author = Ankai Lou
"""

###############################################################################
########### modules & libraries required to allow dynamic importing ###########
###############################################################################

from os.path import dirname, basename, isfile
import glob

###############################################################################
############ code to specify __all__ variable for wildcard imports ############
###############################################################################

modules = glob.glob(dirname(__file__) + "/*.py")
__all__ = [basename(f)[:-3] for f in modules if isfile(f)]
