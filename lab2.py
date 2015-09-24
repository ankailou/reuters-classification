#!/usr/local/python-2.7.5/bin/python

""" lab2.py
    -------
    @author = Ankai Lou
"""

import os
import sys

###############################################################################
####### modules for the KDD process (proprocessing, classification, etc) ######
###############################################################################

from preprocessing import preprocessing
from classification import classification

###############################################################################
################## main function = single point of execution ##################
###############################################################################

def main(argv):
    """ function: main
        --------------
        KDD process for Reuter article database

        :param argv: commend line arguments
    """
    print('Step 1: Preprocessing')
    fv, pfv = preprocessing.begin()

    # UNCOMMENT WHEN DEBUGGING
    # print "feature vectors: ", feature_vectors
    # print "pared down feature vectors: ", pared_feature_vectors

    print('\nStep 2: Classification')
    classification.begin(fv, pfv)

if __name__ == '__main__':
    main(sys.argv[1:])
