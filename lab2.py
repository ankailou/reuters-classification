#!/usr/local/python-2.7.5/bin/python

""" lab2.py
    -------
    @author = Ankai Lou
"""

import os
import sys
import time

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
    start_time = time.time()
    print('Step 1: Preprocessing')
    fv, pfv = preprocessing.begin()
    print('\nStep 2: Classification')
    classification.begin(fv, pfv)
    end_time = time.time() - start_time
    print '\nProcess finished in', end_time, 'seconds!'

if __name__ == '__main__':
    main(sys.argv[1:])
