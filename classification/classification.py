#!/usr/local/python-2.7.5/bin/python

""" classification.py
    -----------------
    @author = Ankai Lou
"""

import os
import sys
import time
import string
import sklearn

###############################################################################
############# function(s) for generating training & testing sets ##############
###############################################################################

def filter_empty(feature_vectors):
    """ function: filter_empty
        ----------------------
        separate fv dataset into classified & unclassified data

        :param feature_vectors: dictionary representing feature vector dataset
    """

###############################################################################
##################### function(s) for knn classification ######################
###############################################################################

def decision_tree(feature_vectors):
    """ function: decision_tree
        -----------------------
        use decision-tree classifier via cross-validation on feature vectors

        :param feature_vectors: dictionary representing feature vector dataset
    """

###############################################################################
################# function(s) for decision-tree classification ################
###############################################################################

def k_nearest_neighbor(feature_vectors):
    """ function: k_nearest_neighbor
        ----------------------------
        perform knn classification via cross-validation on feature vector set

        :param feature_vectors: dictionary representing feature vector dataset
    """

###############################################################################
################# main function for single-point-of-execution #################
###############################################################################

def begin(feature_vectors, pared_feature_vectors):
    """ function: begin
        ---------------
        use knn & decision tree classifiers on two feature vector datasets

        :param feature_vectors: standard dataset generated using tf-idf
        :param pared_feature_vectors: pared down version of @feature_vectors
    """

    # TODO: extract out vectors with empty 'topics' class labels

    # TODO: knn on @feature_vectors
    # TODO: knn on @pared_feature_vectors

    # TODO: decision-tree on @feature_vectors
    # TODO: decision-tree on @pared_feature_vectors
