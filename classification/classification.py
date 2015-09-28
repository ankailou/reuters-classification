#!/usr/local/python-2.7.5/bin/python

""" classification.py
    -----------------
    @author = Ankai Lou, Daniel Jaung
"""

###############################################################################
############ modules & libraries required for classifying articles ############
###############################################################################

from crossvalidator.crossvalidator import CrossValidator
from classifier.knearestneighbor import KNN
from classifier.decisiontree import DecisionTree
from classifier.bayesian import Bayesian

###############################################################################
############################### global variables ##############################
###############################################################################

num_partitions = 5
num_neighbors = 3
epsilon = 0.0

###############################################################################
############# function(s) for generating training & testing sets ##############
###############################################################################

def __filter_empty(feature_vectors):
    """ function: filter_empty
        ----------------------
        separate fv dataset into classified & unclassified data

        :param feature_vectors: dictionary representing feature vector dataset
        :returns: dictionaries of empty and non-empty feature vectors
    """
    empty = dict([])
    nonempty = dict([])
    for document, doc_dict in feature_vectors.iteritems():
        if len(feature_vectors[document].topics) == 0:
            empty[document] = feature_vectors[document]
        else:
            nonempty[document] = feature_vectors[document]
    return nonempty, empty

###############################################################################
################# main function for single-point-of-execution #################
###############################################################################

def begin(feature_vectors, pared_feature_vectors):
    """ function: begin
        ---------------
        use knn, decision tree, and naive bayesian classifiers on two feature vector datasets

        :param feature_vectors: standard dataset generated using tf-idf
        :param pared_feature_vectors: pared down version of @feature_vectors
    """
    # extract out vectors with empty 'topics' class labels
    fv, efv = __filter_empty(feature_vectors)
    pfv, epfv = __filter_empty(pared_feature_vectors)
    # implement cross validation with n = 5 or n = 10
    cross_validator = CrossValidator(fv,num_partitions)
    pared_cross_validator = CrossValidator(pfv,num_partitions)
    # knn on @feature_vectors
    print('Experiment: k-nearest-neighbor on standard feature vector...')
    cross_validator.classify(KNN(num_neighbors))
    # knn on @pared_feature_vectors
    print('Experiment: k-nearest-neighbor on pared down feature vector...')
    pared_cross_validator.classify(KNN(num_neighbors))
    # decision-tree on @feature_vectors
    print('Experiment: decision tree on standard feature vector...')
    cross_validator.classify(DecisionTree(epsilon))
    # decision-tree on @pared_feature_vectors
    print('Experiment: decision tree on pared down feature vector...')
    pared_cross_validator.classify(DecisionTree(epsilon))
    # bayesian on @feature vectors
    print('Experiment: bayesian on standard feature vector...')
    cross_validator.classify(Bayesian(epsilon))
    # bayesian on @pared_feature_vectors
    print('Experiment: bayesian on pared down feature vector...')
    pared_cross_validator.classify(Bayesian(epsilon))
