#!/usr/local/python-2.7.5/bin/python

""" feature.py
    ----------
    @author = Ankai Lou
"""

###############################################################################
############## module & libraries required for printing datasets ##############
###############################################################################

import os

###############################################################################
######### modules for weighting, feature selection & feature vectors ##########
###############################################################################

from weighting import WeightTable
from featureselect import FeatureSelector

###############################################################################
########## global variables for single-point of control over change ###########
###############################################################################

datafile = ['datasets/dataset1.csv', 'datasets/dataset2.csv']

###############################################################################
############### function for printing dataset to .csv document ################
###############################################################################

def __generate_csv(file, features, feature_vectors):
    """ function: generate_csv
        ----------------------
        print feature vectors & class labels to .csv file
        concurrently generating feature vector dictionary

        :param file: string representing file to write
    """
    # generate path if necessary
    path = os.path.join(os.getcwd(), 'datasets')
    if not os.path.isdir(path):
        os.makedirs(path)
    # generate dataset & feature vector dict
    dataset = open(file, "w")
    dataset.write('id\t')
    for feature in features:
        dataset.write(feature)
        dataset.write('\t')
    dataset.write('class-label:topics\t')
    dataset.write('class-label:places\t')
    dataset.write('\n')
    # feature vector for each document
    for i, feature_vector in feature_vectors.iteritems():
        # document id number
        dataset.write(str(i))
        dataset.write('\t')
        # each tf-idf score
        for score in feature_vector.vector:
            dataset.write(str(score))
            dataset.write('\t')
        # generate topic/places in fv
        dataset.write(str(feature_vector.topics))
        dataset.write(str(feature_vector.places))
        dataset.write('\n')
    dataset.close()

###############################################################################
################ main function for generating refined dataset #################
###############################################################################

def generate(documents, lexicon):
    """ function: generate
        ------------------
        select features from @lexicon for feature vectors
        generate dataset of feature vectors for @documents

        :param documents: list of well-formatted, processable documents
        :param lexicon:   list of word stems for selecting features
    """
    print 'Generating feature vectors & datasets...'
    weights = WeightTable(documents)
    # generate feature list
    print('Selecting features for the feature vectors...')
    selector = FeatureSelector(weights.table,documents)
    # write feature vectors to csv files
    for i, feature in enumerate(selector.features):
        print 'Writing feature vector data @', datafile[i]
        __generate_csv(datafile[i], selector.features[i], selector.feature_vectors[i])
        print 'Finished generating dataset @', datafile[i]
    return selector.feature_vectors
