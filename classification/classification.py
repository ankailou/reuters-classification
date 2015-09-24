#!/usr/local/python-2.7.5/bin/python

""" classification.py
    -----------------
    @author = Ankai Lou
"""

import os
import sys
import time
import string
import random
import operator
import math

###############################################################################
##################### modules required for classifications ####################
###############################################################################

import sklearn

###############################################################################
############################### global variables ##############################
###############################################################################

num_partitions = 5
num_neighbors = 3

###############################################################################
############# function(s) for generating training & testing sets ##############
###############################################################################

def filter_empty(feature_vectors):
    """ function: filter_empty
        ----------------------
        separate fv dataset into classified & unclassified data

        :param feature_vectors: dictionary representing feature vector dataset
        :returns: dictionaries of empty and non-empty feature vectors
    """
    empty = dict([])
    nonempty = dict([])
    for document, doc_dict in feature_vectors.iteritems():
        if len(feature_vectors[document]['topics']) == 0:
            empty[document] = feature_vectors[document]
        else:
            nonempty[document] = feature_vectors[document]
    return nonempty, empty

def partition(feature_vectors):
    """ function: partition
        -------------------
        partition set of feature vectors into n equivalent partitions

        :param feature_vectors: dictionary representing feature vector dataset
        :returns: dictionary of n dictionaries whose union is @feature_vectors
    """
    partitions = dict([])
    size = len(feature_vectors) / num_partitions
    # randomize keys
    keys = sorted(feature_vectors.keys(), key=lambda k: random.random())
    parts = [keys[i:i + size] for i in range(0, len(feature_vectors), size)]
    # generate partitions
    for i, part in enumerate(parts):
        partitions[i] = []
        for key in part:
            partitions[i].append(feature_vectors[key])
    return partitions

###############################################################################
##################### function(s) for knn classification ######################
###############################################################################

def decision_tree(partitions):
    """ function: decision_tree
        -----------------------
        use decision-tree classifier via cross-validation on feature vectors

        :param partitions: dictionary representing feature vector dataset
    """

###############################################################################
################# function(s) for decision-tree classification ################
###############################################################################

def euclidean_distance(x1, x2):
    """ function: euclidean_distance
        ----------------------------
        compute the distance between @x1 and @x2

        :param x1: list of dimensions of a vector
        :param x2: list of dimensions of a vector
        :returns: float representing distance between @x1 and @x1
    """
    distance = 0.0
    for i in range(len(x1)):
        distance += pow( x1[i] - x2[i], 2)
    return math.sqrt(distance)

def get_knn(training, test):
    """ function: get_knn
        -----------------
        get k nearest points in @training to the point @test

        :param training: list of feature vectors in the model
        :param test: one feature vector to match to model
        :returns: list of class labels of k nearest neighbors
    """
    distances = []
    for point in training:
        dist = euclidean_distance(test['feature_vector'], point['feature_vector'])
        distances.append((point['topics'], dist))
    distances.sort(key=operator.itemgetter(1))
    class_labels = []
    for x in range(num_neighbors):
        class_labels += distances[x][0]
    return class_labels

def k_nearest_neighbor(partitions):
    """ function: k_nearest_neighbor
        ----------------------------
        perform knn classification via cross-validation on feature vector set

        :param partitions: dictionary representing feature vector dataset
    """
    average_offline = 0.0
    average_online = 0.0
    average_accuracy = 0.0
    # cross-validation across partitions
    for i in xrange(num_partitions):
        test = partitions[i]
        # build model - get offline cost
        offline_start = time.time()
        training = []
        for j in xrange(num_partitions):
            if j != i:
                training += partitions[j]
        offline_total = time.time() - offline_start
        average_offline += offline_total
        print 'Offline cost for trial', i, '-', offline_total
        # test classifier - get online cost
        online_start = time.time()
        accuracy = 0.0
        for fv in test:
            labels = get_knn(training, fv)
            if len(set(labels) & set(fv['topics'])) > 0:
                accuracy += 1.0
        average_accuracy += accuracy / len(test)
        print 'Accuracy for trial', i, '-', accuracy
        online_total = time.time() - online_start
        average_online += online_total
        print 'Online cost for trial', i, '-', online_total
    # compute final statistics
    average_offline /= num_partitions
    average_online /= num_partitions
    average_accuracy /= num_partitions
    print 'Average offline efficiency cost:', average_offline
    print 'Average online efficiency cost:', average_online
    print 'Average accuracy of the classifier:', average_accuracy

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

    # extract out vectors with empty 'topics' class labels
    fv, efv = filter_empty(feature_vectors)
    pfv, epfv = filter_empty(pared_feature_vectors)

    # implement cross validation with n = 5 or n = 10
    fv_partitions = partition(fv)
    pfv_partitions = partition(pfv)

    # knn on @feature_vectors
    print('Experiment: k-nearest-neighbor on standard feature vector...')
    k_nearest_neighbor(fv_partitions)
    # knn on @pared_feature_vectors
    print('Experiment: k-nearest-neighbor on pared down feature vector...')
    k_nearest_neighbor(pfv_partitions)

    # decision-tree on @feature_vectors
    print('Experiment: decision tree on standard feature vector...')
    decision_tree(fv_partitions)
    # decision-tree on @pared_feature_vectors
    print('Experiment: decision tree on pared down feature vector...')
    decision_tree(pfv_partitions)
