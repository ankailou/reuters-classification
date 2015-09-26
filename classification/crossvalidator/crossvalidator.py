#!/usr/local/python-2.7.5/bin/python

""" crossvalidator.py
    -----------------
    @author = Ankai Lou
"""

###############################################################################
############## modules & libraries required for cross validation ##############
###############################################################################

import time
import random

###############################################################################
################# class definition for cross validator object #################
###############################################################################

class CrossValidator:
    def __init__(self,feature_vectors,num_partitions):
        """ function: constructor
            ---------------------
            instantiate new cross validator object

            :param feature_vectors: set of vectors to cross validate
            :param num_partitions:  number of partitions to split across
        """
        self.partitions = dict([])
        self.num_partitions = num_partitions
        self.__partition(feature_vectors)

    ###########################################################################
    ########## method for generating partitions for cross-validation ##########
    ###########################################################################

    def __partition(self,feature_vectors):
        """ function: partition
        -------------------
        partition set of feature vectors into n equivalent partitions

        :param feature_vectors: dictionary representing feature vector dataset
        :returns: dictionary of n dictionaries whose union is @feature_vectors
        """
        size = len(feature_vectors) / self.num_partitions
        # randomize keys
        keys = sorted(feature_vectors.keys(), key=lambda k: random.random())
        parts = [keys[i:i + size] for i in range(0, len(feature_vectors), size)]
        # generate partitions
        for i, part in enumerate(parts):
            self.partitions[i] = []
            for key in part:
                self.partitions[i].append(feature_vectors[key])

    ###########################################################################
    ######## method for performing cross validation with a classifier #########
    ###########################################################################

    def classify(self,classifier):
        """ function: classify
            ------------------
            perform cross validation across self.partitions

            :param classifier: name of classifier to use (knn, dt, etc)
        """
        average_offline = 0.0
        average_online = 0.0
        average_accuracy = 0.0
        # cross-validation across partitions
        for i in xrange(self.num_partitions):
            test = self.partitions[i]
            # build model - get offline cost
            offline_start = time.time()
            training = self.__build_model(classifier,i)
            offline_total = time.time() - offline_start
            average_offline += offline_total
            print 'Offline cost for trial', i, '-', offline_total, 'seconds'
            # test classifier - get online cost
            online_start = time.time()
            accuracy = classifier.test_model(test)
            average_accuracy += accuracy / len(test)
            print 'Total accuracy of trial', i, '-', accuracy / len(test)
            online_total = time.time() - online_start
            average_online += online_total
            print 'Online cost for trial', i, '-', online_total, 'seconds'
        # compute final statistics
        print 'Average offline efficiency cost:', average_offline / self.num_partitions, 'seconds'
        print 'Average online efficiency cost:', average_online / self.num_partitions, 'seconds'
        print 'Average accuracy of the classifier:', average_accuracy / self.num_partitions

    ###########################################################################
    ############## method for generating a training set & model ###############
    ###########################################################################

    def __build_model(self,classifier,index):
        """ function: build_model
            ---------------------
            generate training data and model from training data

            :param classifier: classifier to use (knn, dt, etc)
            :param index: partition to exclude from training set
            :returns: training set and model (if applicable)
        """
        model = None
        training = []
        # generate training set
        for j in xrange(self.num_partitions):
            if j != index:
                training += self.partitions[j]
        # generate model if applicable (decision tree)
        classifier.build_model(training)
        return training
