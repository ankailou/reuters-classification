#!/usr/local/python-2.7.5/bin/python

""" knearestneighbor.py
    -------------------
    @author = Ankai Lou
"""

###############################################################################
############### modules required to compute k-nearest neighbors ###############
###############################################################################

import math
import operator

###############################################################################
############# class definition for k-nearest neighbor classifier ##############
###############################################################################

class KNN:
    def __init__(self,num_neighbors):
        """ function: constructor
            ---------------------
            instantiate a knn classifier
        """
        self.name = "k-nearest-neighbors"
        self.training = None
        self.num_neighbors = num_neighbors

    ###########################################################################
    #################### functions for knn classification #####################
    ###########################################################################

    def __euclidean_distance(self,x1, x2):
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

    def __get_knn(self,test):
        """ function: get_knn
            -----------------
            get k nearest points in @training to the point @test

            :param test: one feature vector to match to model
            :returns: list of class labels of k nearest neighbors
        """
        distances = []
        for point in self.training:
            dist = self.__euclidean_distance(test.vector,point.vector)
            distances.append((point.topics,dist))
        distances.sort(key=operator.itemgetter(1))
        class_labels = []
        for x in range(self.num_neighbors):
            class_labels += distances[x][0]
        return class_labels

    ###########################################################################
    ############### main functions to generate & test the model ###############
    ###########################################################################

    def build_model(self,training):
        """ function: build_model
            ---------------------
            generate model necessary for the classifier

            :param training: set of feature vectors used to construct model
        """
        self.training = training

    def test_model(self,test):
        """ function: test_model
            --------------------
            test classifier against @test set

            :param test: set of feature vectors used to test model
            :returns: accuracy score of the classifier on @test
        """
        accuracy = 0.0
        for fv in test:
            labels = self.__get_knn(fv)
            if len(set(labels) & set(fv.topics)) > 0:
                accuracy += 1.0
        return accuracy

