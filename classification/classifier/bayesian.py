#!/usr/local/python-2.7.5/bin/python

""" bayesian.py
    ---------------
    @author = Daniel Jaung
"""

###############################################################################
############ modules required to compute multinomial naive Bayes###############
###############################################################################

from sklearn.naive_bayes import MultinomialNB
import warnings

###############################################################################
############## class definition for multinomail nb classifier #################
###############################################################################

class Bayesian:
    def __init__(self,epsilon):
        """ function: constructor
            ---------------------
            instantiate a multinomial nb classifier
        """
        self.name = "naive bayes"
        self.mnb = None
        self.label_space = []
        self.epsilon = epsilon

    ###########################################################################
    ############### functions for multinomial nb classification ###############
    ###########################################################################

    def __run_mnb(self,test):
        """ function: run_mnb
            ------------------
            generate list of labels for a @test vector based on self.mnb
            :param test: one feature vector to match to model
            :returns: list of class labels mnb
        """
        probabilities = self.mnb.predict_proba([test.vector])
        class_labels = []
        for i, p in enumerate(probabilities[0]):
            if p > self.epsilon:
                class_labels += self.label_space[i]
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
        warnings.filterwarnings('ignore')
        fv_space = []
        self.label_space = []
        for fv in training:
            fv_space.append(fv.vector)
            self.label_space.append(fv.topics)
        clf = MultinomialNB()
        self.mnb = clf.fit(fv_space, self.label_space)

    def test_model(self,test):
        """ function: test_model
            --------------------
            test classifier against @test set
            :param test: set of feature vectors used to test model
            :returns: accuracy score of the classifier on @test
        """
        warnings.filterwarnings('ignore')
        accuracy = 0.0
        for fv in test:
            labels = self.__run_mnb(fv)
            if len(set(labels) & set(fv.topics)) > 0:
                accuracy += 1.0
        return accuracy
