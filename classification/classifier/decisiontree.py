#!/usr/local/python-2.7.5/bin/python

""" decisiontree.py
    ---------------
    @author = Ankai Lou
"""

###############################################################################
################## modules required to compute decision tree ##################
###############################################################################

from sklearn import tree

###############################################################################
############### class definition for decision tree classifier #################
###############################################################################

class DecisionTree:
    def __init__(self,epsilon):
        """ function: constructor
            ---------------------
            instantiate a decision tree classifier
        """
        self.dt = None
        self.label_space = []
        self.epsilon = epsilon

    ###########################################################################
    ############### functions for decision tree classification ################
    ###########################################################################

    def __run_tree(self,test):
        """ function: run_tree
            ------------------
            generate list of labels for a @test vector based on self.dt

            :param test: one feature vector to match to model
            :returns: list of class labels of decision tree traversal
        """
        probabilities = self.dt.predict_proba([test.vector])
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
        fv_space = []
        self.label_space = []
        for fv in training:
            fv_space.append(fv.vector)
            self.label_space.append(fv.topics)
        clf = tree.DecisionTreeClassifier()
        self.dt = clf.fit(fv_space, self.label_space)

    def test_model(self,test):
        """ function: test_model
            --------------------
            test classifier against @test set

            :param test: set of feature vectors used to test model
            :returns: accuracy score of the classifier on @test
        """
        accuracy = 0.0
        for fv in test:
            labels = self.__run_tree(fv)
            if len(set(labels) & set(fv.topics)) > 0:
                accuracy += 1.0
        return accuracy

