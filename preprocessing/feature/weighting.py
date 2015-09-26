#!/usr/local/python-2.7.5/bin/python

""" weighting.py
    ------------
    @author = Ankai Lou
"""

###############################################################################
############## modules & libraries required for tf-idf analysis ###############
###############################################################################

import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

###############################################################################
################## class definition for weight table object ###################
###############################################################################

class WeightTable:
    def __init__(self,documents):
        """ function: constructor
            ---------------------
            instantiate weight table object

            :param documents: list of documents used to compute weights
        """
        self.table = dict([])
        self.__generate_weights(documents)

    ###########################################################################
    ######## methods for computing tf-idf and populating weight table #########
    ###########################################################################

    def __generate_weights(self,documents):
        """ function: generate_weights
            --------------------------
            populate the weight table with tfidf scores

            :param documents: list of document used to compute weights
        """
        words, weights = self.__tfidf(documents)
        weight_array = weights.toarray()
        for document, row in enumerate(weight_array):
            # instantiate new dictionary for each document
            self.table[document] = dict([])
            for i, word in enumerate(words):
                self.table[document][word] = weight_array[document][i]

    def __tfidf(self,documents):
        """ function: tfidf
            ---------------
            calculate tf-idf score for each (document,word) pair
        """
        token_dict = dict([])
        for i, document in enumerate(documents):
            token_dict[i] = ' '.join(document.words.title + document.words.body)
        # compute tfidf using scikit-learn library
        tfidf = TfidfVectorizer()
        weights = tfidf.fit_transform(token_dict.values())
        features = tfidf.get_feature_names()
        return features, weights
