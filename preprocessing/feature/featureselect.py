#!/usr/local/python-2.7.5/bin/python

""" featureselect.py
    ----------------
    @author = Ankai Lou
"""

###############################################################################
############## modules & libraries for feature vector generation ##############
###############################################################################

import operator

###############################################################################
################# class definition for feature vector object ##################
###############################################################################

class FeatureVector:
    def __init__(self):
        """ function: constructor
            ---------------------
            instantiate feature vector object
        """
        self.vector = []
        self.topics = []
        self.places = []

###############################################################################
############## class definition for feature selector & generator ##############
###############################################################################

class FeatureSelector:
    def __init__(self,weights,documents):
        """ function: constructor
            ---------------------
            select features and generate feature vector sets from @documents

            :param weights: table of (document,word) tf-idf scores
            :param documents: list of document objects
        """
        self.features = []
        self.feature_vectors = []
        # generate feature sets
        self.__select_features(weights)
        # generate standard dataset
        for feature_set in self.features:
            self.__generate_dataset(weights,documents,feature_set)

    ###########################################################################
    ###################### method for feature selection #######################
    ###########################################################################

    def __select_features(self,weights):
        """ function: select_features
            -------------------------
            generate reduced feature for vector generation

            :param weights: table of (document,word) tf-idf scores
            :returns: sorted lists of features and pared features
        """
        features = set()
        scores = dict([])
        # generate normal feature vector
        for doc, doc_dict in weights.iteritems():
            top = dict(sorted(doc_dict.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])
            for term, score in top.iteritems():
                if score > 0.0:
                    features.add(term)
                    scores[term] = score
        # pare down feature vector to 10%
        new_size = len(features) / 10
        pared_features = dict(sorted(scores.iteritems(), key=operator.itemgetter(1), reverse=True)[:new_size])
        # ADD NEW FEATURE SELECTION METHODS HERE
        # sort sets into list
        self.features.append(sorted(features))
        self.features.append(sorted(pared_features))
        # APPEND NEW FEATURE SETS HERE

    ###########################################################################
    ############## method for feature vector dataset generation ###############
    ###########################################################################

    def __generate_dataset(self,weights,documents,features):
        """ function: generate_dataset
            --------------------------
            generate dataset of feature vectors of @features

            :param documents: list of document objects
            :param weights: table of (document,word) tf-idf scores
            :oaram features: list of feature words to select
        """
        dataset = dict([])
        for i, document in enumerate(documents):
            # instantiate new feature vector object
            dataset[i] = FeatureVector()
            # grab tfidf scores
            for feature in features:
                dataset[i].vector.append(weights[i][feature])
            # grab class labels
            dataset[i].topics = document.topics
            dataset[i].topics = document.places
        self.feature_vectors.append(dataset)
