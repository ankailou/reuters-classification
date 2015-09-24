#!/usr/local/python-2.7.5/bin/python

""" feature3.py
    -----------
    @author = Ankai Lou
"""

import os
import sys
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import operator

###############################################################################
########## global variables for single-point of control over change ###########
###############################################################################

datafile = 'dataset1.csv'
pared_datafile = 'dataset2.csv'

###############################################################################
############### function for printing dataset to .csv document ################
###############################################################################

def generate_csv(file, documents, features, weights):
    """ function: generate_csv
        ----------------------
        print feature vectors & class labels to .csv file

        :param file: string representing file to write
        :param documents: dictionary of document objects
        :param features: sorted list of features to represent
    """
    dataset = open(file, "w")
    dataset.write('id\t')
    for feature in features:
        dataset.write(feature)
        dataset.write('\t')
    dataset.write('class-label:topics\t')
    dataset.write('class-label:places\t')
    dataset.write('\n')
    # feature vector for each document
    for i, document in enumerate(documents):
        # document id number
        dataset.write(str(i))
        dataset.write('\t')
        # each tf-idf score
        for feature in features:
            dataset.write(str(weights[i][feature]))
            dataset.write('\t')
        # topics/places class labels
        dataset.write(str(document['topics']))
        dataset.write(str(document['places']))
        dataset.write('\n')
    dataset.close()

###############################################################################
###################### function(s) for feature selection ######################
###############################################################################

def select_features(weights):
    """ function: select_features
        -------------------------
        generated reduced feature list for vector generation

        :param weights: dictionary from results of the tf-idf calculations
        :returns: sorted list of terms representing the selected features
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
    # sort sets into list
    return sorted(features), sorted(pared_features)

###############################################################################
############## function(s) for generating weighted tf-idf scores ##############
###############################################################################

def generate_weights(documents):
    """ function: generate_weights
        --------------------------
        perform tf-idf to generate importance scores for words in documents

        :param document: list of documents to use in calculations
        :returns: dictionary of dictionaries: {"id_" : {"word" : score,...}}
    """
    # generate dict for sklearn
    token_dict = dict([])
    for i, document in enumerate(documents):
        token_dict[i] = ' '.join(document['words']['title'] + document['words']['body'])
    # scikit-learn tfidf
    tfidf = TfidfVectorizer()
    weights = tfidf.fit_transform(token_dict.values())
    features = tfidf.get_feature_names()
    return features, weights

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
    print '\nGenerating dataset @', datafile
    words, weights = generate_weights(documents)

    # generate dictionary for feature selection
    weight_array = weights.toarray()
    weight_dict = dict([])
    for i, row in enumerate(weight_array):
        weight_dict[i] = dict([])
        for j, word in enumerate(words):
            weight_dict[i][word] = weight_array[i][j]

    # generate feature list
    print 'Selecting features for the feature vectors @', datafile
    features, pared_features = select_features(weight_dict)

    # write vectors to dataset1.csv
    print 'Writing feature vector data @', datafile
    generate_csv(datafile, documents, features, weight_dict)
    print 'Finished generating dataset @', datafile

    # write pared vectors to dataset2.csv
    print 'Writing feature vector data @', pared_datafile
    generate_csv(pared_datafile, documents, pared_features, weight_dict)
    print 'Finished generating dataset @', pared_datafile

