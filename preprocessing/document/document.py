#!/usr/local/python-2.7.5/bin/python

""" document.py
    -----------
    @author = Ankai Lou
"""

###############################################################################
############### modules & libraries required for text extraction ##############
###############################################################################

import os
import sys
import nltk
import string
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

###############################################################################
###################### class definition for word list #########################
###############################################################################

class WordList:
    def __init__(self):
        """ function: constructor
            ---------------------
            instantiate word list object
        """
        self.title = []
        self.body = []

###############################################################################
##################### class definition for document object ####################
###############################################################################

class Document:
    def __init__(self,article):
        """ function: constructor
            ---------------------
            instantiate document object;

            :param article: bs4 child node of parse tree representing article;
        """
        # instantiate document fields
        self.topics = []
        self.places = []
        self.words = WordList()
        # populate document fields
        self.__populate_class_labels(article)

    ###########################################################################
    ############ class label words not permitted in feature vector ############
    ###########################################################################

    banned_words = set()

    ###########################################################################
    ######### method to pre-populate class labels for document object #########
    ###########################################################################

    def __populate_class_labels(self,article):
        """ function: populate_class_labels
            -------------------------------
            fill self.topics and self.places; update @banned_words concurrently;

            :param article: bs4 child node of parse tree representing article;
        """
        for topic in article.topics.children:
            topic_label = topic.text.encode('ascii','ignore')
            self.topics.append(topic_label)
            Document.banned_words.add(topic_label)
        for place in article.places.children:
            place_label = place.text.encode('ascii','ignore')
            self.places.append(place_label)
            Document.banned_words.add(place_label)

    ###########################################################################
    ######## method(s) for post-populating word lists w/o class labels ########
    ###########################################################################

    def populate_word_list(self,article):
        """ function: populate_word_list
            ----------------------------
            generate token list from title/body text blocks of @article
            non-private method - must execute after @banned_words is filled

            :param article: bs4 child node of parse tree representing article
        """
        text = article.find('text')
        title = text.title
        body = text.body
        if title != None:
            self.words.title = self.__tokenize(title.text)
        if body != None:
            self.words.body = self.__tokenize(body.text)

    def __tokenize(self,text):
        """ function: tokenize
            ------------------
            generate list of tokens given a block of @text

            :param text: string representing article text field
            :returns: list of tokens with various modifications
        """
        ascii = text.encode('ascii', 'ignore')
        # remove digits & punctuation
        no_digits = ascii.translate(None, string.digits)
        no_punctuation = no_digits.translate(None, string.punctuation)
        # separate text blocks into tokens
        tokens = nltk.word_tokenize(no_punctuation)
        # remove class labels, stopwords, and non-english words
        no_class_labels = [w for w in tokens if not w in Document.banned_words]
        no_stop_words = [w for w in no_class_labels if not w in stopwords.words('english')]
        eng = [y for y in no_stop_words if wordnet.synsets(y)]
        # lemmatization
        lemmas = []
        lmtzr = WordNetLemmatizer()
        for token in eng:
           lemmas.append(lmtzr.lemmatize(token))
        # stemming
        stems = []
        stemmer = PorterStemmer()
        for token in lemmas:
            stem = stemmer.stem(token).encode('ascii', 'ignore')
            if len(stem) >= 4:
                stems.append(stem)
        return stems
