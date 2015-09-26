#!/usr/local/python-2.7.5/bin/python

""" lexicon.py
    ----------
    @author = Ankai Lou
"""

###############################################################################
##################### class definition for lexicon object #####################
###############################################################################

class Lexicon:
    def __init__(self,documents):
        """ function: constructor
            ---------------------
            instantiate lexicon object

            :param documents: list of document objects used to build lexicon
        """
        self.title = set()
        self.body = set()
        self.__build_lexicon(documents)

    ###########################################################################
    ############# method to populate title/body sets for lexicon ##############
    ###########################################################################

    def __build_lexicon(self,documents):
        """ function: build_lexicon
            -----------------------
            populate word sets for title/body given list of @documents

            :param documents: list of document objects used to build lexicon
        """
        for document in documents:
            for term in document.words.title:
                self.title.add(term)
            for term in document.words.body:
                self.body.add(term)
