Report 2 - Classification
=========================


## Group Members & Contributions

* Ankai Lou
    * Cleaned up and reorganized code from preprocessing lab to be more maintainable
    * Implemented feature vector generation and feature paring to create two different feature vectors
    * Implemented functionality for knn-classification from scratch
    * Implemented functionality for decision-tree classification using scikit-learn
    * Documented all code and README.md

## Problem Statement

* Predict the TOPICS class labels of test set of feature vectors representing Reuters articles
* Generate two sets of feature vector datasets (one being a pared down version of the other)
* Implement two different classifiers for the feature vectors datasets
* Test the scalability, cost, and accuracy of the 2x2 set of experiments

## Proposed Solution

* The first feature vector was selected using the following methodology:
    * Remove stopwords and class label words from article body/title text
    * Tokenize article body/title text
    * Lemmatization and Stemming
    * Removal of sufficiently short stems
    * Compute TF-IDF scores for each document/word pair
    * Select top 5 words from each document to be appended to feature vector
* The second feature vector was selected using the following methodology:
    * Use standard feature vector previously generated as a foundation
    * Pare down the number of features to 10% of the original size
    * Select features by taking the top 10% of words by TF-IDF score
* Implement K-Nearest-Neighbors classification:
    * Cross validation was used with k = 5
    * Distances and nearest neighbors are computed with n = 3
    * Union of class labels for neighbors is used as classification
* Implement Decision-Tree classification:
    * Cross validation was used with k = 5
    * Decision tree model was generated using sklearn.tree
    * Tree used to predict probabilities of different classifications
    * Union of labels with probabilities greater than epsilon = 0.000001 are used
* The 2x2 Experiements were performed:
    * KNN using the standard feature vector
    * KNN using the pared-down feature vector
    * Decision-Tree using the standard feature vector
    * Decision-Tree using the pared-down feature vector      

## Explanation & Rationale of Classification

* The rationale for the methodology used in generating the first feature vectors is that the top 5 words for each article hold valuable context for that specific article - with the tf-idf score being a sufficient measure of a word's normalized value to its corresponding document. Any fewer features run the risk of losing valuable information that may impact the accuracy of classification.
* The second feature vector was specifically chosen to be 10% the size of the original feature vector in order to allow the feature vectors to differ in size by an order of magnitude - this allows for sufficient granularity/precision when comparing the performance of the two datasets.

* Cross-validation was implement with `n = 5` for both classifiers because it is far more robust and accurate in determining a classifier's cost and accuracy than a simple 2-way split. 
* The KNN classifier was chosen because it is the easier classifier to implement. KNN comes with a very small offline cost and a very large online cost - with accuracy being determined by how many neighbors are computed (which also impacts online cost). `k = 3` neighbors were computed in order to minimize the online cost without the accuracy hit the `k = 1` possesses.
* The Decision-Tree classifier was chosen because it has a far more sophisticated model than the KNN-classifier. The Decision-Tree classifiers comes with a much larger offline cost (to generate the model) and a small online cost (because running the model is faster than computing an NxN array of distances). The Decision-Tree classifier and KNN has very significant differences.

* The selection of feature vectors and classifiers were all chosen in order to maximize the different between the four experiments executed - more granularity corresponds to better quality information.

## Resources

* This module was implemented for Python 2.7.5 and tested on OS X 10.10.4 and stdlinux
* The following libraries were used in the implementation of this module:
    * os
    * sys
    * time
    * string
    * random
    * math
    * operator
    * BeautifulSoup4
    * nltk
        * nltk.stem.porter.*
        * nltk.corpus.stopwords
        * nltk.corpus.wordnet
        * nltk.stem.wordnet.WordNetLemmatizer 
    * scikit-learn
        * sklearn.feature_extraction.text.TfidfVectorizer
        * sklearn.tree.DecisionTreeClassifier 

## Experiments & Results

Note that the results from the classification step will usually differ between iterations because the cross validation partitioning is randomized. Thus, accuracies can range from abysmal to passable. The following results were computed for each of the 2x2 experiments on the full 21 file Reuters dataset:

### Offline Cost (Scalability)

* For the KNN on the standard feature vector, across 5 iterations of cross-validation,
* For the KNN on the pared feature vector, across 5 iterations of cross-validation,
* For the decision tree on the standard feature vector, across 5 iterations of cross-validation,
* For the decision tree on the pared feature vector, across 5 iterations of cross-validation,

### Online Cost (Classification Time)

* For the KNN on the standard feature vector, across 5 iterations of cross-validation,
* For the KNN on the pared feature vector, across 5 iterations of cross-validation,
* For the decision tree on the standard feature vector, across 5 iterations of cross-validation,
* For the decision tree on the pared feature vector, across 5 iterations of cross-validation,

### Accuracy of Classification

* For the KNN on the standard feature vector, across 5 iterations of cross-validation,
* For the KNN on the pared feature vector, across 5 iterations of cross-validation,
* For the decision tree on the standard feature vector, across 5 iterations of cross-validation,
* For the decision tree on the pared feature vector, across 5 iterations of cross-validation,

## Interpretation of Output

### Benefits

### Detriments