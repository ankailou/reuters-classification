Report 2 - Classification
=========================


## Group Members & Contributions

* Ankai Lou
    * Provided an API for expanding the code 
    * Refactored the code to allow for simple addition of feature vectors & classifiers
    * Cleaned up and reorganized code from preprocessing lab to be more maintainable
    * Implemented feature vector generation and feature paring to create two different feature vectors - one standard, one pared-down
    * Implemented functionality for knn-classification from scratch
    * Implemented functionality for decision-tree classification using scikit-learn
    * Documented all code and README.md

* Daniel Jaung
    * Implemented functionality for multinomial naive bayes classification using scikit-learn
    * Updated documentation

## Problem Statement

* Predict the TOPICS class labels of test set of feature vectors representing Reuters articles
* Generate two sets of feature vector datasets (one being a pared down version of the other)
* Implement three different classifiers for the feature vectors datasets
* Test the scalability, cost, and accuracy of the 3x2 set of experiments

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
* Implement Multinomial Naive Bayes classification:
    * Cross validation was used with k = 5
    * Decision tree model was generated using sklearn.naive_bayes
    * Event model used to predict probabilities of different classifications
* The 3x2 Experiements were performed:
    * KNN using the standard feature vector
    * KNN using the pared-down feature vector
    * Decision-Tree using the standard feature vector
    * Decision-Tree using the pared-down feature vector      
    * Naive Bayes using the standard feature vector
    * Naive Bayes using the pared-down standard feature vector

## Explanation & Rationale of Classification

* The rationale for the methodology used in generating the first feature vectors is that the top 5 words for each article hold valuable context for that specific article - with the tf-idf score being a sufficient measure of a word's normalized value to its corresponding document. Any fewer features run the risk of losing valuable information that may impact the accuracy of classification.
* The second feature vector was specifically chosen to be 10% the size of the original feature vector in order to allow the feature vectors to differ in size by an order of magnitude - this allows for sufficient granularity/precision when comparing the performance of the two datasets.

* Cross-validation was implement with `n = 5` for both classifiers because it is far more robust and accurate in determining a classifier's cost and accuracy than a simple 2-way split. 
* The KNN classifier was chosen because it is the easier classifier to implement. KNN comes with a very small offline cost and a very large online cost - with accuracy being determined by how many neighbors are computed (which also impacts online cost). `k = 3` neighbors were computed in order to minimize the online cost without the accuracy hit the `k = 1` possesses.
* The Decision-Tree classifier was chosen because it has a far more sophisticated model than the KNN-classifier. The Decision-Tree classifiers comes with a much larger offline cost (to generate the model) and a small online cost (because running the model is faster than computing an NxN array of distances). 
* The Multinomial Naive Bayes classifier was chosen as its model computation lends itself to document classification. It builds an event model using word frequency to calculate the probability of a feature. However, MNB can also work well using tf-idf rather than word frequency, which is the case here. MNB operates using Bayes Theorem with the 'naive' assupmtion that each feature is conditionally independent.

* The selection of feature vectors and classifiers were all chosen in order to maximize the different between the six experiments executed - more granularity corresponds to better quality information.

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
        * sklearn.naive_bayes.MultinomialNB

## Experiments & Results

Note that the results from the classification step will usually differ between iterations because the cross validation partitioning is randomized. Thus, accuracies can range from abysmal to passable. The following results were computed for each of the 3x2 experiments on 1 file in the Reuters dataset:

### Offline Cost (Scalability)

* For the KNN on the standard feature vector, across 5 iterations of cross-validation, the offline costs were 3.81469726562e-06, 1.28746032715e-05, 1.28746032715e-05, 5.91278076172e-05, and 1.62124633789e-05 seconds.
* For the KNN on the pared feature vector, across 5 iterations of cross-validation, the offline costs were 9.05990600586e-06, 1.00135803223e-05, 2.19345092773e-05, 1.00135803223e-05, and 1.00135803223e-05 seconds. 
* For the decision tree on the standard feature vector, across 5 iterations of cross-validation, the offline costs were 0.113656997681, 0.122953176498, 0.110245943069, 0.101885080338, and 0.118744134903 seconds. 
* For the decision tree on the pared feature vector, across 5 iterations of cross-validation, the offline costs were 0.0163049697876, 0.0159809589386, 0.0168070793152, 0.0148220062256, and 0.0160479545593 seconds.
* For the multinomial naive bayes on the standard feature vector, across 5 iterations of cross-validation, the offline costs were 0.106421947479, 0.0985600948334, 0.10511803627, 0.0990388393402, and 0.0975611209869 seconds.
* For the multinomial naive bayes on the pared feature vector, across 5 iterations of cross-validation, the offline costs were 0.0163910388947, 0.0164449214935, 0.017884016037, 0.0181641578674, and 0.0168759822845 seconds.

The averages of the 3x2 experiment set is represented with the table:

     | Feature Vector 1     | Feature Vector 2
---- | -------------------- | -----------------
KNN  | 2.09808349609e-05 s  | 1.220703125e-05 s
Tree | 0.113497066498 s     | 0.0159925937653 s
MNB  | 0.101340007782 s     | 0.0171520233154 s

From here, we can make the following observations:

* Paring down the feature vector does not lead to a significant change in offline cost for the KNN-classifier; this is because model construction for KNN is simply adding the training space into a list - which is not dependent on the number of features.
* Paring down the feature vector leads to a reduction in offline cost for the decision-tree classifier by an order of magnitude consistent with how much the features were reduced; this is because the relative height of the decision tree is linearly dependent on the number of features in the feature vector. Thus, there is a gain in scalability when a pared down feature vector is used for the decision tree.
* Paring down the feature vector with the MNB-classifier leads to a comparable reduction in offline cost as with the decision tree. The time it takes to build the model scales with the number of features. 
* In general, the offline costs for the decision tree and the MNB-classifier were greater than the offline cost for the KNN-classifier by 3 to 4 orders of magnitude. This means that model construction is far simpler and faster for the KNN-classifier.   

### Online Cost (Classification Time)

* For the KNN on the standard feature vector, across 5 iterations of cross-validation, the online costs were 27.4860038757, 29.0641348362, 30.7863907814, 29.2113089561, and 30.0259530544 seconds.
* For the KNN on the pared feature vector, across 5 iterations of cross-validation, the online costs were 3.48056292534, 3.6446480751, 3.68867111206, 3.61712503433, and 3.38688397408 seconds.
* For the decision tree on the standard feature vector, across 5 iterations of cross-validation, the online costs were 0.0267839431763, 0.020122051239, 0.0214660167694, 0.0195162296295, and 0.0245912075043 seconds.
* For the decision tree on the pared feature vector, across 5 iterations of cross-validation, the online costs were 0.00739312171936, 0.00783205032349, 0.00714898109436, 0.00731086730957, and 0.0107901096344 seconds.
* For the multinomial naive bayes on the standard feature vector, across 5 iterations of cross-validation, the online costs were 0.0472280979156, 0.0454790592194, 0.0478901863098, 0.0472888946533, and 0.0467879772186 seconds.
* For the multinomial naive bayes on the pared feature vector, across 5 iterations of cross-validation, the online costs were 0.0194499492645, 0.0200910568237, 0.0194730758667, 0.0190689563751, and 0.0192909240723 seconds.

The averages of the 3x2 experiment set is represented with the table:

     | Feature Vector 1  | Feature Vector 2
---- | ----------------- | -----------------
KNN  | 29.3147583008 s   | 3.56357822418 s
Tree | 0.0224958896637 s | 0.00809502601624 s
MNB  | 0.0469348430634 s | 0.0194747924805 s
From here, we can make the following observations:

* Paring down the feature vector improves the online cost for KNN by an order of magnitude. This is because there are far fewer computations required to compute Euclidean distance. In general, the KNN classifier for both the standard and pared-down feature vector are both abysmal in terms of online performance - especially when considering scalability (as the online cost scales with respect to both the dataset size and the feature vector size).
* Paring down the feature vector improves the online cost for the decision tree by an order of magnitude. This is because the height of the tree is dependent on the feature vector size; thus, a single traversal of the tree also scales linearly with respect to the features.
* Paring down the feature vector improves the online cost for the MNB classifier by about half. While slower, it performed at similar speeds as the the decision tree classifier, as the MNB classifier also scales linearly with the number of features.
* The online cost of the decision-tree classifier is better than the online cost of the KNN-classifier by 3 to 4 orders of magnitude. This is because the decision tree model is far more sophisticated than the KNN model and requires far few computations/comparisons to classify a feature vector. When scaled up to a larger dataset - online cost becomes the primary metric for determining speed (since offline cost is small for both classifiers).

### Accuracy of Classification

* For the KNN on the standard feature vector, across 5 iterations of cross-validation, the accuracies of the classifier were 0.835978835979, 0.804232804233, 0.84126984127, 0.830687830688, and 0.857142857143 across each of the iterations.
* For the KNN on the pared feature vector, across 5 iterations of cross-validation, the accuracies of the classifier were 0.777777777778, 0.708994708995, 0.746031746032, 0.772486772487, and 0.798941798942 percent across each of the iterations.
* For the decision tree on the standard feature vector, across 5 iterations of cross-validation, the accuracies of the classifier were 0.137566137566, 0.0634920634921, 0.730158730159, 0.0634920634921, and 0.756613756614 percent each of the iterations.
* For the decision tree on the pared feature vector, across 5 iterations of cross-validation, the accuracies of the classifier were 0.767195767196, 0.724867724868, 0.47619047619, 0.740740740741, 0.772486772487 across each of the iterations.
* For the multinomial naive bayes on the standard feature vector, across 5 iterations of cross-validation, the accuracies of the classifier were 0.94708994709, 0.888888888889, 0.952380952381, 0.931216931217, and 0.94708994709 across each of the iterations.
* For the multinomial naive bayes on the pared feature vector, across 5 iterations of cross-validation, the accuracies of the classifier were 0.936507936508, 0.830687830688, 0.904761904762, 0.904761904762, and 0.915343915344 across each of the iterations.

The averages of the 3x2 experiment set is represented with the table:

     | Feature Vector 1  | Feature Vector 2
---- | ----------------- | -----------------
KNN  | 0.833862433862    | 0.760846560847
Tree | 0.350264550265    | 0.696296296296
MNB  | 0.933333333333    | 0.898412698413 

From here, we can make the following observations:

* Paring down the feature vector leads to the slight performance drop in the KNN-classifier. This is because loss of dimensionality leads to less granular data - leading to less precise classifications of the testing data.
* Paring down the feature vector seems to lead to a performance improvement in the decision-tree classifier - perhaps this is due to overfitting with regards to the standard feature vector.
* Paring down the feature vector leads to a slight drop in accuracy for the MNB-classifier correlating with the reduction with sample size. However, the accuracy remains the highest of the classifiers.
* The KNN-classifier and MNB-classifier performed far better and far more consistently than the decision-tree classifier. This is likely due to the breadth of the computation that KNN and MNB perform compared to the decision-tree.

## Interpretation of Output

* The offline costs for the KNN-classifier, the decision-tree classifier, and the MNB-classifier were very small - even when scaled up to a larger dataset. This means that offline cost is not a good metric for measuring classifier quality in this case.
* Paring down the feature vector affected the offline cost for both the decision tree and MNB-classifier by an order of magnitude - which means feature reduction is good for scalability for both classifiers. The KNN classifier was not affected.

* The online cost for the decision tree was far better than the online cost for the KNN-classifier and slightly better than for the MNB-classifier. Therefore, if speed is the only concern, then the decision-tree classifier should be used over the other two.
* Paring down the feature vector led to an improvement in online cost in both classifiers by an order of magnitude. Therefore, if speed is a concern, then it is beneficial to pare down the vector for both classifiers.

* Paring down the feature vector led to a slight decrease in accuracy in the KNN-classifier and the MNB-classifier and an increase in accuracy in the decision-tree classifier. Thus, if accuracy is a concern, it is preferable to not pare down the feature vector for KNN.
* The MNB-classifier was notably more accurate than he KNN-classifier and significantly more accurate than the decision-tree classifer. The MNB was the most accurate as its relatively simple computations allow it to avoid overfitting at smaller sample sizes. The accuracy of KNN is expected as the online costs were far larger for the KNN-classifier. If accuracy is a concern, then the MNB-classifier should be chosen over the other two classifiers.
* The decision tree is a far less stable classifier than the other two classifiers in terms of accuracy. Perhaps this is due to underfitting for certain iterations of cross validation. Either way, the accuracy of the decision tree is clearly inferior to the other classifiers even when scaled up to a much larger dataset. If accuracy is the primary metric for the quality of the classifier, a decision tree should not be chosen - or a better sample size should be used to prevent underfitting and overfitting. The MNB-classifier is found to be the most accurate and stable at this sample size.
