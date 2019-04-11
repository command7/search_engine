import search_engine
import pickle
from search_engine import SearchEngine, InvertedIndex, DocumentProcessing, \
    Document, NaiveBayesClassifier, ClassifierDataFrame, KNN

# and_query, and_docs = search_engine.run("--bs","Harry Potter "
#                                                     "Aviator Vera Bafta Clive Owen Drake Helen")
#
# # Test positional query
# pos_query, pos_docs = search_engine.run("--ps", "Harry Potter and the "
#                                                 "prisoner of "
#                                      "Azkaban")
# # Test VSM query
# vsm_query, vsm_docs = search_engine.run("--vsm", "Harry Aviator Vera")
# Test Naive Bayes
# nb_query = search_engine.run("--nb", "testing_set/sport/164.txt")
# Test KNN
knn_query = search_engine.run("--knn", "testing_set/sport/164.txt")

print(knn_query)