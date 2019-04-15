import search_engine
import pickle
from search_engine import SearchEngine, InvertedIndex, DocumentProcessing, \
    Document, NaiveBayesClassifier, ClassifierDataFrame, KNN
import time

start_time = time.time()
classifications, docs = search_engine.run("--vsm", "Harry Potter India")
print("It takes {} seconds for one query".format(time.time() - start_time))
print("Total Docs : {}".format(len(classifications["all"])))
print("Politics Docs : {}".format(len(classifications["politics"])))
print("Business Docs : {}".format(len(classifications["business"])))
print("Tech Docs : {}".format(len(classifications["tech"])))
print("Sport Docs : {}".format(len(classifications["sport"])))
print("Entertainment Docs : {}".format(len(classifications["entertainment"])))