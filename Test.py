import search_engine
import pickle
from search_engine import SearchEngine, InvertedIndex, DocumentProcessing, \
    Document, NaiveBayesClassifier, ClassifierDataFrame, KNN
import time

start_time = time.time()
classifications, docs = search_engine.run("--vsm", "Harry Potter India")
print("It takes {} seconds for one query".format(time.time() - start_time))
all_ids = classifications["all"]
for id in all_ids:
    print(docs[id]+ "\n\n")