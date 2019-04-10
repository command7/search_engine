import search_engine
import pickle
from search_engine import SearchEngine, InvertedIndex, DocumentProcessing, Document, NaiveBayesClassifier, ClassifierDataFrame

test = pickle.load(open("pickled_objects/Class_ID_Matching.p", "rb"))
print(test)