from collections import defaultdict
import os
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import sys
from functools import reduce

# total number of documents
n = 0

termList = set()

# a dictionary to store document filenames
fileNames = {}

# a dictionary to store the postings lists  
postings = defaultdict(dict)

# a dictionary to store document frequency
doc_f = defaultdict(int)

# directory name containing the corpus
dir_name = 'bbc'

# dictionary to store document vector length
doc_vector_length = defaultdict(float)

def main():
    global n
    read_text_files()
    populate_dictionaries()
    initialize_doc_f()
    compute_lengths()
    user_search()

def read_text_files():
    global n, fileNames
    directories = os.listdir(dir_name)
    for dir in directories:
        if not dir.endswith(".TXT"):
            subfolder = os.path.join(dir_name,dir)
            new_dir = os.listdir(subfolder)
            for d in new_dir:
                doc_path = os.path.join(subfolder, d)
                n = n + 1
                fileNames[n] = doc_path
                

def populate_dictionaries():
    global fileNames, termList, postings
    for doc_id in fileNames:
        try:
            with open(fileNames[doc_id], "r") as doc:
                whole_doc = doc.read()
        except:
            print("Error reading file: {}".format(fileNames[doc_id]))
            return None
        preprocessed_tokens = nltk.word_tokenize(whole_doc)
        stop_words = set(stopwords.words('english') + list(punctuation))
        preprocessed_tokens = [word.lower() for word in preprocessed_tokens if not word in stop_words]
        stemmer = PorterStemmer()
        preprocessed_tokens = [stemmer.stem(word) for word in preprocessed_tokens]
        unique_tokens = set(preprocessed_tokens)
        termList = termList.union(unique_tokens)
        for token in unique_tokens:
            postings[token][doc_id] = preprocessed_tokens.count(token)

def initialize_doc_f():
    global doc_f
    for term in termList:
        doc_f[term] = len(postings[term])

def compute_lengths():
    global doc_vector_length, termList, fileNames
    for doc_id in fileNames:
        length = 0
        for term in termList:
            length += score(term, doc_id)**2
        doc_vector_length[doc_id] = math.sqrt(length)

def score(term, doc_id):
    if doc_id in postings[term]:
        return (1 + math.log(postings[term][doc_id],2))*idf(term)
    else:
        return 0.0

def idf(term):
    global n, termList
    if term in termList:
        return math.log(n/doc_f[term],2)
    else:
        return 0.0

def user_search():
    query = input("Query: ")
    if query is not None:
        preprocessed_query = nltk.word_tokenize(query)
        stop_words = set(stopwords.words('english') + list(punctuation))
        preprocessed_query = [word.lower() for word in preprocessed_query if not word in stop_words]
        stemmer = PorterStemmer()
        preprocessed_query = [stemmer.stem(word) for word in preprocessed_query]

        relevant_docs = intersect([set(postings[term].keys()) for term in preprocessed_query])

        if relevant_docs is None:
            print("No documents matched search.")
        else:
            scores = sorted([(doc_id, similarity(preprocessed_query, doc_id))
                             for doc_id in relevant_docs],
                            key = lambda x: x[1],
                            reverse = True)
            print("Score: filename")
            for (doc_id, score) in scores:
                print("{}: {}".format(score, fileNames[doc_id]))
                      
def intersect(sets):
    return reduce(set.intersection, [s for s in sets])

def similarity(query, doc_id):
    similarity = 0.0
    for term in query:
        if term in termList:
            similarity += idf(term)*score(term, doc_id)
    similarity = similarity/ doc_vector_length[doc_id]
    return similarity
    
if __name__ == "__main__":
    main()
