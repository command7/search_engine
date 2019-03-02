import numpy as np
from nltk.tokenize import word_tokenize
import os

test_filename = "bbc/business/001.txt"
class InvertedIndex:
    def __init__(self):
        self.documents = list()
        self.terms = None
        self.posting_lists = None

    def parse_document(self, document_contents):
        pass

    def read_text_file(self, filename):
        current_dir = os.getcwd()
        doc_path = os.path.join(current_dir, filename)
        with open(doc_path, "r") as doc:
            whole_doc = doc.read()
        return whole_doc
        #     self.documents.append(whole_doc)
        # return

    def pre_process(self):

        pass

    def update_inv_index(self):
        pass



class SearchEngine:
    def __init__(self):
        self.InvertedIndex = None

    def boolean_query(self):
        pass

    def merge_intersect(self):
        pass

    def positional_intersect(self):
        pass

if __name__ == "__main__":
    inv_index = InvertedIndex()
    inv_index.read_text_file(test_filename)
    print(inv_index.documents)
