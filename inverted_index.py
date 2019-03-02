import numpy as np
import nltk
#nltk.download() #!!!Run this the first time you run your script!!!
import os
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


test_filename = "test.txt"

class Document():
    def __init__(self, document_id, position):
        self.id = document_id
        self.positions = []
        self.add_position(position)


    def add_position(self, position):
        self.positions.append(position)


class DocumentProcessing():
    def __init__(self):
        self.documents = 0

    def pre_process(self, document_content):
        stop_words = set(stopwords.words('english') + list(punctuation))
        tokens = nltk.word_tokenize(document_content)
        filtered_words = [word.lower() for word in tokens if not word in stop_words]
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        return stemmed_words


class InvertedIndex(DocumentProcessing):
    num_documents = 0
    def __init__(self):
        self.documents = list()
        self.terms = list()
        self.posting_lists = list()

    def assign_document_id(self):
        InvertedIndex.num_documents += 1
        return InvertedIndex.num_documents

    def parse_document(self, file_name):
        document_id = self.assign_document_id()
        document_text = self.read_text_file(file_name)
        self.add_document(document_text)
        processed_tokens = self.pre_process(document_text)
        self.update_inv_index(processed_tokens, document_id)

    def add_document(self, document_content):
        self.documents.append(document_content)

    def read_text_file(self, filename):
        current_dir = os.getcwd()
        doc_path = os.path.join(current_dir, filename)
        with open(doc_path, "r") as doc:
            whole_doc = doc.read()
        return whole_doc

    def update_inv_index(self, processed_tokens, document_id):
        for token_index in range(0, len(processed_tokens)):
            if (processed_tokens[token_index] not in self.terms):
                self.terms.append(processed_tokens[token_index])
                new_postings_list = list()
                new_doc = Document(document_id, token_index+1)
                new_postings_list.append(new_doc)
                self.posting_lists.append(new_postings_list)
            else:
                term_index = self.terms.index(processed_tokens[token_index])
                existing_posting_list = self.posting_lists[term_index]
                new_doc = Document(document_id, token_index+1)
                existing_posting_list.append(new_doc)

    def __repr__(self):
        output = ""
        for i in range(0, len(self.terms)):
            output += "{:>12}\t".format(self.terms[i])
            doclist = self.posting_lists[i]
            for j in range(0, len(doclist)):
                output += "[{}]".format(doclist[j].id)
                output += " <"
                output += "{}, ".format(str(doclist[j].positions))
                output += ">\t"
            output += "\n"
        return output


class SearchEngine(DocumentProcessing):
    def __init__(self, inverted_index):
        self.terms = inverted_index.terms
        self.documents = inverted_index.documents
        self.posting_lists = inverted_index.posting_lists

    def boolean_and_query(self):
        pass

    def merge_intersect(self, term_one, term_two):

        intersect_documents = []
        term_one_index = self.terms.index(term_one)
        term_two_index = self.terms.index(term_two)
        post_list_one = self.posting_lists[term_one_index]
        post_list_two = self.posting_lists[term_two_index]
        pointer_one = 0
        pointer_two = 0
        while pointer_one < len(post_list_one):
            while pointer_two < len(post_list_two):
                if (post_list_one[pointer_one].id == post_list_two[pointer_two].id):
                    intersect_documents.append(post_list_two[pointer_two])
                    pointer_one += 1
                    pointer_two +=1
                    pass
                elif (post_list_one[pointer_one].id > post_list_two[pointer_two].id):
                    pointer_two += 1
                else:
                    pointer_one += 1
        return intersect_documents

    def check_existence(self, term):
        if (term in self.terms):
            return True
        return False

    def positional_intersect(self):
        pass

if __name__ == "__main__":
    inv_index = InvertedIndex()
    inv_index.parse_document("test1.txt")
    inv_index.parse_document("test2.txt")
    inv_index.parse_document("test3.txt")
    inv_index.parse_document("test4.txt")
    for i in inv_index.documents:
        print(i)
    print(inv_index)
    engine = SearchEngine(inv_index)
    merged_documents = engine.merge_intersect("nlp", "text")
    # print(inv_index.terms)
    # for i in inv_index.posting_lists:
    #     for j in i:
    #         print(j.id)
    #         print(j.position)


