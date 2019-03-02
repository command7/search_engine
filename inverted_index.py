import numpy as np
import nltk
#nltk.download() #!!!Run this the first time you run your script!!!
import os
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


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

    def get_postings_list(self, term):
        term_index = self.terms.index(term)
        return self.posting_lists[term_index]


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

    def boolean_and_query(self, query):
        processed_query = self.pre_process(query)
        query_results = None
        all_terms_exist = True
        for token in processed_query:
            if (self.check_existence(token) == False):
                all_terms_exist = False
        if all_terms_exist:
            if len(processed_query) == 2:
                posting_list_one = self.get_postings_list(processed_query[0])
                posting_list_two = self.get_postings_list(processed_query[1])
                query_results = self.merge_intersect(posting_list_one, posting_list_two)
            elif (len(processed_query) > 2):
                posting_list_one = self.get_postings_list(processed_query.pop(0))
                posting_list_two = self.get_postings_list(processed_query.pop(0))
                query_results = self.merge_intersect(posting_list_one, posting_list_two)
                while len(query_results) > 0 and len(processed_query) > 0:
                    posting_list_ = self.get_postings_list(processed_query.pop(0))
                    query_results = self.merge_intersect(query_results, posting_list_)
        else:
            return None
        return query_results

    def positional_search(self, query):
        processed_query = self.pre_process(query)
        query_results = None
        all_terms_exist = True
        for token in processed_query:
            if (self.check_existence(token) == False):
                all_terms_exist = False
        if all_terms_exist:
            if len(processed_query) == 2:
                posting_list_one = self.get_postings_list(processed_query[0])
                posting_list_two = self.get_postings_list(processed_query[1])
                query_results = self.positional_intersect(posting_list_one, posting_list_two)
            elif (len(processed_query) > 2):
                posting_list_one = self.get_postings_list(processed_query.pop(0))
                posting_list_two = self.get_postings_list(processed_query.pop(0))
                query_results = self.positional_intersect(posting_list_one, posting_list_two)
                while len(query_results) > 0 and len(processed_query) > 0:
                    posting_list_ = self.get_postings_list(processed_query.pop(0))
                    query_results = self.positional_intersect(query_results, posting_list_)
        else:
            return None
        return query_results

    def merge_intersect(self, post_list_one, post_list_two):
        intersect_documents = []
        pointer_one = 0
        pointer_two = 0
        while pointer_one < len(post_list_one) and pointer_two < len(post_list_two):
            if (post_list_one[pointer_one].id == post_list_two[pointer_two].id):
                intersect_documents.append(post_list_two[pointer_two])
                pointer_one += 1
                pointer_two +=1
            elif (post_list_one[pointer_one].id > post_list_two[pointer_two].id):
                pointer_two += 1
            else:
                pointer_one += 1
        return intersect_documents

    def print_search_results(self, result_docs):
        if len(result_docs) == 0:
            print("No documents found.")
        else:
            for result_doc in result_docs:
                print(self.documents[result_doc.id-1])

    def check_existence(self, term):
        if (term in self.terms):
            return True
        return False

    def positional_intersect(self, post_list_one, post_list_two):
        intersect_documents = []
        pointer_one = 0
        pointer_two = 0
        while pointer_one < len(post_list_one) and pointer_two < len(post_list_two):
            if (post_list_one[pointer_one].id == post_list_two[pointer_two].id):
                position_list_1 = post_list_one[pointer_one].positions
                position_list_2 = post_list_two[pointer_two].positions
                pos_point_1 = 0
                pos_point_2 = 0
                match_found = False
                while pos_point_1 < len(position_list_1) and match_found == False:
                    while pos_point_2 < len(position_list_2) and match_found == False:
                        if position_list_1[pos_point_1] - position_list_2[pos_point_2] == -1:
                            intersect_documents.append(post_list_two[pointer_two])
                            match_found = True
                            break
                        pos_point_2 += 1
                    pos_point_2 = 0
                    pos_point_1 += 1
                pointer_one += 1
                pointer_two += 1
            elif (post_list_one[pointer_one].id > post_list_two[pointer_two].id):
                pointer_two += 1
            else:
                pointer_one += 1
        return intersect_documents

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
    # merged_documents = engine.boolean_and_query("nlp text mining")
    position_docs = engine.positional_search("data big")
    engine.print_search_results(position_docs)


