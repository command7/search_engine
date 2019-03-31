import numpy as np
import nltk
# nltk.download() #!!!Run this the first time you run your script!!!
import os
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import operator
import pandas as pd
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import sys

""" Stores Document ID and positions of a term """


class Document():
    def __init__(self, document_id, *args, store_term_weights = False):
        self.id = document_id
        self.positions = []
        if store_term_weights == True:
            self.term_weight = 1
        else:
            self.positions = []
            self.add_position(args[0])

    # Adds a position of a term to the Doc object.
    def add_position(self, position):
        self.positions.append(position)

    def increment_frequency(self):
        self.term_weight += 1


""" Parent class for InvertedIndex and SearchEngine. 
Deals with preprocessing queries and document text as well as retreiving posting lists."""


class DocumentProcessing():
    def __init__(self):
        self.documents = 0

    # Remove stop words and perform stemming
    def pre_process(self, document_content, remove_stopwords=False, stemming=False):
        preprocessed_tokens = nltk.word_tokenize(document_content)
        preprocessed_tokens = [word.lower() for word in preprocessed_tokens]
        if remove_stopwords:
            stop_words = set(stopwords.words('english') + list(punctuation))
            preprocessed_tokens = [word for word in preprocessed_tokens if not word in stop_words]
        if stemming:
            stemmer = PorterStemmer()
            preprocessed_tokens = [stemmer.stem(word) for word in preprocessed_tokens]
        return preprocessed_tokens

    # Returns the postings list of a term
    def get_postings_list(self, term):
        term_index = self.terms.index(term)
        return self.posting_lists[term_index]


""" Storage class for parsing documents and constructing inverted index. """
class InvertedIndex(DocumentProcessing):
    num_documents = 0
    def __init__(self, document_loc=None, purpose="bs", is_dir=True, auto_load=True):
        self.documents = list()
        self.terms = list()
        self.posting_lists = list()
        self.purpose = purpose
        self.auto_load = auto_load
        self.classifier_df = ClassifierDataFrame()
        self.docLengths = dict()
        if self.auto_load:
            if self.purpose == "vsm":
                if is_dir:
                    self.load_data(document_loc, ignore_stopwords=False)
                else:
                    self.load_data(document_loc, ignore_stopwords=False, is_text = True)
                self.calculate_tfidf()
            else:
                if is_dir:
                    self.load_data(document_loc)
                else:
                    self.load_data(is_text=True)
            self.classifier_df.split_training_testing_set(t_size=0.2)

    def load_data(self, directory, ignore_stopwords = True, is_text = False):
        if is_text:
            self.parse_document(directory, ignore_stopwords, is_text=True)
        else:
            current_directory = os.getcwd()
            doc_directory = os.path.join(current_directory, directory)
            for class_ in os.listdir(doc_directory):
                class_docs_loc = os.path.join(doc_directory, class_)
                if (os.path.isdir(class_docs_loc)):
                    for class_document in os.listdir(class_docs_loc):
                        if not class_document.startswith("."):
                            doc_location = os.path.join(class_docs_loc, class_document)
                            self.classifier_df.add_document(doc_location, class_)
                            self.parse_document(doc_location, ignore_stopwords)

    # Assign a new document id for new documents
    def assign_document_id(self):
        InvertedIndex.num_documents += 1
        return InvertedIndex.num_documents - 1

    # Parses a new document, preprocesses it and updates the inverted index
    def parse_document(self, file_name, ignore_stopwords, is_text = False):
        document_id = self.assign_document_id()
        if is_text:
            document_text = file_name
        else:
            document_text = self.read_text_file(file_name)
        self.add_document(document_text)
        if ignore_stopwords == True:
            processed_tokens = self.pre_process(document_text, remove_stopwords=True, stemming=True)
        else:
            processed_tokens = self.pre_process(document_text, stemming=True)
        self.update_inv_index(processed_tokens, document_id)

    # Adds the document to storage
    def add_document(self, document_content):
        self.documents.append(document_content)

    # Reads a document and returns its content as a string
    def read_text_file(self, filename):
        current_dir = os.getcwd()
        doc_path = os.path.join(current_dir, filename)
        try:
            with open(doc_path, "r") as doc:
                whole_doc = doc.read()
        except:
            print("Error reading file: {}".format(filename))
            return None
        return whole_doc

    # Add a new term and posting list to inverted index if a new term is found. Update inverted index otherwise.
    def update_inv_index(self, processed_tokens, document_id):
        for token_index in range(0, len(processed_tokens)):
            if (processed_tokens[token_index] not in self.terms):
                self.terms.append(processed_tokens[token_index])
                new_postings_list = list()
                if self.purpose == "vsm":
                    new_doc = Document(document_id, store_term_weights = True)
                else:
                    new_doc = Document(document_id, token_index + 1)
                new_postings_list.append(new_doc)
                self.posting_lists.append(new_postings_list)
            else:
                existing_posting_list = self.get_postings_list(processed_tokens[token_index])
                doc_exists = False
                for i in range(len(existing_posting_list)):
                    if existing_posting_list[i].id == document_id:
                        if self.purpose == "vsm":
                            existing_posting_list[i].increment_frequency()
                        else:
                            existing_posting_list[i].add_position(token_index + 1)
                        doc_exists = True
                if doc_exists == False:
                    if self.purpose == "vsm":
                        new_doc = Document(document_id, store_term_weights = True)
                    else:
                        new_doc = Document(document_id, token_index + 1)
                    existing_posting_list.append(new_doc)

    def calculate_tfidf(self):
        total_num_docs = len(self.documents)
        for term in self.terms:
            term_posting_list = self.get_postings_list(term)
            num_docs_term = len(term_posting_list)
            invert_doc_frequency = np.log10(total_num_docs/(num_docs_term*1.0))
            for indiv_doc in term_posting_list:
                tfidf = (1 + np.log10(indiv_doc.term_weight)) * invert_doc_frequency
                indiv_doc.term_weight = tfidf
                if indiv_doc.id in self.docLengths.keys():
                    self.docLengths[indiv_doc.id] += np.square(tfidf)
                else:
                    self.docLengths[indiv_doc.id] = np.square(tfidf)
        for i in range(len(self.docLengths)):
            self.docLengths[i] = np.sqrt(self.docLengths[i])

    # Print out inverted index
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


"""Deals with search queries."""

class SearchEngine(DocumentProcessing):
    def __init__(self, inverted_index):
        self.purpose = inverted_index.purpose
        self.terms = inverted_index.terms
        self.documents = inverted_index.documents
        self.posting_lists = inverted_index.posting_lists
        if self.purpose == "vsm":
            self.docLengths = inverted_index.docLengths

    # Provide a string query and returns a posting list with Documents containing all the terms
    def boolean_and_query(self, query):
        tokenized_query = self.pre_process(query, remove_stopwords=True, stemming=True)
        processed_query = self.sort_on_tf(tokenized_query)
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
            elif len(processed_query) > 2:
                posting_list_one = self.get_postings_list(processed_query.pop(0))
                posting_list_two = self.get_postings_list(processed_query.pop(0))
                query_results = self.merge_intersect(posting_list_one, posting_list_two)
                while len(query_results) > 0 and len(processed_query) > 0:
                    posting_list_ = self.get_postings_list(processed_query.pop(0))
                    query_results = self.merge_intersect(query_results, posting_list_)
        else:
            return None
        return query_results

    # Returns tokens sorted in terms of document frequency
    def sort_on_tf(self, query_tokens):
        tf_info = {}
        sorted_terms = []
        for token in query_tokens:
            tf_info[token] = len(self.get_postings_list(token))
        sorted_words = sorted(tf_info.items(), key=operator.itemgetter(1))
        for term_info in sorted_words:
            sorted_terms.append(term_info[0])
        return sorted_terms

    # Returns documents containing query terms in order
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

    # Compare postings lists using merge algorithm
    def merge_intersect(self, post_list_one, post_list_two):
        intersect_documents = []
        pointer_one = 0
        pointer_two = 0
        while pointer_one < len(post_list_one) and pointer_two < len(post_list_two):
            if (post_list_one[pointer_one].id == post_list_two[pointer_two].id):
                intersect_documents.append(post_list_two[pointer_two])
                pointer_one += 1
                pointer_two += 1
            elif (post_list_one[pointer_one].id > post_list_two[pointer_two].id):
                pointer_two += 1
            else:
                pointer_one += 1
        return intersect_documents

    def ranked_search(self, query):
        vsm_scores = dict()
        if self.purpose == "bs":
            print("Cannot proceed as Inverted Index supplied does not contain term weights.")
            raise Exception
        else:
            query_tokens = self.pre_process(query, remove_stopwords=False, stemming=True)
            for q_token in query_tokens:
                if q_token not in self.terms:
                    continue
                else:
                    q_posting_list = self.get_postings_list(q_token)
                    query_token_tfidf = (1 + np.log10(1)) * (len(self.documents)* 1.0 / len(q_posting_list))
                    for document_ in q_posting_list:
                        score = document_.term_weight * query_token_tfidf
                        if score in vsm_scores.keys():
                            vsm_scores[document_.id] += score
                        else:
                            vsm_scores[document_.id] = score
            ranked_results = sorted(vsm_scores.items(), key=operator.itemgetter(1))
            print(vsm_scores)
            if len(ranked_results) > 10:
                result_docs = [ranked_results[rank][0] for rank in range(0, 10)]
            else:
                result_docs = [ranked_results[rank][0] for rank in range(0, len(ranked_results))]
            return result_docs

    # Prints out search results
    def print_search_results(self, result_docs):
        if len(result_docs) == 0:
            print("No documents found.")
        else:
            for result_doc in result_docs:
                print(self.documents[result_doc.id - 1])

    # Checks if a term exists in dictionary
    def check_existence(self, term):
        if (term in self.terms):
            return True
        return False

    # Compare postings lists to check if terms appear in order
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


class ClassifierDataFrame():
    def __init__(self):
        self.columns = ["document_contents", "class"]
        self.df = pd.DataFrame(columns=self.columns)
        self.features = None
        self.target = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def add_document(self, file_name, class_value):
        try:
            with open(file_name) as document:
                data_instance = pd.DataFrame([[document.read(), class_value]], columns=self.columns)
                self.df = pd.concat([self.df, data_instance]).reset_index(drop=True)
        except:
            print("Error reading file {} in class {}".format(file_name, class_value))

    def split_target_features(self):
        self.features = self.df["document_contents"].copy()
        self.target = self.df["class"].copy()

    def split_training_testing_set(self, t_size):
        self.split_target_features()
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=t_size, random_state=7)
        for train_index, test_index in stratified_split.split(self.features, self.target):
            self.X_train = pd.DataFrame(np.reshape(self.features.loc[train_index].values, (-1,1)), \
                                        columns=["document_contents"]).reset_index(drop=True)
            self.y_train = pd.DataFrame(np.reshape(self.target.loc[train_index].values, (-1,1)), \
                                        columns=["class"]).reset_index(drop=True)
            self.X_test = pd.DataFrame(np.reshape(self.features.loc[test_index].values, (-1,1)), \
                                       columns=["document_contents"]).reset_index(drop=True)
            self.y_test = pd.DataFrame(np.reshape(self.target.loc[test_index].values, (-1,1)), \
                                       columns=["class"]).reset_index(drop=True)


class NaiveBayesClassifier(DocumentProcessing):
    def __init__(self, Classifier_df):
        self.raw_data = None
        self.raw_training_documents = Classifier_df.X_train
        self.training_class_labels = Classifier_df.y_train
        self.priors = dict()
        self.conditional_probabilities = dict()
        self.total_vocab_count = 0
        self.class_vocab_count = dict()
        self.class_values = ["business", "sport", "politics", "entertainment", "tech"]
        self.metrics = dict()
        self.consolidate_training_set()
        self.parse_vocabulary()
        self.N = self.raw_data.shape[0]
        self.bernoulli_index = dict()

    def consolidate_training_set(self):
        consolidated_df = pd.concat([self.raw_training_documents, self.training_class_labels], axis=1)
        self.raw_data = consolidated_df

    def get_conditional_probability(self, word, class_value):
        class_df = self.conditional_probabilities[class_value]
        return float(class_df[class_df.terms == word].loc[:, "conditional_probability"])

    def get_bernoulli_condition_probability(self, word, class_value):
        class_df = self.conditional_probabilities[class_value]
        return float(class_df[class_df.terms == word].loc[:, "bernoulli_probability"])

    def fit(self):
        for class_value in self.class_values:
            self.build_bernoulli_index(class_value)
            self.calculate_probabilities(class_value)

    def build_bernoulli_index(self, class_value):
        class_docs = list(self.raw_data[self.raw_data["class"] == class_value].copy()["document_contents"])
        inverted_index = InvertedIndex(auto_load=False)
        for class_doc in class_docs:
            inverted_index.load_data(class_doc, is_text=True)
        self.bernoulli_index[class_value] = inverted_index

    def parse_vocabulary(self):
        for class_value_ in self.class_values:
            class_docs = list(self.raw_data[self.raw_data["class"] == class_value_].copy()["document_contents"])
            for class_doc in class_docs:
                tokens = self.pre_process(class_doc, remove_stopwords=True, stemming=True)
                for token in tokens:
                    self.total_vocab_count += 1

    def calculate_probabilities(self, class_value):
        terms = list()
        voc_count = 0
        num_instances = list()
        bernoulli_inv_index = self.bernoulli_index[class_value]
        num_docs = list()
        class_docs = list(self.raw_data[self.raw_data["class"] == class_value].copy()["document_contents"])
        N_c = len(class_docs)
        prior = np.log(N_c / self.N)
        self.priors[class_value] = prior
        for class_doc in class_docs:
            tokens = self.pre_process(class_doc, remove_stopwords=True, stemming=True)
            for token in tokens:
                voc_count += 1
                if token not in terms:
                    terms.append(token)
                    num_instances.append(0)
                    num_docs.append(0)
        for class_doc_ in class_docs:
            tokens_ = self.pre_process(class_doc_, remove_stopwords=True, stemming=True)
            for token_ in tokens_:
                term_index = terms.index(token_)
                num_docs[term_index] += len(bernoulli_inv_index.get_postings_list(token_))
                num_instances[term_index] += 1
        self.class_vocab_count[class_value] = voc_count
        conditional_probs = np.array([terms, num_instances, num_docs])
        conditional_probs = conditional_probs.T
        conditional_df = pd.DataFrame(conditional_probs, columns=["terms", "number_of_instances", "number_of_docs"])
        conditional_df["number_of_instances"] = conditional_df.number_of_instances.astype(int)
        conditional_df["number_of_docs"] = conditional_df.number_of_docs.astype(int)
        conditional_df["conditional_probability"] = (conditional_df["number_of_instances"] + 1)/(voc_count + self.total_vocab_count * 1.0)
        conditional_df["bernoulli_probability"] = (conditional_df["number_of_docs"] + 1 * 1.0)/(N_c + 2)
        self.conditional_probabilities[class_value] = conditional_df

    def calculate_metrics(self, predictions, testing_labels):
        precision = precision_score(testing_labels, predictions, average="weighted")
        recall = recall_score(testing_labels, predictions, average="weighted")
        f_score = f1_score(testing_labels, predictions, average="weighted")
        confusion_mat = confusion_matrix(testing_labels, predictions)
        accuracy = accuracy_score(testing_labels, predictions)
        return precision, recall, f_score, accuracy

    def predict_single(self, pred_doc, mode):
        tokens = self.pre_process(pred_doc, remove_stopwords=True, stemming=True)
        argmax = dict()
        if mode == "b": # bernoulli
            for class_value in self.class_values:
                class_df = self.conditional_probabilities[class_value]
                output = self.priors[class_value]
                for word in np.array(class_df.terms):
                    instance = self.get_bernoulli_condition_probability(word, class_value)
                    if word in tokens:
                        output += np.log(instance)
                    else:
                        output += 1 - np.log(instance)
                argmax[class_value] = output
            return max(argmax, key=argmax.get)
        elif mode == "m": # multinomial
            for class_value in self.class_values:
                class_df = self.conditional_probabilities[class_value]
                output = self.priors[class_value]
                for word in tokens:
                    if word in class_df.terms.unique():
                        instance = self.get_conditional_probability(word, class_value)
                        output += np.log(instance)
                    else:
                        output += np.log(1/(self.class_vocab_count[class_value] + self.total_vocab_count))
            return max(argmax, key=argmax.get)

    def predict_multiple(self, testing_df, mode):
        predictions = []
        if mode == "m": #multinomial
            for document_content in testing_df["document_contents"].values:
                tokens = self.pre_process(str(document_content))
                maxima = dict()
                for class_value in self.class_values:
                    class_df = self.conditional_probabilities[class_value]
                    output = self.priors[class_value]
                    for word in tokens:
                        if word in class_df.terms.unique():
                            instance = self.get_conditional_probability(word, class_value)
                            output += np.log(instance)
                        else:
                            output += np.log(1 / (self.class_vocab_count[class_value] + 1))
                    maxima[class_value] = output
                predictions.append(max(maxima, key=maxima.get))
            predictions_df = pd.DataFrame(predictions, columns=["class_predictions"])
            return predictions_df
        elif mode == "b": # bernoulli
            for document_content in testing_df["document_contents"].values:
                tokens = self.pre_process(str(document_content))
                maxima = dict()
                for class_value in self.class_values:
                    class_df = self.conditional_probabilities[class_value]
                    output = self.priors[class_value]
                    for word in np.array(class_df.terms):
                        instance = self.get_bernoulli_condition_probability(word, class_value)
                        if word in tokens:
                            output += np.log(instance)
                        else:
                            output += 1 - np.log(instance)
                    maxima[class_value] = output
                    print(maxima)
                predictions.append(max(maxima, key=maxima.get))
            predictions_df = pd.DataFrame(predictions, columns=["class_predictions"])
            return predictions_df


if __name__ == "__main__":
    inv_index = InvertedIndex("test")
    cl_df = inv_index.classifier_df
    nb = NaiveBayesClassifier(cl_df)
    nb.fit()
    b_preds = nb.predict_multiple(cl_df.X_test, mode="b")
    m_preds = nb.predict_multiple(cl_df.X_test, mode="m")
    test_labels = cl_df.y_test

    for i in range(len(b_preds)):
        print("{} --- {}".format(b_preds.iloc[i], test_labels.iloc[i]))

    # nbm_pred_ = nbm_pred.class_predictions.map({"politics":0, "entertainment":1,"sport":2,"business":3, "tech" :4}).values
    # nbb_pred_ = nbb_pred.class_predictions.map(
    #     {"politics": 0, "entertainment": 1, "sport": 2, "business": 3, "tech": 4}).values
    # test_labels_ = test_labels["class"].map(
    #     {"politics": 0, "entertainment": 1, "sport": 2, "business": 3, "tech": 4}).values
    # m_precision, m_recall, m_f_score, m_accuracy = nb.calculate_metrics(nbm_pred_, test_labels_)
    # b_precision, b_recall, b_f_score, b_accuracy = nb.calculate_metrics(nbb_pred_, test_labels_)
    # print("Multinomial Model")
    # print("Accuracy: {}".format(m_accuracy))
    # print("Bernoulli Model")
    # print("Accuracy: {}".format(b_accuracy))



    # df = pickle.load(open("raw_data_df.p", "rb"))
    # nb = pickle.load(open("Naive_Bayes.p", "rb"))
    # predictions = pd.read_csv("test_predictions.csv")
    # encoded_test = df.y_test["class"].map({"politics":0, "entertainment":1,"sport":2,"business":3, "tech" :4}).values
    # encoded_pred = predictions["class_predictions"].map({"politics":0, "entertainment":1,"sport":2,"business":3, "tech" :4}).values
    # precision, recall, fscore, conf_mat = nb.calculate_metrics(encoded_pred, encoded_test)
    # print("Precision : {}".format(precision))
    # print("Recall : {}".format(recall))
    # print("F1 Score : {}".format(fscore))

    # if sys.argv[1] == "--nb":
    #     mode = sys.argv[2]
    #     document_name = sys.argv[3]
    #     print("Using Naive Bayes Classifier to predict given document: {} ".format(document_name))
    #     nb_model = pickle.load(open("Naive_Bayes.p", "rb"))
    #     prediction = nb_model.predict_single(document_name, mode)
    #
    # elif sys.argv[1] == "--bs":
    #     print("Boolean Search")
    # elif sys.argv[1] == "--vsm":
    #     print("Vector Space Model")