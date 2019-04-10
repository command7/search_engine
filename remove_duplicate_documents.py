import hashlib
import os

class_directories = ["entertainment", "politics", "sport", "business", "tech"]

def remove_duplicates(dir_path):
    unique_docs = []
    for document in os.listdir(dir_path):
        doc_path = os.path.join(dir_path, document)
        document_hash = hashlib.md5(open(doc_path,
                                         "rb").read()).digest()
        if document_hash not in unique_docs:
            unique_docs.append(document_hash)
        else:
            os.remove(doc_path)

for class_dir in class_directories:
    current_dir = os.getcwd()
    doc_path = os.path.join(current_dir, "documents", class_dir)
    remove_duplicates(doc_path)