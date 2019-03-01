class InvertedIndex:
    def __init__(self):
        self.documents = None
        self.terms = None
        self.posting_lists = None

    def parse_document(self, document_contents):
        pass

    def read_text_file(self, filename):
        pass

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
    pass