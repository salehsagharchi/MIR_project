class Indexer:
    index = {}

    @classmethod
    def load_index(cls):
        pass

    @classmethod
    def add_document_to_index(cls, term_list: list, docid):
        for i, term in enumerate(term_list):
            if cls.index.get(term) is None:
                cls.index[term] = {docid: [i]}
            elif cls.index[term].get(docid) is None:
                cls.index[term][docid] = [i]
            else:
                cls.index[term][docid].append(i)

    @classmethod
    def delete_document_from_index(cls, term_list: list, docid):
        for i, term in enumerate(term_list):
            if cls.index.get(term) is not None:
                if cls.index[term].get(docid) is not None:
                    del cls.index[term][docid]
                    if len(cls.index[term]) == 0:
                        del cls.index[term]

    @classmethod
    def add_doc_file_to_index(cls, path, docid):
        f = open(path, 'r')
        terms = f.read()
        f.close()
        cls.add_document_to_index(terms.split(' '), docid)


    @classmethod
    def delete_doc_file_from_index(cls, path, docid):
        f = open(path, 'r')
        terms = f.read()
        f.close()
        cls.delete_document_from_index(terms.split(' '), docid)

    @classmethod
    def get_df(cls, term):
        if cls.index.get(term) is None:
            return 0
        return len(cls.index[term])

    @classmethod
    def get_tf(cls, term, docid):
        if cls.index.get(term) is None:
            return 0
        if cls.index[term].get(docid) is None:
            return 0
        return len(cls.index[term][docid])


if __name__ == '__main__':
    ind = {'t1': {1: [1, 5, 6], 2: [10, 15]},
           't2': {1: [3, 8], 2: [4, 9]},
           't3': {2: [20]},
           't4': {3: [7, 17]}}