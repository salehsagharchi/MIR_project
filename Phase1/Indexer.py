import os
import pickle

from Phase1.DataModels import Document
from Phase1 import Bigram
from Phase1 import Compressor


class Indexer:
    index = {}
    TOTAL_DOCS = 0

    @classmethod
    def load_index(cls, mode):
        compressor = Compressor.Compressor()
        index = compressor.load_from_file(mode)

    @classmethod
    def save_index(cls, mode):
        compressor = Compressor.Compressor()
        compressor.save_to_file(cls.index, mode)

    @classmethod
    def add_document_to_index(cls, term_list: list, doc_id):
        for i, term in enumerate(term_list):
            if cls.index.get(term) is None:
                cls.index[term] = {doc_id: [i]}
                Bigram.Bigram.add_term_to_bigram(term)
            elif cls.index[term].get(doc_id) is None:
                cls.index[term][doc_id] = [i]
            else:
                cls.index[term][doc_id].append(i)

    @classmethod
    def add_files(cls, docs_dir="data/wiki_docs/"):
        for filename in os.listdir(docs_dir):
            if filename.endswith('.o'):
                file_path = os.path.join(docs_dir, filename)
                with open(file_path, 'rb') as f:
                    doc: Document = pickle.load(f)
                    cls.add_document_to_index(doc.tokens, doc.docid)
                    cls.TOTAL_DOCS += 1

    @classmethod
    def delete_document_from_index(cls, term_list: list, doc_id):
        for i, term in enumerate(term_list):
            if cls.index.get(term) is not None:
                if cls.index[term].get(doc_id) is not None:
                    del cls.index[term][doc_id]
                    if len(cls.index[term]) == 0:
                        del cls.index[term]
                        Bigram.Bigram.delete_term_from_bigram(term)

    @classmethod
    def delete_doc_file_from_index(cls, path):
        with open(path, 'rb') as f:
            doc: Document = pickle.load(f)
            cls.delete_document_from_index(doc.tokens, doc.docid)

    @classmethod
    def get_df(cls, term):
        if cls.index.get(term) is None:
            return 0
        return len(cls.index[term])

    @classmethod
    def get_docs_containing_term(cls, term):
        if cls.index.get(term) is None:
            return []
        res = []
        for doc_id in cls.index[term]:
            res.append(doc_id)
        return res

    @classmethod
    def get_tf(cls, term, doc_id):
        if cls.index.get(term) is None:
            return 0
        if cls.index[term].get(doc_id) is None:
            return 0
        return len(cls.index[term][doc_id])


# if __name__ == '__main__':
#     pass
