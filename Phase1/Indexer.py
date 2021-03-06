import os
import pickle
from math import log

import click

from Phase1.DataModels import Document
from Phase1.Bigram import Bigram
from Phase1.Compressor import Compressor
from Phase1.Preferences import Preferences
from Phase1 import Constants


class Indexer:
    index = {}
    TOTAL_DOCS = 0

    @classmethod
    def load_index(cls):
        compressor = Compressor()
        mode = Preferences.pref[Constants.pref_compression_type_key]
        cls.index = compressor.load_from_file(mode)
        cls.TOTAL_DOCS = len(os.listdir(Constants.docs_dir))

    @classmethod
    def save_index(cls):
        compressor = Compressor()
        mode = Preferences.pref[Constants.pref_compression_type_key]
        return compressor.save_to_file(cls.index, mode)

    @classmethod
    def add_document_to_index(cls, term_list: list, doc_id):
        for i, term in enumerate(term_list):
            if cls.index.get(term) is None:
                cls.index[term] = {doc_id: [i]}
                Bigram.add_term_to_bigram(term)
            elif cls.index[term].get(doc_id) is None:
                cls.index[term][doc_id] = [i]
            else:
                cls.index[term][doc_id].append(i)

    @classmethod
    def add_files(cls, docs_dir=Constants.docs_dir):
        cls.index.clear()
        cls.TOTAL_DOCS = len(os.listdir(docs_dir))
        with click.progressbar(label='Creating indexes from docs', length=cls.TOTAL_DOCS, fill_char="█") as bar:
            for filename in os.listdir(docs_dir):
                if filename.endswith('.o'):
                    file_path = os.path.join(docs_dir, filename)
                    with open(file_path, 'rb') as f:
                        doc: Document = pickle.load(f)
                        cls.add_document_to_index(doc.tokens, doc.docid)
                bar.update(1)

    @classmethod
    def delete_document_from_index(cls, term_list: list, doc_id):
        for i, term in enumerate(term_list):
            if cls.index.get(term) is not None:
                if cls.index[term].get(doc_id) is not None:
                    del cls.index[term][doc_id]
                    if len(cls.index[term]) == 0:
                        del cls.index[term]
                        Bigram.delete_term_from_bigram(term)

    @classmethod
    def delete_doc_file_from_index(cls, path):
        with open(path, 'rb') as f:
            doc: Document = pickle.load(f)
            cls.delete_document_from_index(doc.tokens, doc.docid)

    @classmethod
    def get_docs_containing_term(cls, term):
        if cls.index.get(term) is None:
            return []
        res = []
        for doc_id in cls.index[term]:
            res.append(doc_id)
        return res

    @classmethod
    def get_df(cls, term):
        if cls.index.get(term) is None:
            return 0
        return len(cls.index[term])

    @classmethod
    def get_idf(cls, term):
        if cls.index.get(term) is None:
            return 1
        return log(cls.TOTAL_DOCS / cls.get_df(term))

    @classmethod
    def get_tf(cls, term, doc_id):
        if cls.index.get(term) is None:
            return 0
        if cls.index[term].get(doc_id) is None:
            return 0
        return len(cls.index[term][doc_id])
