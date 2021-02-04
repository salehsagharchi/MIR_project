import json
from math import log, sqrt

from Phase3.Parser import TextNormalizer
from Phase3.Constants import *


class TFIDF:
    total_docs = 0
    document_frequency = dict()
    document_vector_list = list()
    vocab = set()
    reference_label_list = list()
    link_list = list()

    @classmethod
    def start(cls):
        cls.document_frequency.clear()
        cls.document_vector_list = list()
        cls.vocab.clear()
        cls.link_list = list()
        documents = json.loads(open(FARSI_DOCUMENTS_PATH, 'r', encoding="utf-8").read())
        cls.total_docs = len(documents)
        cls.get_reference_labels(documents)

        document_dicts = []
        for doc in documents:
            document_dicts.append(cls.create_document_dict(cls.get_list_of_terms_from_document(doc)))
            cls.link_list.append(doc['link'])

        for doc_dict in document_dicts:
            for term in doc_dict:
                doc_dict[term] *= cls.get_idf(term)
            cls.normalize_vector(doc_dict)

        cls.create_document_vectors(document_dicts)

    @classmethod
    def get_list_of_terms_from_document(cls, document):
        list_of_terms = TextNormalizer.prepare_text(document['title'], "fa")
        list_of_terms += TextNormalizer.prepare_text(document['summary'], "fa")
        return list_of_terms

    @classmethod
    def get_reference_labels(cls, documents, primary_mode=True):
        cls.reference_label_list.clear()
        for doc in documents:
            if primary_mode:
                tag = doc['tags'][0].split('>')[0][:-1]
            else:
                tag = doc['tags'][0].split('>')[1][1:]
            cls.reference_label_list.append(tag)

    @classmethod
    def create_document_vectors(cls, document_dicts: list):
        for doc_dict in document_dicts:
            vector = list()
            for term in cls.vocab:
                if doc_dict.get(term) is None:
                    vector.append(0)
                else:
                    vector.append(doc_dict[term])
            cls.document_vector_list.append(vector)

    @classmethod
    def create_document_dict(cls, terms: list):
        for term in terms:
            cls.vocab.add(term)

        doc_dict = dict()
        for term in terms:
            if doc_dict.get(term) is None:
                doc_dict[term] = 0.0
            doc_dict[term] += 1.0

        for term in set(terms):
            if cls.document_frequency.get(term) is None:
                cls.document_frequency[term] = 0
            cls.document_frequency[term] += 1

        return doc_dict

    @classmethod
    def get_idf(cls, term):
        if cls.document_frequency.get(term) is None:
            return 1
        return log(cls.total_docs / cls.document_frequency[term])

    @classmethod
    def normalize_vector(cls, vector: dict):
        s = 0.0
        for x in vector.values():
            s += (x * x)
        s = sqrt(s)
        for term in vector:
            vector[term] /= s
