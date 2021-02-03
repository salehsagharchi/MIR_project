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
        cls.document_vector_list.clear()
        cls.vocab.clear()
        cls.link_list.clear()
        documents = json.loads(open(FARSI_DOCUMENTS_PATH, 'r', encoding="utf-8").read())
        cls.total_docs = len(documents)
        cls.get_reference_labels(documents)

        document_dicts = []
        for doc in documents:
            list_of_terms = TextNormalizer.prepare_text(doc['summary'], "fa")
            document_dicts.append(cls.create_document_dict(list_of_terms))
            cls.link_list.append(doc['link'])

        for doc_dict in document_dicts:
            for term in doc_dict:
                doc_dict[term] *= cls.get_idf(term)
            cls.normalize_vector(doc_dict)

        cls.create_document_vectors(document_dicts)

    @classmethod
    def get_reference_labels(cls, documents):
        cls.reference_label_list.clear()
        for doc in documents:
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

    @classmethod
    def distance(cls, vector1: dict, vector2: dict):
        d = 0
        for term in vector1:
            if vector2.get(term) is None:
                d += vector1[term] ** 2
            else:
                d += (vector1[term] - vector2[term]) ** 2
        for term in vector2:
            if vector1.get(term) is None:
                d += vector2[term] ** 2

        return sqrt(d)


if __name__ == '__main__':
    pass
