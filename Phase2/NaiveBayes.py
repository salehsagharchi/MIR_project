import os
import pickle
from math import log, inf

from Phase2 import Constants
from Phase2.DataModels import Document


class NaiveBayes:
    number_of_classes = 2
    table = {}
    total_terms_in_class = [0 for _ in range(number_of_classes)]

    @classmethod
    def start(cls, docs_dir=Constants.docs_dir):
        for filename in os.listdir(docs_dir):
            if filename.endswith('.o'):
                file_path = os.path.join(docs_dir, filename)
                with open(file_path, 'rb') as f:
                    doc: Document = pickle.load(f)
                    cls.add_doc_to_table(doc.tokens, doc.view)

        for term_counts in cls.table.values():
            for c in range(cls.number_of_classes):
                cls.total_terms_in_class[c] += term_counts[c]

        vocabulary_size = len(cls.table)
        for term in cls.table:
            for c in range(cls.number_of_classes):
                cls.table[term][c] = (cls.table[term][c] + 1) / (cls.total_terms_in_class[c] + vocabulary_size)

    @classmethod
    def add_doc_to_table(cls, document: list, label):
        for term in document:
            if cls.table.get(term) is None:
                cls.table[term] = [0.0 for _ in range(cls.number_of_classes)]
            cls.table[term][label] += 1

    @classmethod
    def classify(cls, document):
        max_score = -inf
        best_class = -1
        total_terms = sum(cls.total_terms_in_class)
        for c in range(cls.number_of_classes):
            score = log(cls.total_terms_in_class[c] / total_terms)
            for term in document:
                score += log(cls.table[term][c])
            if score > max_score:
                max_score = score
                best_class = c
        return best_class
