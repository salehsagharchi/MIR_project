import os
import pickle
from math import log, inf

from Phase2 import Constants
from Phase2.DataModels import Document


class NaiveBayes:
    table = dict()
    total_terms_in_class = list()
    number_of_classes = len(Constants.label_index)

    @classmethod
    def start(cls, docs_dir=Constants.docs_dir):
        cls.table.clear()
        for filename in os.listdir(docs_dir):
            if filename.endswith('train.o'):
                file_path = os.path.join(docs_dir, filename)
                with open(file_path, 'rb') as f:
                    doc: Document = pickle.load(f)
                    cls.add_doc_to_table(doc.tokens, Constants.label_index[doc.view])

        cls.total_terms_in_class = [0 for _ in range(cls.number_of_classes)]
        for term_counts in cls.table.values():
            for c in range(cls.number_of_classes):
                cls.total_terms_in_class[c] += term_counts[c]

        vocabulary_size = len(cls.table)
        for term in cls.table:
            for c in range(cls.number_of_classes):
                cls.table[term][c] = (cls.table[term][c] + 1) / (cls.total_terms_in_class[c] + vocabulary_size)

    @classmethod
    def add_doc_to_table(cls, tokens: list, label):
        for term in tokens:
            if cls.table.get(term) is None:
                cls.table[term] = [0 for _ in range(cls.number_of_classes)]
            cls.table[term][label] += 1

    @classmethod
    def classify(cls, tokens):
        max_score = -inf
        best_class = -1
        for c in range(cls.number_of_classes):
            # calculating score of a class
            score = log(cls.total_terms_in_class[c] / sum(cls.total_terms_in_class))
            for term in tokens:
                if cls.table.get(term) is not None:
                    score += log(cls.table[term][c])

            if score > max_score:
                max_score = score
                best_class = c

        return best_class

    @classmethod
    def test(cls, docs_dir=Constants.docs_dir):
        correct_prediction_count = 0
        total_tests = 0
        for filename in os.listdir(docs_dir):
            if filename.endswith('test.o'):
                total_tests += 1
                file_path = os.path.join(docs_dir, filename)
                with open(file_path, 'rb') as f:
                    doc: Document = pickle.load(f)
                    if cls.classify(doc.tokens) == Constants.label_index[doc.view]:
                        correct_prediction_count += 1
        return correct_prediction_count / total_tests
