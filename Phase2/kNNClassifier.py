import os
import pickle
from math import sqrt, log

from Phase2.DataModels import Document, kNNData
from Phase2 import Constants
from Phase2.Utils import test_classifier, get_test_result


class kNNClassifier:
    k = 1
    total_docs = 0
    train_data = list()
    document_frequency = dict()

    @classmethod
    def start(cls, k=1):
        cls.k = k
        cls.total_docs = 0
        cls.train_data.clear()
        cls.document_frequency.clear()
        for filename in os.listdir(Constants.docs_dir):
            if filename.endswith('train.o'):
                cls.total_docs += 1
                file_path = os.path.join(Constants.docs_dir, filename)
                with open(file_path, 'rb') as f:
                    doc: Document = pickle.load(f)
                    cls.add_doc_to_data(doc.tokens, Constants.label_index[doc.view])

        for data in cls.train_data:
            cls.multiply_vector_by_idf(data)

    @classmethod
    def add_doc_to_data(cls, tokens: list, label):
        cls.train_data.append(kNNData(tokens, label))

        for term in set(tokens):
            if cls.document_frequency.get(term) is None:
                cls.document_frequency[term] = 0
            cls.document_frequency[term] += 1

    @classmethod
    def get_idf(cls, term):
        if cls.document_frequency.get(term) is None:
            return 1
        return log(cls.total_docs / cls.document_frequency[term])

    @classmethod
    def multiply_vector_by_idf(cls, data: kNNData):
        if data.is_multiplied_by_idf:
            return
        for term in data.vector:
            data.vector[term] *= cls.get_idf(term)
        data.is_multiplied_by_idf = True

    @classmethod
    def distance(cls, vector1: dict, vector2: dict):
        d = 0.0
        for term in vector2:
            if vector1.get(term) is not None:
                d += (vector1[term] - vector2[term]) ** 2
        return sqrt(d)

    @classmethod
    def get_k_nearest_neighbours(cls, target: kNNData, validation=False):
        neighbours = []
        start = int(validation) * (len(cls.train_data) // 10)
        for data in cls.train_data[start:]:
            neighbours.append((data, cls.distance(data.vector, target.vector)))
        neighbours.sort(key=lambda x: x[1])

        if cls.k >= len(neighbours):
            return [n[0] for n in neighbours]
        return [n[0] for n in neighbours[:cls.k]]

    @classmethod
    def classify(cls, target: kNNData, validation=False):
        cls.multiply_vector_by_idf(target)
        nearest_neighbours = cls.get_k_nearest_neighbours(target, validation)
        labels_count = [0 for _ in range(len(Constants.label_index))]
        for neighbour in nearest_neighbours:
            labels_count[neighbour.label] += 1
        return labels_count.index(max(labels_count))

    @classmethod
    def validation(cls, k):
        cls.k = k
        predictions = []
        validation_size = len(cls.train_data) // 10
        for i in range(validation_size):
            predictions.append(cls.classify(cls.train_data[i], validation=True))

        labels = [data.label for data in cls.train_data[:validation_size]]
        return get_test_result(labels, predictions)

    @classmethod
    def test(cls, test_files='test'):
        return test_classifier(Constants.kNN, cls.classify, test_files)
