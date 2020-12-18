import os
import pickle
from math import sqrt

from Phase2.DataModels import Document, kNNData
from Phase2 import Constants


class kNNClassifier:
    k = 1
    train_data = list()

    @classmethod
    def start(cls, docs_dir=Constants.docs_dir, k=1):
        cls.k = k
        cls.train_data.clear()
        for filename in os.listdir(docs_dir):
            if filename.endswith('train.o'):
                file_path = os.path.join(docs_dir, filename)
                with open(file_path, 'rb') as f:
                    doc: Document = pickle.load(f)
                    cls.train_data.append(kNNData(doc.tokens, Constants.label_index[doc.view]))

    @classmethod
    def distance(cls, vector1: dict, vector2: dict):
        d = 0.0
        for term in vector2:
            if vector1.get(term) is not None:
                d += (vector1[term] - vector2[term]) ** 2
        return sqrt(d)

    @classmethod
    def get_k_nearest_neighbours(cls, target: kNNData):
        neighbours = []
        [neighbours.append((data, cls.distance(data.vector, target.vector))) for data in cls.train_data]
        neighbours.sort(key=lambda x: x[1])

        if cls.k >= len(neighbours):
            return [n[0] for n in neighbours]
        return [n[0] for n in neighbours[:cls.k]]

    @classmethod
    def classify(cls, target: kNNData):
        nearest_neighbours = cls.get_k_nearest_neighbours(target)
        labels = [n.label for n in nearest_neighbours]
        return max(set(labels), key=labels.count)

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
                    prediction = cls.classify(kNNData(doc.tokens, None))
                    if prediction == Constants.label_index[doc.view]:
                        correct_prediction_count += 1
        return correct_prediction_count / total_tests
