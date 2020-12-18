import os
import pickle
from math import sqrt

from Phase2.DataModels import Document, kNNData
from Phase2 import Constants


class kNNClassifier:
    train_data = []

    @classmethod
    def start(cls, docs_dir=Constants.docs_dir):
        for filename in os.listdir(docs_dir):
            if filename.endswith('.o'):
                file_path = os.path.join(docs_dir, filename)
                with open(file_path, 'rb') as f:
                    doc: Document = pickle.load(f)
                    cls.train_data.append(kNNData(doc.tokens, doc.view))

    @classmethod
    def distance(cls, vector1, vector2):
        d = 0.0
        for i in range(len(vector1)):
            d += (vector1[i] - vector2[i]) ** 2
        return sqrt(d)

    @classmethod
    def get_k_nearest_neighbours(cls, target, k):
        neighbours = []
        [neighbours.append((data, cls.distance(data.vector, target))) for data in cls.train_data]
        neighbours.sort(key=lambda x: x[1])

        if k >= len(neighbours):
            return [n[0] for n in neighbours]
        return [n[0] for n in neighbours[:k]]

    @classmethod
    def classify(cls, target, k):
        nearest_neighbours = cls.get_k_nearest_neighbours(target, k)
        labels = [n.label for n in nearest_neighbours]
        return max(set(labels), key=labels.count)
