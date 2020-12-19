import os
import pickle

from Phase2.DataModels import Document, kNNData
from Phase2 import Constants
from enum import Enum


class TestingType(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3

class Tester:

    @classmethod
    def test(cls, classifier_type, test_function):
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0

        for filename in os.listdir(Constants.docs_dir):
            if filename.endswith('test.o'):
                file_path = os.path.join(Constants.docs_dir, filename)
                with open(file_path, 'rb') as f:
                    doc: Document = pickle.load(f)
                    label = Constants.label_index[doc.view]
                    if classifier_type == Constants.naive_bayes:
                        prediction = test_function(doc.tokens)
                    elif classifier_type == Constants.kNN:
                        prediction = test_function(kNNData(doc.tokens, None))
                    if bool(label):
                        if bool(prediction):
                            true_positive += 1
                        else:
                            false_negative += 1
                    else:
                        if bool(prediction):
                            false_positive += 1
                        else:
                            true_negative += 1

        result = {}
        result[Constants.PRECISION] = round(true_positive / (true_positive + false_positive), 6)
        result[Constants.RECALL] = round(true_positive / (true_positive + false_negative), 6)
        result[Constants.ACCURACY] = round((true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative), 6)
        result[Constants.F1] = round((2 * result[Constants.PRECISION] * result[Constants.RECALL]) / (result[Constants.PRECISION] + result[Constants.RECALL]), 6)
        return result