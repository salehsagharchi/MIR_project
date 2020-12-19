import os
import pickle

from Phase2.DataModels import Document, kNNData
from Phase2 import Constants


def test_classifier(classifier_type, test_function, test_files):
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    for filename in os.listdir(Constants.docs_dir):
        if filename.endswith(test_files + '.o'):
            file_path = os.path.join(Constants.docs_dir, filename)
            with open(file_path, 'rb') as f:
                doc: Document = pickle.load(f)

                if classifier_type == Constants.naive_bayes:
                    prediction = test_function(doc.tokens)
                elif classifier_type == Constants.kNN:
                    prediction = test_function(kNNData(doc.tokens, None))

                label = Constants.label_index[doc.view]
                if bool(label):
                    if label == prediction:
                        true_positive += 1
                    else:
                        false_negative += 1
                else:
                    if label == prediction:
                        true_negative += 1
                    else:
                        false_positive += 1

    result = {}
    # print('true_positive', true_positive)
    # print('false_positive', false_positive)
    # print('true_negative', true_negative)
    # print('false_negative', false_negative)
    if (true_positive + false_positive) == 0:
        precision = 0
    else:
        precision = round(true_positive / (true_positive + false_positive), 6)
    if (true_positive + false_negative) == 0:
        recall = 0
    else:
        recall = round(true_positive / (true_positive + false_negative), 6)
    result[Constants.PRECISION] = precision
    result[Constants.RECALL] = recall
    result[Constants.ACCURACY] = round((true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative), 6)
    result[Constants.F1] = round((2 * precision * recall) / (precision + recall), 6)
    return result
