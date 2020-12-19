import copy
import os
import pickle

import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split

from Phase2.Tester import Tester, TestingType
from Phase2 import Constants
from Phase2.DataModels import Document
from Phase2.VectorSpaceModel import VectorSpaceCreator


class SVMClassifier:
    def __init__(self, model):
        self.vector_space = model
        self.tf_idf_train = np.array(self.vector_space.tf_idf_train)
        self.X, self.X_test, self.Y, self.Y_test = VectorSpaceCreator.dataSplit(self.vector_space)
        self.X_train, self.X_valid, self.Y_train, self.Y_valid = train_test_split(self.X, self.Y, test_size=0.1, random_state=20)
        self.clf = None

    def fit(self, penalty):
        self.clf = svm.SVC(kernel='linear', C=penalty)
        self.clf.fit(self.X_train, self.Y_train)

    def test(self, testing_type: TestingType, clf):
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0

        source = []
        predicted = []
        if testing_type == TestingType.TRAIN:
            predicted = clf.predict(self.X_train)
            source = self.Y_train
        elif testing_type == TestingType.TEST:
            predicted = clf.predict(self.X_test)
            source = self.Y_test
        elif testing_type == TestingType.VALIDATION:
            predicted = clf.predict(self.X_valid)
            source = self.Y_valid

        for i in range(len(source)):
            if source[i] == "1":
                if predicted[i] == "1":
                    true_positive += 1
                else:
                    false_negative += 1
            else:
                if predicted[i] == "1":
                    false_positive += 1
                else:
                    true_negative += 1

        result = {}
        result[Constants.PRECISION] = round(true_positive / (true_positive + false_positive), 6)
        result[Constants.RECALL] = round(true_positive / (true_positive + false_negative), 6)
        result[Constants.ACCURACY] = round((true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative), 6)
        result[Constants.F1] = round(
            (2 * result[Constants.PRECISION] * result[Constants.RECALL]) / (result[Constants.PRECISION] + result[Constants.RECALL]), 6)
        return result

    def batch_fiting(self, Cs):
        max_acc = -1
        best_clf = None
        best_c = 0
        for c in Cs:
            print("\nStart Fitting for C = " + c + " :")
            self.fit(float(c))
            print("Fitting Finished. Test model for validation data ...")
            result = self.test(TestingType.VALIDATION, self.clf)
            acc = result[Constants.ACCURACY]
            print(f'For C = {c}  ==>  Accuracy = {acc}')
            if acc > max_acc:
                max_acc = acc
                best_c = c
                best_clf = copy.deepcopy(self.clf)

        print(f"\nMax Accuracy = {max_acc} for C = {best_c}")
        print(f"\nStart Test Model for All Data with C = {best_c} ...")
        result = self.test(TestingType.TRAIN, best_clf)
        result2 = self.test(TestingType.VALIDATION, best_clf)
        result3 = self.test(TestingType.TEST, best_clf)
        print(f'Result for TRAIN Data : {result}')
        print(f'Result for VALIDATION Data : {result2}')
        print(f'Result for TEST Data : {result3}')

