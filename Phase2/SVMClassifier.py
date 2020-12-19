import copy
import os
import pickle

import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split

from Phase2.Tester import TestingType
from Phase2 import Constants
from Phase2.DataModels import Document
from Phase2.VectorSpaceModel import VectorSpaceCreator
from Utils import get_test_result


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
        if testing_type == TestingType.TRAIN:
            predicted = clf.predict(self.X_train)
            source = self.Y_train
        elif testing_type == TestingType.TEST:
            predicted = clf.predict(self.X_test)
            source = self.Y_test
        elif testing_type == TestingType.VALIDATION:
            predicted = clf.predict(self.X_valid)
            source = self.Y_valid

        return get_test_result(list(map(int, source)), list(map(int, predicted)))

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

