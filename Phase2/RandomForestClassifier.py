from sklearn.ensemble import RandomForestClassifier

from Phase2.Tester import TestingType
from Phase2.VectorSpaceModel import VectorSpaceCreator
from Phase2.Utils import get_test_result


class RFClassifier:
    def __init__(self, model):
        self.vector_space = model
        self.X_train, self.X_test, self.Y_train, self.Y_test = VectorSpaceCreator.dataSplit(self.vector_space)
        self.clf = None

    def fit(self):
        self.clf = RandomForestClassifier(n_estimators=100)
        self.clf.fit(self.X_train, self.Y_train)

    def test(self, testing_type: TestingType):
        if testing_type == TestingType.TRAIN:
            predicted = self.clf.predict(self.X_train)
            source = self.Y_train
        elif testing_type == TestingType.TEST:
            predicted = self.clf.predict(self.X_test)
            source = self.Y_test
        else:
            return None

        return get_test_result(list(map(int, source)), list(map(int, predicted)))
