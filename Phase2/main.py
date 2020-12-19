import os
import pickle
import click

from Phase2.NaiveBayesClassifier import NaiveBayesClassifier
from Phase2.VectorSpaceModel import VectorSpaceCreator
from Phase2.kNNClassifier import kNNClassifier
from Phase2 import Parser
from Phase2.Parser import TextNormalizer as Normalizer, TextNormalizer
from Phase2 import Constants
from Phase2.DataModels import Document
from Phase2.SVMClassifier import SVMClassifier
from Phase2.RandomForestClassifier import RFClassifier
from Phase2.Tester import TestingType


def prompt_from_list(options: list, prompt_msg="Please Select One Option"):
    n = len(options)
    click.secho("\n:: ", nl=False, fg="green")
    click.echo(prompt_msg)
    for i in range(1, n + 1):
        click.secho(" " + str(i) + " ", nl=False, fg="blue")
        click.echo(options[i - 1])
    choice = click.prompt(
        click.style('> ', fg='green'),
        type=click.IntRange(1, n),
        prompt_suffix="",
    )
    print("\n")
    return choice - 1


class Main:

    def __init__(self, stopword_dir: str, docs_dir: str, tedtalks_raw_train: str, tedtalks_raw_test: str):
        self.stopword_dir = stopword_dir
        self.docs_dir = docs_dir
        self.tedtalks_raw_train = tedtalks_raw_train
        self.tedtalks_raw_test = tedtalks_raw_test
        self.parser: Parser.DocParser = Parser.DocParser(stopword_dir, docs_dir, tedtalks_raw_train, tedtalks_raw_test)
        self.vector_space_model = None

    # def initialize_classes(self):
    #     print("Loading files ...")
    #     try:
    #         Preferences.load_pref()
    #         Indexer.load_index()
    #         Bigram.load_file()
    #     except AttributeError:
    #         pass

    def parsing_files(self):
        self.parser.parse_tedtalks()

    def stopword_remove(self):
        self.parser.remove_stopwords("en")

    def naive_bayes_test(self):
        NaiveBayesClassifier.start()
        test_files = input("test with test data or train data? (default is test data)")
        if test_files == '' or test_files != 'train':
            test_files = 'test'
        print(f"Naive Bayes Classification test with {test_files} data:")
        print(NaiveBayesClassifier.test(test_files))

    def kNN_test(self):
        best_accuracy, best_k = 0, -1
        kNNClassifier.start()
        for k in Constants.kNN_parameter_list:
            result = kNNClassifier.validation(k)
            print(f"validating with 10% of train data with k={k}")
            print(result)
            print()
            if result[Constants.ACCURACY] > best_accuracy:
                best_accuracy = result[Constants.ACCURACY]
                best_k = k
        print(f"best parameter k for this train data is {best_k} with accuracy {best_accuracy}")
        print(f"kNN classification testing with k={best_k}")
        print(kNNClassifier.test())

    def create_vector_model(self):
        if not os.path.isfile(f'{Constants.data_dir_root}/VectorSpaceModel.o'):
            self.vector_space_model = VectorSpaceCreator()
            self.vector_space_model.load_documents()
            self.vector_space_model.make_model()
            self.vector_space_model.dumpModel()
            print("Vector space model created successfully !")
        else:
            self.vector_space_model = VectorSpaceCreator.readModel()
            print("Vector space model loaded successfully !")
        return True

    def svm_test(self):
        if self.vector_space_model is None:
            print("Vector space model is not ready, trying to make or load it ...")
            if not self.create_vector_model():
                return
        Cs = (input("\nPlease enter your C parameters list (separated with ','): "))
        clist = list(map(lambda x: x.strip(), Cs.split(",")))
        svm = SVMClassifier(self.vector_space_model)
        svm.batch_fiting(clist)

    def RFTest(self):
        if self.vector_space_model is None:
            print("Vector space model is not ready, trying to make or load it ...")
            if not self.create_vector_model():
                return
        rf = RFClassifier(self.vector_space_model)
        rf.fit()
        result = rf.test(TestingType.TRAIN)
        result2 = rf.test(TestingType.TEST)
        print(f'result for testing with TRAIN Data : {result}')
        print(f'result for testing with TEST Data : {result2}')

    def start(self):
        welcome_text = "Welcome to this application !"
        print(welcome_text)
        main_jobs = {
            "Parsing raw files and generating documents": self.parsing_files,
            "Removing stopwords": self.stopword_remove,
            "Create or load vector space model": self.create_vector_model,
            "Naive Bayes test": self.naive_bayes_test,
            "kNN test": self.kNN_test,
            "SVM Classification": self.svm_test,
            "Random Forest Classification": self.RFTest,
            "EXIT": -1
        }
        finish = False
        while not finish:
            job = prompt_from_list(list(main_jobs), "Please select a job you want to execute : ")
            command = list(main_jobs.values())[job]
            finish = command == -1
            if callable(command):
                command()


if __name__ == "__main__":
    my_main = Main(Constants.data_dir_root, Constants.docs_dir, Constants.tedtalks_raw_train, Constants.tedtalks_raw_test)
    my_main.start()
