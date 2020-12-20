import math
import os
import pickle

import heapq
import click
import numpy as np

from Phase2 import Constants
from Phase2.DataModels import Document


class VectorSpaceCreator:
    def __init__(self):
        self.all_documents: dict = dict()
        self.tf_idf_train = None
        self.tf_idf_test = None
        self.most_freq: list = []
        self.idfs: list = []

    def load_documents(self):
        print("Loading documents ...")
        for filename in os.listdir(Constants.docs_dir):
            if filename.endswith('.o'):
                file_path = os.path.join(Constants.docs_dir, filename)
                with open(file_path, 'rb') as f:
                    doc: Document = pickle.load(f)
                    self.all_documents[filename] = doc

    def make_model(self):
        print("Please wait ...")
        wordfreq = dict()
        for doc in self.all_documents.values():
            for token in doc.tokens:
                if token not in wordfreq.keys():
                    wordfreq[token] = 1
                else:
                    wordfreq[token] += 1

        # self.most_freq = heapq.nlargest(10000, wordfreq, key=wordfreq.get)
        self.most_freq = []
        sortedwordfreq = sorted(wordfreq, key=wordfreq.get, reverse=True)

        for k in sortedwordfreq:
            if wordfreq[k] >= 20:
                self.most_freq.append(k)
            else:
                break

        tfidf_values = []
        with click.progressbar(label=f'Making Vector Space Model', length=len(self.most_freq),
                               fill_char="█") as bar:
            for token in self.most_freq:
                bar.update(1)
                doc_containing_word = 0
                sent_tf_vector = []
                for doc in self.all_documents.keys():
                    term_freq = 0
                    for word in self.all_documents[doc].tokens:
                        if token == word:
                            term_freq += 1
                    if term_freq > 0:
                        doc_containing_word += 1
                    sent_tf_vector.append((term_freq, doc))
                idf = math.log(len(self.all_documents) / doc_containing_word)
                self.idfs.append(idf)
                tfidf_sentences = [(element[0] * idf, element[1]) for element in sent_tf_vector]
                tfidf_values.append(tfidf_sentences)

        print("Please wait ...")

        tf_idf_general = np.asarray(tfidf_values)
        tf_idf_general = np.transpose(tf_idf_general)

        temp = np.array(tf_idf_general[0]).tolist()
        keys = np.array(tf_idf_general[1][:, 0])
        toView = lambda x: self.all_documents[x].view
        views = np.array([toView(xi) for xi in keys])
        self.tf_idf_train = []
        self.tf_idf_test = []
        for i in range(len(temp)):
            if 'test' in keys[i]:
                self.tf_idf_test.append([float(x) for x in temp[i]] + [views[i], keys[i]])
            else:
                self.tf_idf_train.append([float(x) for x in temp[i]] + [views[i], keys[i]])

        print("OK")

    def dumpModel(self):
        with open(f'{Constants.data_dir_root}/VectorSpaceModel.o', "wb") as file:
            pickle.dump(self, file)

    def create_td_idf_for_tokens(self, tokens):
        assert self.most_freq != [] and len(self.idfs) == len(self.most_freq)

        tfidf_values = []
        for i in range(len(self.most_freq)):
            term_freq = 0
            for word in tokens:
                if self.most_freq[i] == word:
                    term_freq += 1
            tfidf_values.append(term_freq * self.idfs[i])

        return np.array(tfidf_values)

    @staticmethod
    def readModel(fromphase1=False):
        path = f'{Constants.data_dir_root}/VectorSpaceModel.o'
        if fromphase1:
            path = "../Phase2/" + path
        if not os.path.isfile(path):
            print("Model file not found, try to make that.")
            return None
        with open(path, "rb") as file:
            modelobject: VectorSpaceCreator = pickle.load(file)
        return modelobject

    @staticmethod
    def dataSplit(vector_space_creator):
        tf_idf_train = np.array(vector_space_creator.tf_idf_train)
        tf_idf_test = np.array(vector_space_creator.tf_idf_test)
        assert tf_idf_train.shape[1] == tf_idf_test.shape[1]
        words_len = tf_idf_train.shape[1]
        Y_train = tf_idf_train[:, words_len - 2]
        Y_test = tf_idf_test[:, words_len - 2]
        X_train = np.delete(tf_idf_train, [words_len - 2, words_len - 1], 1)
        X_test = np.delete(tf_idf_test, [words_len - 2, words_len - 1], 1)
        return X_train, X_test, Y_train, Y_test
