from math import sqrt

from sklearn.mixture import GaussianMixture

from TFIDF import TFIDF
from Phase3.Constants import *
from Phase3.Utils import *


class GMMClustering:
    vector_list = list()
    label_list = list()
    reference_label_list = list()
    link_list = list()

    @classmethod
    def start(cls):
        cls.label_list = list()
        TFIDF.start()
        cls.link_list = TFIDF.link_list
        cls.vector_list = TFIDF.document_vector_list
        cls.reference_label_list = TFIDF.reference_label_list

    @classmethod
    def cluster_by_tfidf(cls):
        gmm = GaussianMixture(n_components=cls.get_n_components())
        gmm.fit(cls.vector_list)
        cls.label_list = gmm.predict(cls.vector_list)
        cls.write_results_to_file(TFIDF_MODE)

    @classmethod
    def get_n_components(cls):
        return int(sqrt(len(cls.link_list)) / 2) + 1

    @classmethod
    def write_results_to_file(cls, mode):
        if mode == TFIDF_MODE:
            path = GMM_TFIDF_RESULT_FILE
        elif mode == WORD2VEC_MODE:
            path = GMM_WORD2VEC_RESULT_FILE
        else:
            return
        write_to_file(path, cls.link_list, cls.label_list, cls.reference_label_list)

    @classmethod
    def read_results_from_file(cls):
        cls.link_list = list()
        cls.label_list = list()
        cls.reference_label_list = list()
        with open(GMM_TFIDF_RESULT_FILE, mode='r', encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0 and (line_count % 2 == 0):
                    cls.link_list.append(row[0])
                    cls.label_list.append(int(row[1]))
                    cls.reference_label_list.append(row[2])
                    pass
                line_count += 1

    @classmethod
    def evaluate(cls):
        return evaluate_clustering(cls.get_n_components(), cls.label_list, cls.reference_label_list)


if __name__ == '__main__':
    GMMClustering.start()
    GMMClustering.cluster_by_tfidf()
    # GMMClustering.read_results_from_file()
    print(GMMClustering.evaluate())
