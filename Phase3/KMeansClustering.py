from math import sqrt

from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt

from TFIDF import TFIDF
from Phase3.Constants import *
from Phase3.Utils import *


class KMeansClustering:
    vector_list = list()
    link_list = list()
    reference_label_list = list()
    label_list = list()

    @classmethod
    def start(cls):
        cls.label_list = list()
        TFIDF.start()
        cls.link_list = TFIDF.link_list
        cls.vector_list = TFIDF.document_vector_list
        cls.reference_label_list = TFIDF.reference_label_list

    @classmethod
    def get_k(cls):
        return int(sqrt(len(cls.link_list))) + 1

    @classmethod
    def cluster_by_tfidf(cls, vectors=None):
        if vectors is None:
            vectors = cls.vector_list
        kmeans = KMeans(n_clusters=cls.get_k(), random_state=0).fit(vectors)
        cls.label_list = kmeans.labels_
        cls.write_results_to_file(TFIDF_MODE)

    @classmethod
    def write_results_to_file(cls, mode):
        if mode == TFIDF_MODE:
            path = KMEANS_TFIDF_RESULT_FILE
        elif mode == WORD2VEC_MODE:
            path = KMEANS_WORD2VEC_RESULT_FILE
        else:
            return
        write_to_file(path, cls.link_list, cls.label_list, cls.reference_label_list)

    @classmethod
    def read_results_from_file(cls):
        cls.link_list = list()
        cls.label_list = list()
        cls.reference_label_list = list()
        with open(KMEANS_TFIDF_RESULT_FILE, mode='r', encoding="utf-8") as csv_file:
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
    def select_k_best_features(cls, k):
        return SelectKBest(chi2, k=k).fit_transform(cls.vector_list, cls.reference_label_list)

    @classmethod
    def evaluate(cls):
        return evaluate_clustering(cls.get_k(), cls.label_list, cls.reference_label_list)

    @classmethod
    def get_graphical_results(cls):
        cls.start()
        k = cls.get_k()
        k_list = []
        purity_list = []
        ARI_list = []
        NMI_list = []
        while k < 6300:
            new_vectors = cls.select_k_best_features(k)
            cls.cluster_by_tfidf(new_vectors)
            evaluation = evaluate_clustering(cls.get_k(), cls.label_list, cls.reference_label_list)
            k_list.append(k)
            purity_list.append(evaluation['purity'])
            ARI_list.append(evaluation['ARI'])
            NMI_list.append(evaluation['NMI'])
            k = int(k * 1.2)
        cls.cluster_by_tfidf()
        evaluation = evaluate_clustering(cls.get_k(), cls.label_list, cls.reference_label_list)
        k_list.append(len(cls.vector_list[0]))
        purity_list.append(evaluation['purity'])
        ARI_list.append(evaluation['ARI'])
        NMI_list.append(evaluation['NMI'])

        plt.plot(k_list, purity_list, label='purity')
        plt.plot(k_list, ARI_list, label='ARI')
        plt.plot(k_list, NMI_list, label='NMI')
        plt.ylim(0, 1)
        plt.xlabel('vector size')
        plt.legend()
        plt.title('changes w.r.t. number of features selected')
        plt.savefig('data/kmeans_tfidf - plot 1.png')
        plt.close()


if __name__ == '__main__':
    KMeansClustering.get_graphical_results()

