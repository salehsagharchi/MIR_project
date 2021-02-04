from math import sqrt

from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt

from TFIDF import TFIDF
from Phase3.Constants import *
from Phase3.Utils import *
from Phase3.word2vec import *


class KMeansClustering:
    vector_list = list()
    link_list = list()
    reference_label_list = list()
    label_list = list()

    @classmethod
    def start(cls, vectorization_mode=TFIDF_MODE):
        cls.label_list = list()
        if vectorization_mode == TFIDF_MODE:
            TFIDF.start()
            cls.link_list = TFIDF.link_list
            cls.vector_list = TFIDF.document_vector_list
            cls.reference_label_list = TFIDF.reference_label_list
        elif vectorization_mode == WORD2VEC_MODE:
            w2v = Word2vec(2, 2000, 3, 3, 1)
            cls.vector_list, cls.link_list, cls.reference_label_list = w2v.createVectorsOfSentenceByPath(FARSI_DOCUMENTS_PATH)

    @classmethod
    def get_k(cls):
        return int(sqrt(len(cls.link_list))) + 1

    @classmethod
    def cluster(cls, vectors=None, vectorization_mode=TFIDF_MODE):
        if vectors is None:
            vectors = cls.vector_list
        kmeans = KMeans(n_clusters=cls.get_k(), random_state=0).fit(vectors)
        cls.label_list = kmeans.labels_
        cls.write_results_to_file(vectorization_mode)

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
    def read_results_from_file(cls, vectorization_mode=TFIDF_MODE):
        cls.link_list = list()
        cls.label_list = list()
        cls.reference_label_list = list()
        path = KMEANS_TFIDF_RESULT_FILE
        if vectorization_mode == WORD2VEC_MODE:
            path = KMEANS_WORD2VEC_RESULT_FILE
        with open(path, mode='r', encoding="utf-8") as csv_file:
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
        return evaluate_clustering(cls.label_list, cls.reference_label_list)

    @classmethod
    def get_graphical_results(cls, vectorization_mode=TFIDF_MODE):
        cls.start(vectorization_mode)
        k_list = []
        purity_list = []
        ARI_list = []
        NMI_list = []
        AMI_list = []

        cls.cluster(vectorization_mode=vectorization_mode)
        evaluation = evaluate_clustering(cls.label_list, cls.reference_label_list)
        k = len(cls.vector_list[0])
        k_list.append(k)
        purity_list.append(evaluation['purity'])
        ARI_list.append(evaluation['ARI'])
        NMI_list.append(evaluation['NMI'])
        AMI_list.append(evaluation['AMI'])
        k = int(k / 1.2)

        while k > cls.get_k():
            new_vectors = cls.select_k_best_features(k)
            cls.cluster(vectors=new_vectors, vectorization_mode=vectorization_mode)
            evaluation = evaluate_clustering(cls.label_list, cls.reference_label_list)
            k_list.append(k)
            purity_list.append(evaluation['purity'])
            ARI_list.append(evaluation['ARI'])
            NMI_list.append(evaluation['NMI'])
            AMI_list.append(evaluation['AMI'])
            print(k, evaluation)
            k = int(k / 1.2)

        plt.plot(k_list, purity_list, label='purity')
        plt.plot(k_list, ARI_list, label='ARI (adjusted rand index)')
        plt.plot(k_list, NMI_list, label='NMI (normalized mutual info)')
        plt.plot(k_list, AMI_list, label='AMI (adjusted mutual info)')
        plt.ylim(0, 1)
        plt.xlabel('vector size')
        plt.legend()
        plt.text(k, 0.9, f'number of clusters = {cls.get_k()}')
        plt.title('changes w.r.t. number of features selected')
        if vectorization_mode == TFIDF_MODE:
            plt.savefig('data/kmeans_tfidf_plot_4.png')
        elif vectorization_mode == WORD2VEC_MODE:
            plt.savefig('data/kmeans_word2vec_plot_3.png')
        plt.close()


if __name__ == '__main__':
    # change this
    # mode = WORD2VEC_MODE
    #
    # KMeansClustering.start(vectorization_mode=mode)
    # vectors = None
    # if mode == TFIDF_MODE:
    #     vectors = KMeansClustering.select_k_best_features(3500)
    # KMeansClustering.cluster(vectors=vectors, vectorization_mode=mode)
    # vs = len(KMeansClustering.vector_list[0])
    # print(vs, KMeansClustering.evaluate())

    KMeansClustering.get_graphical_results(WORD2VEC_MODE)
