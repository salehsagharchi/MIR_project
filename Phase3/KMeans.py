import csv
from collections import Counter
from math import sqrt, inf, log

from sklearn.cluster import KMeans
from scipy.stats import entropy

from TFIDF import TFIDF
from Phase3.Constants import *


class KMeansClustering:
    vector_list = list()
    label_list = list()
    reference_label_list = list()
    link_list = list()

    @classmethod
    def start(cls):
        cls.label_list.clear()
        TFIDF.start()
        cls.link_list = TFIDF.link_list
        cls.vector_list = TFIDF.document_vector_list
        cls.reference_label_list = TFIDF.reference_label_list

    @classmethod
    def get_k(cls):
        return int(sqrt(len(cls.label_list))) + 1

    @classmethod
    def cluster_by_tfidf(cls):
        kmeans = KMeans(n_clusters=cls.get_k(), random_state=0).fit(cls.vector_list)
        cls.label_list = kmeans.labels_
        cls.write_results_to_file()

    @classmethod
    def calculate_rss(cls, kmeans):
        rss = 0
        centers = kmeans.cluster_centers_
        for i, label in enumerate(kmeans.labels_):
            rss += cls.distance2(cls.vector_list[i], centers[label])
        return rss

    @classmethod
    def distance2(cls, vector1, vector2):
        if len(vector1) != len(vector2):
            return inf
        d = 0
        for i in range(len(vector1)):
            d += (vector1[i] - vector2[i]) ** 2
        return d

    @classmethod
    def write_results_to_file(cls):
        with open(KMEANS_TFIDF_RESULT_FILE, mode='w', encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["link", "label", "reference"])
            for i in range(len(cls.link_list)):
                csv_writer.writerow([cls.link_list[i], str(cls.label_list[i]), cls.reference_label_list[i]])

    @classmethod
    def read_results_from_file(cls):
        cls.link_list.clear()
        cls.label_list.clear()
        cls.reference_label_list.clear()
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
    def calculate_purity(cls):
        reference_labels_for_each_cluster = [[] for _ in range(cls.get_k())]
        for i, label in enumerate(cls.label_list):
            reference_labels_for_each_cluster[label].append(cls.reference_label_list[i])
        count_of_purity = 0
        for i in range(len(reference_labels_for_each_cluster)):
            occurence_count = Counter(reference_labels_for_each_cluster[i])
            most_common = occurence_count.most_common(1)[0][0]
            count_of_purity += reference_labels_for_each_cluster[i].count(most_common)
        print(count_of_purity)
        return count_of_purity / len(cls.label_list)

    @classmethod
    def create_contingency_matrix(cls):
        reference_label_index = dict()
        reference_label_set = set(cls.reference_label_list)
        for i, label in enumerate(reference_label_set):
            reference_label_index[label] = i
        contingency_matrix = [[0 for j in range(len(reference_label_set))] for i in range(cls.get_k())]
        for i, label in enumerate(cls.label_list):
            reference_index = reference_label_index[cls.reference_label_list[i]]
            contingency_matrix[label][reference_index] += 1
        return contingency_matrix

    @classmethod
    def calculate_ARI(cls):
        contingency = cls.create_contingency_matrix()
        total_sigma, row_sigma, col_sigma = 0, 0, 0
        for i in range(len(contingency)):
            row = sum(contingency[i])
            row_sigma += (row * (row - 1)) / 2
            for j in range(len(contingency[0])):
                c = contingency[i][j]
                total_sigma += (c * (c - 1)) / 2
        for j in range(len(contingency[0])):
            col = 0
            for i in range(len(contingency)):
                col += contingency[i][j]
            col_sigma += (col * (col - 1)) / 2
        n = len(cls.label_list)
        temp = row_sigma * col_sigma / (n * (n - 1) / 2)
        return ((total_sigma - temp) / ((row_sigma + col_sigma) / 2 - temp))

    @classmethod
    def calculate_NMI(cls):
        contingency = cls.create_contingency_matrix()
        U = [sum(row) for row in contingency]
        V = []
        for j in range(len(contingency[0])):
            s = 0
            for i in range(len(contingency)):
                s += contingency[i][j]
            V.append(s)
        sum_U, sum_V, N = sum(U), sum(V), len(cls.label_list)
        U = [x / sum_U for x in U]
        V = [x / sum_V for x in V]
        MI = 0
        for i in range(len(contingency)):
            for j in range(len(contingency[0])):
                if contingency[i][j] == 0:
                    continue
                P_uv = contingency[i][j] / N
                MI += P_uv * log(P_uv / (U[i] * V[j]))
        return (MI / (0.5 * (entropy(U) + entropy(V))))


if __name__ == '__main__':
    KMeansClustering.start()
    KMeansClustering.cluster_by_tfidf()
    KMeansClustering.read_results_from_file()
    print(KMeansClustering.calculate_purity())
    print(KMeansClustering.calculate_ARI())
    print(KMeansClustering.calculate_NMI())
