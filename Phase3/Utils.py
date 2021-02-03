import csv
from collections import Counter
from math import log

from scipy.stats import entropy


def write_to_file(path, link_list: list, label_list: list, reference_label_list: list):
    with open(path, mode='w', encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["link", "label", "reference"])
        for i in range(len(link_list)):
            csv_writer.writerow([link_list[i], str(label_list[i]), reference_label_list[i]])


def evaluate_clustering(n_clusters, label_list: list, reference_label_list: list):
    evaluation = dict()
    evaluation['purity'] = round(calculate_purity(n_clusters, label_list, reference_label_list), 6)
    evaluation['ARI'] = round(calculate_ARI(n_clusters, label_list, reference_label_list), 6)
    evaluation['NMI'] = round(calculate_NMI(n_clusters, label_list, reference_label_list), 6)
    return evaluation


def create_contingency_matrix(n_clusters, label_list: list, reference_label_list: list):
    reference_label_index = dict()
    reference_label_set = set(reference_label_list)
    for i, label in enumerate(reference_label_set):
        reference_label_index[label] = i
    contingency_matrix = [[0 for j in range(len(reference_label_set))] for i in range(n_clusters)]
    for i, label in enumerate(label_list):
        reference_index = reference_label_index[reference_label_list[i]]
        contingency_matrix[label][reference_index] += 1
    return contingency_matrix


def calculate_purity(n_clusters, label_list: list, reference_label_list: list):
    reference_labels_for_each_cluster = [[] for _ in range(n_clusters)]
    for i, label in enumerate(label_list):
        reference_labels_for_each_cluster[label].append(reference_label_list[i])
    count_of_purity = 0
    for i in range(len(reference_labels_for_each_cluster)):
        occurrence_count = Counter(reference_labels_for_each_cluster[i])
        most_common = occurrence_count.most_common(1)[0][0]
        count_of_purity += reference_labels_for_each_cluster[i].count(most_common)
    return count_of_purity / len(label_list)


def calculate_ARI(n_clusters, label_list: list, reference_label_list: list):
    contingency = create_contingency_matrix(n_clusters, label_list, reference_label_list)
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
    n = len(label_list)
    temp = row_sigma * col_sigma / (n * (n - 1) / 2)
    return ((total_sigma - temp) / ((row_sigma + col_sigma) / 2 - temp))


def calculate_NMI(n_clusters, label_list: list, reference_label_list: list):
    contingency = create_contingency_matrix(n_clusters, label_list, reference_label_list)
    U = [sum(row) for row in contingency]
    V = []
    for j in range(len(contingency[0])):
        s = 0
        for i in range(len(contingency)):
            s += contingency[i][j]
        V.append(s)
    sum_U, sum_V, N = sum(U), sum(V), len(label_list)
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
