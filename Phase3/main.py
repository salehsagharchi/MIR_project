import os
import pickle
import click

from Phase3 import Parser
from Phase3.Crawler import Crawler
from Phase3.PageRank import PageRank
from Phase3.Parser import TextNormalizer as Normalizer, TextNormalizer
from Phase3 import Constants
from Phase3.KMeansClustering import KMeansClustering
from Phase3.GMMClustering import GMMClustering
from Phase3.HierarchicalClustering import HierarchicalClustering
from Phase3.Constants import *


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
    print("")
    return choice - 1


class Main:
    def crawler(self):
        limit = input("Enter Limit (default 5000) : ")
        try:
            limit = int(limit)
            assert limit > 0
        except:
            limit = 5000
        c = Crawler(Constants.crawler_start_file, limit, Constants.crawler_data_dir_root, "")
        c.start_crawling()
        c.close_crawler()
        print("OK")

    def page_rank(self):
        alpha = input("Enter Alpha (default 0.85) : ")
        try:
            alpha = float(alpha)
            assert 0 <= alpha <= 1
        except:
            alpha = 0.85
        pagerank = PageRank(os.path.join(Constants.crawler_data_dir_root, "result.json"))
        pagerank.make_graph()
        pagerank.calculate_page_rank(alpha)

    def kmeans_tfidf(self):
        print("please wait, this might take a minute ...")
        vec_size = 111
        KMeansClustering.start(vectorization_mode=TFIDF_MODE)
        vectors = KMeansClustering.select_k_best_features(vec_size)
        KMeansClustering.cluster(vectors=vectors, vectorization_mode=TFIDF_MODE)
        print(f"K-means Clustering by tf-idf with {KMeansClustering.get_k()} clusters and vector size {vec_size}")
        print(KMeansClustering.evaluate())

    def kmeans_w2v(self):
        print("please wait, this might take a minute ...")
        KMeansClustering.start(vectorization_mode=WORD2VEC_MODE)
        KMeansClustering.cluster(vectorization_mode=WORD2VEC_MODE)
        print(f"K-means Clustering by word2vec with {KMeansClustering.get_k()} clusters and vector size {len(KMeansClustering.vector_list[0])}")
        print(KMeansClustering.evaluate())

    def gmm_tfidf(self):
        print("please wait, this might take a minute ...")
        vec_size = 140
        GMMClustering.start(vectorization_mode=TFIDF_MODE)
        vectors = GMMClustering.select_k_best_features(vec_size)
        GMMClustering.cluster(vectors=vectors, vectorization_mode=TFIDF_MODE)
        print(f"GMM Clustering by tf-idf with {GMMClustering.get_n_components()} clusters and vector size {vec_size}")
        print(GMMClustering.evaluate())

    def gmm_w2v(self):
        print("please wait, this might take a minute ...")
        GMMClustering.start(vectorization_mode=WORD2VEC_MODE)
        GMMClustering.cluster(vectors=None, vectorization_mode=WORD2VEC_MODE)
        print(f"GMM Clustering by word2vec with {GMMClustering.get_n_components()} clusters and vector size {len(GMMClustering.vector_list[0])}")
        print(GMMClustering.evaluate())

    def hierarchical_tfidf(self):
        print("please wait, this might take a minute ...")
        vec_size = 83
        HierarchicalClustering.start(vectorization_mode=TFIDF_MODE)
        vectors = HierarchicalClustering.select_k_best_features(vec_size)
        HierarchicalClustering.cluster(vectors=vectors, vectorization_mode=TFIDF_MODE)
        print(f"Hierarchical Clustering by tf-idf with {HierarchicalClustering.get_k()} clusters and vector size {vec_size}")
        print(HierarchicalClustering.evaluate())

    def hierarchical_w2v(self):
        print("please wait, this might take a minute ...")
        HierarchicalClustering.start(vectorization_mode=WORD2VEC_MODE)
        HierarchicalClustering.cluster(vectorization_mode=WORD2VEC_MODE)
        print(f"Hierarchical Clustering by word2vec with {HierarchicalClustering.get_k()} clusters and vector size {len(HierarchicalClustering.vector_list[0])}")
        print(HierarchicalClustering.evaluate())

    def start(self):
        welcome_text = "Welcome to this application !"
        print(welcome_text)
        main_jobs = {
            "Crawl Microsoft Academic Papers": self.crawler,
            "Calculate Page Rank of Saved Papers": self.page_rank,
            "K-means Clustering by tf-idf vectorization": self.kmeans_tfidf,
            "K-means Clustering by word2vec vectorization": self.kmeans_w2v,
            "GMM Clustering by tf-idf vectorization": self.gmm_tfidf,
            "GMM Clustering by word2vec vectorization": self.gmm_w2v,
            "Hierarchical Clustering by tf-idf vectorization": self.hierarchical_tfidf,
            "Hierarchical Clustering by word2vec vectorization": self.hierarchical_w2v,
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
    my_main = Main()
    my_main.start()
