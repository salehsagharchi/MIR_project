import pickle
import os, json

from Phase3.DataModels import Paper
from Phase3 import Constants, Crawler
import numpy as np
import networkx as nx


class PageRank:
    def __init__(self, json_file):
        self.json_file = json_file
        self.node_list = []
        self.graph: nx.DiGraph = None
        self.papers = []
        with open(json_file, "r") as file:
            json_dict: dict = json.loads("".join(file.readlines()))
        for paper in json_dict['papers']:
            self.papers.append(Crawler.Crawler.convert_dict_to_paper(paper))

    def make_graph(self):
        print("Making Graph ...")
        for paper in self.papers:
            if paper.id not in self.node_list:
                self.node_list.append(paper.id)
            for paper_ref in paper.references:
                if paper_ref not in self.node_list:
                    self.node_list.append(paper_ref)
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.node_list)
        for paper in self.papers:
            for paper_ref in paper.references:
                self.graph.add_edge(paper.id, paper_ref)
        print("Graph Maked")

    def calculate_page_rank(self, alpha=0.85):
        print("Calculating Pagerank ...")
        pr = nx.pagerank(self.graph, alpha=alpha)
        print("TOP 100 RANKS :")
        topranks = [(k, pr[k]) for k in sorted(pr, key=pr.get, reverse=True)]
        topranks = topranks[0:100]
        for i, rank in enumerate(topranks):
            print(f"{i + 1}: \033[36m{rank[0]}\033[0m ==> \033[34m{rank[1]}\033[0m")
        print("\nFINISHED")
