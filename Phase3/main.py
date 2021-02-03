import os
import pickle
import click

from Phase3 import Parser
from Phase3.Crawler import Crawler
from Phase3.PageRank import PageRank
from Phase3.Parser import TextNormalizer as Normalizer, TextNormalizer
from Phase3 import Constants


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

    def start(self):
        welcome_text = "Welcome to this application !"
        print(welcome_text)
        main_jobs = {
            "Crawl Microsoft Academic Papers": self.crawler,
            "Calculate Page Rank of Saved Papers": self.page_rank,
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
