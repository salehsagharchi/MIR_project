import click

from Phase1 import Parser
from Phase1.Parser import TextNormalizer as Normalizer
from Phase1 import Constants


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

    def __init__(self, stopword_dir: str, docs_dir: str, tedtalks_raw: str, wiki_raw: str):
        self.stopword_dir = stopword_dir
        self.docs_dir = docs_dir
        self.tedtalks_raw = tedtalks_raw
        self.wiki_raw = wiki_raw
        self.Parser: Parser.DocParser = Parser.DocParser(stopword_dir, docs_dir, tedtalks_raw, wiki_raw)

    def parsing_files(self):
        self.Parser.parse_wiki()
        self.Parser.parse_tedtalks()

    def stopword_remove(self):
        self.Parser.remove_stopwords("fa")
        self.Parser.remove_stopwords("en")

    def start(self):
        welcome_text = "با سلام به این برنامه خوش آمدید"
        print(Normalizer.reshape_text(welcome_text, "fa"))
        finish = False
        while not finish:
            main_jobs = {
                "Parsing raw files and generating documents": self.parsing_files,
                "Removing stopwords": self.stopword_remove,
                "Make positional index": 3,
                "Enter a term and see its positional index": 4,
                "Bigram searching": 5,
                "Compressing indexes via VariableByte": 6,
                "Compressing indexes via GammaCode": 7,
                "Query correction": 8,
                "Search through documents": 9,
                "EXIT": -1
            }

            job = prompt_from_list(list(main_jobs), "Please select a job you want to execute : ")
            command = list(main_jobs.values())[job]
            finish = command == -1
            if callable(command):
                command()


if __name__ == "__main__":
    my_main = Main(Constants.stopword_dir, Constants.docs_dir, Constants.tedtalks_raw, Constants.wiki_raw)
    my_main.start()
