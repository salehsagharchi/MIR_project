import click
from Phase1 import Parser
from Phase1.Parser import TextNormalizer as Normalizer
from Phase1 import Constants
from Phase1.Bigram import Bigram
from Phase1.Indexer import Indexer
from Phase1.Score import Score

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

    def initialize_classes(self):
        print("loading...")
        try:
            Indexer.load_index()
            Bigram.load_file()
        except AttributeError:
            pass

    def parsing_files(self):
        self.Parser.parse_wiki()
        self.Parser.parse_tedtalks()

    def stopword_remove(self):
        self.Parser.remove_stopwords("fa")
        self.Parser.remove_stopwords("en")

    def bigram_search(self):
        bi = input("please enter a bigram: ")
        if len(bi) != 2:
            print("invalid input")
            return False
        print(Bigram.get_terms_of_bigram(bi))

    def create_index(self):
        Indexer.add_files()
        print("positional index was created successfully")

    def get_posting_list(self):
        term = input("please enter a term: ")
        res = Indexer.get_docs_containing_term(term)
        if len(res) > 0:
            print(f"the term \"{term}\" has occurred in documents: {res}")
        else:
            print(f"the term \"{term}\" doesn't exist in index")

    def get_positional_index(self):
        term = input("please enter a term: ")
        if Indexer.index.get(term) is not None:
            print("doc id\tpositions")
            for posting in Indexer.index[term].keys():
                print(f"{posting}\t{Indexer.index[term][posting]}")
        else:
            print(f"the term \"{term}\" doesn't exist in index")

    def save_via_var_byte(self):
        space = Indexer.save_index(Constants.VAR_BYTE_MODE)
        print("used space before compression: " + str(space[0]))
        print("used space after compression: " + str(space[1]))

    def save_via_gama_codes(self):
        space = Indexer.save_index(Constants.GAMA_CODES_MODE)
        print("used space before compression: " + str(space[0]))
        print("used space after compression: " + str(space[1]))

    def query(self):
        queryStatement = input("pls enter your query: ")
        queryStatement = queryStatement.split(' ')
        score = Score()
        print(score.query(queryStatement)[0:10])

    def save(self):
        Indexer.save_index()
        Bigram.save_to_file()

    def start(self):
        welcome_text = "با سلام به این برنامه خوش آمدید"
        print(Normalizer.reshape_text(welcome_text, "fa"))
        main_jobs = {
            "Parsing raw files and generating documents": self.parsing_files,
            "Removing stopwords": self.stopword_remove,
            "Make positional index": self.create_index,
            "Enter a term and see its posting list": self.get_posting_list,
            "Enter a term and see its positional index": self.get_positional_index,
            "Bigram searching": self.bigram_search,
            "Compressing indexes via VariableByte": self.save_via_var_byte,
            "Compressing indexes via GammaCode": self.save_via_gama_codes,
            "Query correction": self.query,
            "Search through documents": 9,
            "Save": self.save,
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
    my_main = Main(Constants.stopword_dir, Constants.docs_dir, Constants.tedtalks_raw, Constants.wiki_raw)
    my_main.initialize_classes()
    my_main.start()
