import os
import pickle
from time import sleep

import click
from Phase1 import Parser
from Phase1.Parser import TextNormalizer as Normalizer, TextNormalizer
from Phase1 import Constants
from Phase1.Bigram import Bigram
from Phase1.Indexer import Indexer
from Phase1.Score import Score
from Phase1.Preferences import Preferences
from Phase1.DataModels import Document
from Phase2.VectorSpaceModel import VectorSpaceCreator
import Phase2.Constants as PH2Cons


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
        self.parser: Parser.DocParser = Parser.DocParser(stopword_dir, docs_dir, tedtalks_raw, wiki_raw)
        self.vector_space_model: VectorSpaceCreator = None
        self.clf_model = None
        self.all_documents: dict = dict()

    def initialize_classes(self):
        print("Loading files ...")
        try:
            Preferences.load_pref()
            Indexer.load_index()
            Bigram.load_file()
        except AttributeError:
            pass

    def parsing_files(self):
        self.parser.parse_wiki()
        self.parser.parse_tedtalks()

    def stopword_remove(self):
        self.parser.remove_stopwords("fa")
        self.parser.remove_stopwords("en")

    def bigram_search(self):
        bi = input("please enter a bigram: ")
        if len(bi) != 2:
            print("invalid input")
            return False
        print(Bigram.get_terms_of_bigram(bi))

    def create_index(self):
        Indexer.add_files()
        print("positional index was created successfully!")

    def get_posting_list(self):
        term = input("please enter a term: ")
        term_norm = self.parser.prepare_query(term)[0]
        if len(term_norm) != 1:
            print("invalid input, please enter only one term")
            return
        term = term_norm[0]
        print("Normalized Term :", term)
        res = Indexer.get_docs_containing_term(term)
        if len(res) > 0:
            print(f"the term \"{term}\" has occurred in documents: {res}")
        else:
            print(f"the term \"{term}\" doesn't exist in index")

    def get_positional_index(self):
        term = input("please enter a term: ")
        term_norm = self.parser.prepare_query(term)[0]
        if len(term_norm) != 1:
            print("invalid input, please enter only one term")
            return
        term = term_norm[0]
        print("Normalized Term :", term)
        if Indexer.index.get(term) is not None:
            print("doc id\tpositions")
            for posting in Indexer.index[term].keys():
                print(f"{posting}\t{Indexer.index[term][posting]}")
        else:
            print(f"the term \"{term}\" doesn't exist in index")

    def save_via_var_byte(self):
        Preferences.pref[Constants.pref_compression_type_key] = Constants.VAR_BYTE_MODE
        Preferences.save_pref()
        space = Indexer.save_index()
        print("used space before compression: " + str(space[0]))
        print("used space after compression: " + str(space[1]))

    def save_via_gama_codes(self):
        Preferences.pref[Constants.pref_compression_type_key] = Constants.GAMA_CODES_MODE
        Preferences.save_pref()
        space = Indexer.save_index()
        print("used space before compression: " + str(space[0]))
        print("used space after compression: " + str(space[1]))

    def jaccard(self):
        terms = input("please enter 2 terms separated by space: ").split(' ')
        if len(terms) != 2:
            print("invalid input")
            return
        bi_set1 = set([terms[0][i:i + 2] for i in range(len(terms[0]) - 1)])
        bi_set2 = set([terms[1][i:i + 2] for i in range(len(terms[1]) - 1)])
        print(f"Jaccard similarity between \"{terms[0]}\" and \"{terms[1]}\" is: {Bigram.jaccard_measure(bi_set1, bi_set2)}")

    def edit_distance(self):
        terms = input("please enter 2 terms separated by space: ").split(' ')
        if len(terms) != 2:
            print("invalid input")
            return
        print(f"Edit distance for \"{terms[0]}\" and \"{terms[1]}\" is: {Bigram.edit_distance_measure(terms[0], terms[1])}")

    def query(self):
        queryStatement = input("pls enter your query: ")
        score = Score()
        normalized_query = self.parser.prepare_query(queryStatement)
        queryTokens = normalized_query[0]
        print("Normalized query :", normalized_query[1])
        result = score.query(queryTokens)
        if result is not None:
           print(result[0:min(10, len(result))])

    def query_phase2(self):
        if self.load_dependencies() is None:
            return None
        queryStatement = input("Pls enter your query: ")
        queryView = input("Pls enter your view: ")
        labelView = queryView
        if labelView != "1" and labelView != "-1":
            print("Please Enter valid view !")
            return None
        if labelView == "-1":
            labelView = "0"
        score = Score()
        normalized_query = self.parser.prepare_query(queryStatement)
        queryTokens = normalized_query[0]
        print("Normalized query :", normalized_query[1])
        results = score.query(queryTokens)
        if results is None:
            return None
        classified_results = []
        for item in results:
            if str(item[0]) in self.all_documents:
                tokens = self.all_documents[str(item[0])].tokens
                vector = self.vector_space_model.create_td_idf_for_tokens(tokens)
                label = self.clf_model.predict([vector])[0]
                if str(label) == labelView:
                    classified_results.append(item)
        print(f"Found {len(results)} results and {len(classified_results)} with view = {queryView} : ")
        print(classified_results)


    def printdoc(self):
        docid = input("Enter a document id : ")
        path = f"{self.docs_dir}/{docid}_fa.o"
        if not os.path.isfile(path):
            path = f"{self.docs_dir}/{docid}_en.o"

        if not os.path.isfile(path):
            print(f"Couldn't find a document with id = {docid}")
            return

        with open(path, "rb") as file:
            doc: Document = pickle.load(file)
            title = TextNormalizer.reshape_text(doc.title, "fa")
            toprint = f"\nDocument #{doc.docid} with title : {title}\n"
            map_object = map(lambda x: TextNormalizer.reshape_text(x, "fa"), doc.tokens)
            new_list = list(map_object)
            toprint += str(new_list)
        print(toprint)

    def save(self):
        Preferences.save_pref()
        Indexer.save_index()
        Bigram.save_bigram()
        print("files were saved successfully!")

    def load_dependencies(self):
        if not self.all_documents:
            print("Loading documents ...")
            for filename in os.listdir(Constants.docs_dir):
                if filename.endswith('_en.o'):
                    file_path = os.path.join(Constants.docs_dir, filename)
                    with open(file_path, 'rb') as f:
                        doc: Document = pickle.load(f)
                        doc.tokens = doc.normalized_title.split(" ") + doc.tokens
                        self.all_documents[filename.split("_")[0].strip()] = doc
            if not self.all_documents:
                return None
        if self.vector_space_model is None:
            print("Loading vector space model of phase2 docs ...")
            self.vector_space_model = VectorSpaceCreator.readModel(True)
            if self.vector_space_model is None:
                return None
        if self.clf_model is None:
            print("Loading RF trained model ...")
            if not os.path.isfile("../Phase2/" + PH2Cons.rf_clf_model):
                print("Cannot find RF trained model. Try to make that.")
                return None
            with open("../Phase2/" + PH2Cons.rf_clf_model, "rb") as file:
                self.clf_model = pickle.load(file)
        return True

    def start(self):
        welcome_text = "با سلام به این برنامه خوش آمدید"
        print(Normalizer.reshape_text(welcome_text, "fa"))
        main_jobs = {
            "Parsing raw files and generating documents": self.parsing_files,
            "Removing stopwords": self.stopword_remove,
            "Make positional index": self.create_index,
            "Enter a term and see its posting list": self.get_posting_list,
            "Enter a term and see its positional index": self.get_positional_index,
            "Enter docid and see doc details": self.printdoc,
            "Bigram searching": self.bigram_search,
            "Compressing indexes via VariableByte": self.save_via_var_byte,
            "Compressing indexes via GammaCode": self.save_via_gama_codes,
            "Jaccard similarity for 2 terms": self.jaccard,
            "Edit distance for 2 terms": self.edit_distance,
            "Search through documents": self.query,
            "Phase2::Search through documents with view (Only Ted Docs)": self.query_phase2,
            "Save everything": self.save,
            "EXIT": -1
        }
        finish = False
        while not finish:
            sleep(0.5)
            job = prompt_from_list(list(main_jobs), "Please select a job you want to execute : ")
            command = list(main_jobs.values())[job]
            finish = command == -1
            if callable(command):
                command()


if __name__ == "__main__":
    my_main = Main(Constants.stopword_dir, Constants.docs_dir, Constants.tedtalks_raw, Constants.wiki_raw)
    my_main.initialize_classes()
    my_main.start()
