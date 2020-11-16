import os.path
import pickle
import re
import string
import xml.etree.ElementTree as Xml

import click
import hazm
import nltk

import csv
from Phase1.DataModels import Document
from Phase1.main import reshape_text


class TextNormalizer:
    to_remove_fa = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUZWXYZ" + "،؛,|:»«َُِّ<>؟÷×" + "٬ًٌٍؘَؙُؚِّْٰٕٖٜٟؐؑؒؓؔؕؖؗٓٔٗ٘ٙٚٛٝٞ"
    regex_fa_space = re.compile('[%s]' % re.escape(string.punctuation))
    regex_fa_none = re.compile('[%s]' % re.escape(to_remove_fa))
    to_replace_fa = [("\u200c", " "), ("ـ", ""), ("آ", "ا"), ("ۀ", "ه"), ("ة", "ه"),
                     ("ي", "ی"), ("ئ", "ی"), ("ء", ""), ("أ", "ا"), ("إ", "ا"), ("ؤ", "و"), ("ك", "ک"),
                     ("۰", "0"), ("۱", "1"), ("۲", "2"), ("۳", "3"), ("۴", "4"), ("۵", "5"), ("۶", "6"), ("۷", "7"),
                     ("۸", "8"), ("۹", "9")]

    stemmer = hazm.Stemmer()

    regex_en_space = re.compile('[%s]' % re.escape(string.punctuation))
    porter_stemmer = nltk.stem.PorterStemmer()

    @staticmethod
    def prepare_text(text, lang="fa", tokenize=True):
        if lang == "fa":
            return TextNormalizer.prepare_persian_text(text, tokenize)
        return TextNormalizer.prepare_english_text(text, tokenize)

    @staticmethod
    def prepare_english_text(text, tokenize):
        t = text.casefold()
        t = TextNormalizer.regex_en_space.sub(' ', t)
        tokens = nltk.tokenize.word_tokenize(t)
        stemmed_tokens = []
        for x in tokens:
            word = TextNormalizer.porter_stemmer.stem(x)
            if word != "":
                stemmed_tokens.append(word)
        if tokenize:
            return stemmed_tokens
        return " ".join(stemmed_tokens)

    @staticmethod
    def prepare_persian_text(text, tokenize):
        t = text
        for tup in TextNormalizer.to_replace_fa:
            for ch in tup[0]:
                t = t.replace(ch, tup[1])
        t = TextNormalizer.regex_fa_space.sub(' ', t)
        t = TextNormalizer.regex_fa_none.sub('', t)
        tokens = hazm.word_tokenize(t)
        stemmed_tokens = []
        for x in tokens:
            word = TextNormalizer.stemmer.stem(x)
            if word != "":
                stemmed_tokens.append(word)
        if tokenize:
            return stemmed_tokens
        return " ".join(stemmed_tokens)


def print_all_tokens(lang="fa"):
    if lang == "fa":
        doc_directory = "data/wiki_docs/"
    else:
        doc_directory = "data/ted_docs/"

    if click.confirm('\nDo you want to print all tokens?'):
        for filename in os.listdir(doc_directory):
            toprint = ""
            if filename.endswith(".o"):
                file_path = os.path.join(doc_directory, filename)
                with open(file_path, "rb") as file:
                    doc: Document = pickle.load(file)
                    toprint = f"\nDocument #{doc.docid} with title : {reshape_text(doc.title, lang)} \n"
                    map_object = map(lambda x: reshape_text(x, lang), doc.tokens)
                    new_list = list(map_object)
                    toprint += str(new_list)
            print(toprint)
            if not click.confirm('\nDo you want to continue ?'):
                break


def parse_tedtalks(dump=True):

    all_tokens = dict()
    tokens_count = 0

    with open("data/raw_data/ted_talks.csv", encoding='Latin1') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        line_count = -1
        for row in csv_reader:
            line_count += 1
            if line_count == 0:
                continue
            else:
                docid = str(line_count)
                title = row[14]
                text = row[1]
                normalized_title = TextNormalizer.prepare_text(title, "en", False)
                tokens = TextNormalizer.prepare_text(text, "en")



        print(f'Processed {line_count} lines.')


def parse_wiki(dump=True):
    print("Start parsing wiki documents ...")

    ns = "{http://www.mediawiki.org/xml/export-0.10/}"
    wiki_directory = "data/wiki_docs/"
    root = Xml.parse("data/raw_data/Persian.xml").getroot()

    all_tokens = dict()
    tokens_count = 0
    i = 0

    pages = list(root.iterfind(f"{ns}page"))
    with click.progressbar(label='Parsing Wiki Documents', length=len(pages), fill_char="█") as bar:
        page: Xml.Element
        for page in pages:
            i += 1
            docid = next(page.iterfind(f"{ns}id")).text
            title = next(page.iterfind(f"{ns}title")).text
            text = next(next(page.iterfind(f"{ns}revision")).iterfind(f"{ns}text")).text
            normalized_title = TextNormalizer.prepare_text(title, "fa", False)
            tokens = TextNormalizer.prepare_text(text, "fa")
            tokens_count += len(tokens)
            for token in tokens:
                all_tokens[token] = all_tokens.get(token, 0) + 1

            doc = Document(docid, title, normalized_title, text, tokens)
            if dump:
                with open(f"{wiki_directory}{docid}.o", "wb") as file:
                    pickle.dump(doc, file)
            bar.update(1)

    print(f"{i} Documents parsed successfully.")

    stopwords = []
    stopword_threshold = 0.006

    if os.path.isfile('data/stopwords_fa.o'):
        with open(f"data/stopwords_fa.o", "rb") as file:
            stopwords = pickle.load(file)
    else:
        sorted_word = {k: v for k, v in sorted(all_tokens.items(), key=lambda item: item[1], reverse=True)}
        for i, key in enumerate(sorted_word):
            if (sorted_word[key] / tokens_count) >= stopword_threshold:
                stopwords.append(key)
            else:
                break
        with open(f"data/stopwords_fa.o", "wb") as file:
            pickle.dump(stopwords, file)

    map_object = map(lambda x: reshape_text(x, "fa"), stopwords)
    new_list = list(map_object)
    print("Calculated stopwords :", new_list)
    print("Token counts :", tokens_count)
    print_all_tokens("fa")

    # with open(f"data/wiki_docs/{3016}.o", "rb") as file:
    #     doc: Document = pickle.load(file)
    #     print(len(doc.tokens))


def remove_stopwords(lang):
    if lang == "fa":
        doc_directory = "data/wiki_docs/"
        stopwords_file = "data/stopwords_fa.o"
    else:
        doc_directory = "data/ted_docs/"
        stopwords_file = "data/stopwords_en.o"

    stopwords = []
    if not os.path.isfile(stopwords_file):
        print("Stopwords file not found, try to make that.")
        return None

    with open(stopwords_file, "rb") as file:
        stopwords = pickle.load(file)

    diff = 0

    with click.progressbar(label='Removing stopwords from documents', length=len(os.listdir(doc_directory)),
                           fill_char="█") as bar:
        for filename in os.listdir(doc_directory):
            if filename.endswith(".o"):
                file_path = os.path.join(doc_directory, filename)
                new_doc: Document
                with open(file_path, "rb") as file:
                    doc: Document = pickle.load(file)
                    new_doc = doc
                diff += len(new_doc.tokens)
                result = filter(lambda x: x not in stopwords, new_doc.tokens)
                new_doc.tokens = list(result)
                diff -= len(new_doc.tokens)
                with open(file_path, "wb") as file:
                    pickle.dump(new_doc, file)
            bar.update(1)

    print(f"{diff} Stopwords removed from all {lang} documents.")

    with open(f"data/wiki_docs/{3016}.o", "rb") as file:
        doc: Document = pickle.load(file)
        print(len(doc.tokens))
