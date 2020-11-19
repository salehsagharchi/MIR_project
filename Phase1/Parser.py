import os.path
import pickle
import re
import string
import xml.etree.ElementTree as Xml

import arabic_reshaper
from bidi.algorithm import get_display
from Phase1 import Constants

import click
import hazm
import nltk

import csv
from Phase1.DataModels import Document


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

    @staticmethod
    def reshape_text(raw_str, lang, reshape=Constants.reshape_persian_words):
        if lang != "fa":
            return raw_str
        if not reshape:
            return raw_str
        reshaped_text = arabic_reshaper.reshape(raw_str)
        return get_display(reshaped_text)


class DocParser:

    def __init__(self, stopword_dir: str, docs_dir: str, tedtalks_raw: str, wiki_raw: str):
        self.stopword_dir = stopword_dir
        self.docs_dir = docs_dir
        self.tedtalks_raw = tedtalks_raw
        self.wiki_raw = wiki_raw
        if not os.path.isdir(stopword_dir):
            os.makedirs(stopword_dir)
        if not os.path.isdir(docs_dir):
            os.makedirs(docs_dir)

    def print_all_tokens(self, lang="fa"):
        if click.confirm('\nDo you want to print all tokens?'):
            for filename in os.listdir(self.docs_dir):
                toprint = ""
                if filename.endswith(f"_{lang}.o"):
                    file_path = os.path.join(self.docs_dir, filename)
                    with open(file_path, "rb") as file:
                        doc: Document = pickle.load(file)
                        toprint = f"\nDocument #{doc.docid} with title : {TextNormalizer.reshape_text(doc.title, lang)} \n"
                        map_object = map(lambda x: TextNormalizer.reshape_text(x, lang), doc.tokens)
                        new_list = list(map_object)
                        toprint += str(new_list)
                print(toprint)
                if not click.confirm('\nDo you want to continue ?'):
                    break

    def parse_tedtalks(self, dump=True):
        print("Start parsing ted documents ...")

        all_tokens = dict()
        tokens_count = 0

        with open(self.tedtalks_raw, encoding='Latin1') as csvfile:
            csv_reader = list(csv.reader(csvfile, delimiter=','))
            line_count = -1
            with click.progressbar(label='Parsing Ted Documents', length=len(csv_reader), fill_char="█") as bar:
                for row in csv_reader:
                    line_count += 1
                    bar.update(1)
                    if line_count == 0:
                        continue
                    else:
                        docid = str(line_count)
                        title = row[14]
                        text = row[1]
                        normalized_title = TextNormalizer.prepare_text(title, "en", False)
                        tokens = TextNormalizer.prepare_text(text, "en")
                        tokens_count += len(tokens)
                        for token in tokens:
                            all_tokens[token] = all_tokens.get(token, 0) + 1
                        doc = Document(docid, title, normalized_title, text, tokens, "en")
                        if dump:
                            with open(f"{self.docs_dir}/{docid}_en.o", "wb") as file:
                                pickle.dump(doc, file)

        print(f"{line_count} Documents parsed successfully.")

        stopwords = []
        stopword_threshold = 0.0072

        stopwords_path = f"{self.stopword_dir}/stopwords_en.o"
        if os.path.isfile(stopwords_path):
            with open(stopwords_path, "rb") as file:
                stopwords = pickle.load(file)
        else:
            sorted_word = {k: v for k, v in sorted(all_tokens.items(), key=lambda item: item[1], reverse=True)}
            for i, key in enumerate(sorted_word):
                if (sorted_word[key] / tokens_count) >= stopword_threshold:
                    stopwords.append(key)
                else:
                    break
            with open(stopwords_path, "wb") as file:
                pickle.dump(stopwords, file)

        print("Calculated stopwords :", stopwords)
        print("Token counts :", tokens_count)
        self.print_all_tokens("en")

    def parse_wiki(self, dump=True):
        print("Start parsing wiki documents ...")

        ns = "{http://www.mediawiki.org/xml/export-0.10/}"
        root = Xml.parse(self.wiki_raw).getroot()

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

                doc = Document(docid, title, normalized_title, text, tokens, "fa")
                if dump:
                    with open(f"{self.docs_dir}/{docid}_fa.o", "wb") as file:
                        pickle.dump(doc, file)
                bar.update(1)

        print(f"{i} Documents parsed successfully.")

        stopwords = []
        stopword_threshold = 0.006

        stopwords_path = f"{self.stopword_dir}/stopwords_fa.o"
        if os.path.isfile(stopwords_path):
            with open(stopwords_path, "rb") as file:
                stopwords = pickle.load(file)
        else:
            sorted_word = {k: v for k, v in sorted(all_tokens.items(), key=lambda item: item[1], reverse=True)}
            for i, key in enumerate(sorted_word):
                if (sorted_word[key] / tokens_count) >= stopword_threshold:
                    stopwords.append(key)
                else:
                    break
            with open(stopwords_path, "wb") as file:
                pickle.dump(stopwords, file)

        map_object = map(lambda x: TextNormalizer.reshape_text(x, "fa"), stopwords)
        new_list = list(map_object)
        print("Calculated stopwords :", new_list)
        print("Token counts :", tokens_count)
        self.print_all_tokens("fa")

    def remove_stopwords(self, lang):
        stopwords_path = f"{self.stopword_dir}/stopwords_{lang}.o"

        stopwords = []
        if not os.path.isfile(stopwords_path):
            print("Stopwords file not found, try to make that.")
            return None

        with open(stopwords_path, "rb") as file:
            stopwords = pickle.load(file)

        alldiff = 0
        diff = 0

        docfiles = list(filter(lambda x: x.endswith(f"_{lang}.o"), os.listdir(self.docs_dir)))

        with click.progressbar(label=f'Removing stopwords from {lang} documents', length=len(docfiles),
                               fill_char="█") as bar:
            for filename in docfiles:
                if filename.endswith(f"_{lang}.o"):
                    file_path = os.path.join(self.docs_dir, filename)
                    new_doc: Document
                    with open(file_path, "rb") as file:
                        doc: Document = pickle.load(file)
                        new_doc = doc
                    diff = len(new_doc.tokens)
                    result = filter(lambda x: x not in stopwords, new_doc.tokens)
                    new_doc.tokens = list(result)
                    diff -= len(new_doc.tokens)
                    if diff > 0:
                        with open(file_path, "wb") as file:
                            pickle.dump(new_doc, file)
                        alldiff += diff
                bar.update(1)

        print(f"{alldiff} Stopwords removed from all {lang} documents.\n")
