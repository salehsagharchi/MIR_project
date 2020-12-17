import os.path
import pickle
import re
import string
import unicodedata
import xml.etree.ElementTree as Xml
from time import sleep

import arabic_reshaper
from bidi.algorithm import get_display
from Phase2 import Constants

import click
import hazm
import nltk

import csv
from Phase2.DataModels import Document


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
    def get_word_language(text):
        farsi = False
        i = 0
        for ch in text:
            i += 1
            name = unicodedata.name(ch).lower()
            if 'arabic' in name or 'farsi' in name or 'persian' in name:
                farsi = True
                break
            if i == 10:
                break
        return "fa" if farsi else "en"

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
        tokens2 = hazm.word_tokenize(t)
        tokens = []
        for x in tokens2:
            tokens.extend(x.split("_"))
        stemmed_tokens = []
        for x in tokens:
            word = TextNormalizer.stemmer.stem(x)
            if word != "":
                stemmed_tokens.append(word)
        if tokenize:
            return stemmed_tokens
        return " ".join(stemmed_tokens)


class DocParser:

    def __init__(self, stopword_dir: str, docs_dir: str, tedtalks_raw_train: str, tedtalks_raw_test: str):
        self.stopword_dir = stopword_dir
        self.docs_dir = docs_dir
        self.tedtalks_raw_train = tedtalks_raw_train
        self.tedtalks_raw_test = tedtalks_raw_test
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

    def prepare_query(self, text):
        stopwords_path = [f"{self.stopword_dir}/stopwords_en.o", f"{self.stopword_dir}/stopwords_fa.o"]
        stopwords = []
        if not os.path.isfile(stopwords_path[0]) or not os.path.isfile(stopwords_path[1]):
            print("Stopwords file not found, try to make that.")
            return None

        with open(stopwords_path[0], "rb") as file:
            stopwords = pickle.load(file)

        with open(stopwords_path[1], "rb") as file:
            stopwords.extend(pickle.load(file))

        tokens = nltk.tokenize.word_tokenize(text)
        all_tokens = []
        to_show = []
        for t in tokens:
            lang = TextNormalizer.get_word_language(t)
            temp = TextNormalizer.prepare_text(t, lang)
            temp = list(filter(lambda x: x not in stopwords, temp))
            if temp:
                temp_toshow = list(map(lambda x: TextNormalizer.reshape_text(x, lang), temp))
                to_show.extend(temp_toshow)
                all_tokens.extend(temp)
        return all_tokens, TextNormalizer.reshape_text(" ".join(all_tokens), "fa")

    def parse_tedtalks(self, dump=True):
        print("Start parsing ted documents ...")

        all_tokens = dict()
        tokens_count = 0

        for is_test in [False, True]:
            with open((self.tedtalks_raw_test if is_test else self.tedtalks_raw_train), encoding='Latin1') as csvfile:
                csv_reader = list(csv.reader(csvfile, delimiter=','))
                line_count = -1
                with click.progressbar(label=f'Parsing Ted {"Test" if is_test else "Train"} Documents', length=len(csv_reader) + 1, fill_char="█") as bar:
                    bar.update(1)
                    sleep(0.5)
                    for row in csv_reader:
                        line_count += 1
                        bar.update(1)
                        if line_count == 0:
                            continue
                        else:
                            docid = str(line_count)
                            views = int(row[16])
                            title = row[14]
                            text = title + " " + row[1]
                            tokens = TextNormalizer.prepare_text(text, "en")
                            tokens_count += len(tokens)
                            for token in tokens:
                                all_tokens[token] = all_tokens.get(token, 0) + 1
                            doc = Document(docid, title, text, tokens, is_test, views)
                            if dump:
                                with open(f'{self.docs_dir}/{docid}_{"test" if is_test else "train"}.o', "wb") as file:
                                    pickle.dump(doc, file)
            print(f"{line_count} Documents parsed successfully.")

        stopwords = []
        stopword_threshold = 0.0079

        stopwords_path = f"{self.stopword_dir}/stopwords_en.o"

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
        #self.print_all_tokens("en")


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

        docfiles = os.listdir(self.docs_dir)

        with click.progressbar(label=f'Removing stopwords from {lang} documents', length=len(docfiles),
                               fill_char="█") as bar:
            for filename in docfiles:
                if filename.endswith(f".o"):
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
