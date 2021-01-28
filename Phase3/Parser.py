import os.path
import pickle
import re
import string
import unicodedata
from time import sleep

import arabic_reshaper
from bidi.algorithm import get_display
from Phase3 import Constants

import click
import hazm
import nltk

import csv



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
    regex_en_unwanted = re.compile(r'[^A-Za-z0-9\s]+')
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
        t = TextNormalizer.regex_en_unwanted.sub(' ', t)
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

