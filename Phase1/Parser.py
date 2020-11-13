import os.path
import pickle
import re
import string
import xml.etree.ElementTree as Xml
from Phase1.DataModels import Document
import click
import hazm


class TextNormalizer:
    to_remove_fa = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUZWXYZ" + "،؛,|:»«َُِّ<>؟÷×" + "٬ًٌٍؘَؙُؚِّْٰٕٖٜٟؐؑؒؓؔؕؖؗٓٔٗ٘ٙٚٛٝٞ"
    regex_fa_space = re.compile('[%s]' % re.escape(string.punctuation))
    regex_fa_none = re.compile('[%s]' % re.escape(to_remove_fa))
    to_replace_fa = [("\u200c", " "), ("ـ", ""), ("آ", "ا"), ("ۀ", "ه"), ("ة", "ه"),
                     ("ي", "ی"), ("ئ", "ی"), ("ء", ""), ("أ", "ا"), ("إ", "ا"), ("ؤ", "و"), ("ك", "ک"),
                     ("۰", "0"), ("۱", "1"), ("۲", "2"), ("۳", "3"), ("۴", "4"), ("۵", "5"), ("۶", "6"), ("۷", "7"),
                     ("۸", "8"), ("۹", "9")]

    stemmer = hazm.Stemmer()

    @staticmethod
    def prepare_text(text, lang="fa", tokenize=True):
        if lang == "fa":
            return TextNormalizer.prepare_persian_text(text, tokenize)
        return TextNormalizer.prepare_english_text(text, tokenize)

    @staticmethod
    def prepare_english_text(text, tokenize):
        pass

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


def parse_wiki(dump=True):
    print("startparse")
    ns = "{http://www.mediawiki.org/xml/export-0.10/}"
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
            tokens = TextNormalizer.prepare_text(text)
            tokens_count += len(tokens)
            for token in tokens:
                all_tokens[token] = all_tokens.get(token, 0) + 1

            doc = Document(docid, title, normalized_title, text, tokens)
            if dump:
                with open(f"data/wiki_docs/{docid}.o", "wb") as file:
                    pickle.dump(doc, file)
            bar.update(1)

    print("endarse")
    print(i)

    stopwords = []
    stopword_threshold = 0.006

    if os.path.isfile('data/stopwords_fa.o'):
        with open(f"data/stopwords_fa.o", "rb") as file:
            stopwords = pickle.load(file)
            print(stopwords)
    else:
        sorted_word = {k: v for k, v in sorted(all_tokens.items(), key=lambda item: item[1], reverse=True)}
        for i, key in enumerate(sorted_word):
            if (sorted_word[key] / tokens_count) >= stopword_threshold:
                stopwords.append(key)
            else:
                break
        with open(f"data/stopwords_fa.o", "wb") as file:
            pickle.dump(stopwords, file)
        print(stopwords)

    # with open(f"data/wiki_docs/{3016}.o", "rb") as file:
    #     doc: Document = pickle.load(file)
    #     print(len(doc.tokens))
