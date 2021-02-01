import json
import Phase3.Parser
from gensim.models import Word2Vec


class Word2vec:
    def __init__(self):
        self.textArray = []
        self.gensimList = []
        self.vectors = []

    def initialTextArrayFromFile(self, path):
        arr = json.loads(open(path, "r", encoding="utf-8").read())
        for element in arr:
            textFroAnalyse = element['summary']
            current = Phase3.Parser.TextNormalizer.prepare_text(textFroAnalyse, "fa")
            current.append(element["link"])
            self.gensimList.append(current)

    def addText(self, newTexts):
        self.textArray += newTexts

    def createVectors(self):
        self.vectors = Word2Vec(self.gensimList, min_count=1, size=50, workers=3, window=3, sg=1)
