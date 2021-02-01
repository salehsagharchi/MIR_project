import json
import Phase3.Parser
from gensim.models import Word2Vec


class Word2vec:
    def __init__(self):
        self.textArray = []
        self.gensimList = []
        self.vectors = []
        self.DIM = 1000

    def initialGensimListFromArray(self, sentences):
        self.textArray = sentences
        self.initialGensimList()

    def initialGensimList(self):
        for element in self.textArray:
            current = Phase3.Parser.TextNormalizer.prepare_text(element, "fa")
            self.gensimList.append(current)

    def createWordVectors(self):
        self.vectors = Word2Vec(self.gensimList, min_count=20, size=self.DIM, workers=3, window=3, sg=1)

    def createSentenceVector(self, sentence):
        vec = [0 for i in range(self.DIM)]
        counter = 0
        for word in sentence:
            if word in self.vectors.wv.vocab:
                vec += self.vectors.wv[word]
                counter += 1
        return [x / counter for x in vec]
