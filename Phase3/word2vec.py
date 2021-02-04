import json
import Phase3.Parser
from gensim.models import Word2Vec


class Word2vec:
    def __init__(self, minCount, DIM, workers, windows, sg):
        self.textArray = []
        self.gensimList = []
        self.vectors = []
        self.links = []
        self.references = []
        self.DIM = DIM
        self.minCount = minCount
        self.workers = workers
        self.windows = windows
        self.sg = sg

    def initialGensimListFromArray(self, sentences):
        self.textArray = sentences
        self.initialGensimList()

    def initialGensimList(self):
        for element in self.textArray:
            current = Phase3.Parser.TextNormalizer.prepare_text(element, "fa")
            self.gensimList.append(current)

    def createWordVectors(self):
        self.vectors = Word2Vec(self.gensimList, min_count=self.minCount, size=self.DIM, workers=self.workers,
                                window=self.windows, sg=self.sg)

    def createSentenceVector(self, sentence):
        vec = [0 for i in range(self.DIM)]
        counter = 0
        for word in sentence:
            if word in self.vectors.wv.vocab:
                for i in range(len(vec)):
                    vec[i] += self.vectors.wv[word][i]
                counter += 1
        return [x / counter for x in vec]

    def get_reference_label(self, doc, primary_mode=True):
        if primary_mode:
            tag = doc['tags'][0].split('>')[0][:-1]
        else:
            tag = doc['tags'][0].split('>')[1][1:]
        return tag

    def createVectorsOfSentenceByPath(self, path):
        sentences = json.loads(open(path, "r", encoding="utf-8").read())
        for sent in sentences:
            currentText = sent["summary"] + " " + sent["title"]
            self.links.append(sent['link'])
            self.references.append(self.get_reference_label(sent))
            self.textArray.append(currentText)
            parsedText = Phase3.Parser.TextNormalizer.prepare_text(currentText, "fa")
            self.gensimList.append(parsedText)
        self.createWordVectors()
        result = []
        for sent in self.textArray:
            result.append(self.createSentenceVector(sent))
        return result, self.links, self.references


w2v = Word2vec(4, 100, 3, 3, 1)
