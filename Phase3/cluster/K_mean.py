import json
import Phase3
from Phase3.vectorization import word2vec
from sklearn import cluster


class K_means:
    WORD2VEC_MODE = 0
    TF_IDF_MODE = 1

    def __init__(self, numCluster):
        self.NUM_CLUSTER = numCluster
        self.sentences = []
        self.kMeansCluster = None

    def setSentences(self, sentences):
        self.sentences = sentences

    def setSentencesFromFile(self, path):
        self.sentences = json.loads(open(path, "r", encoding="utf-8").read())

    def extractUsesFullText(self):
        result = []
        for sen in self.sentences:
            result.append(sen["summary"])
        return result

    def cluster(self, mode):
        if mode == K_means.WORD2VEC_MODE:
            self.word2vecCluster()

    def word2vecCluster(self):
        vecModule = word2vec.Word2vec()
        vecModule.initialGensimListFromArray(self.extractUsesFullText())
        vecModule.createWordVectors()
        vec = []
        for element in self.sentences:
            textFroAnalyse = element['summary']
            current = Phase3.Parser.TextNormalizer.prepare_text(textFroAnalyse, "fa")
            vec.append(vecModule.createSentenceVector(current))
        self.clusterKMeans(vec)

    def clusterKMeans(self, vec):
        self.kMeansCluster = cluster.KMeans(n_clusters=self.NUM_CLUSTER)
        self.kMeansCluster.fit(vec)


k = K_means(10)
k.setSentencesFromFile("Phase3/data/hamshahri.json")
k.cluster(K_means.WORD2VEC_MODE)
