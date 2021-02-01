import json
import operator

import Phase3
from Phase3.vectorization import word2vec
from sklearn import cluster
import csv


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

    def writeToFile(self):
        labels = self.kMeansCluster.labels_
        with open('innovators.csv', 'w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["link", "label"])
            for i in range(len(labels)):
                currentLink = self.sentences[i]["link"]
                currentLabel = labels[i]
                print(currentLink)
                print(currentLabel)
                writer.writerow([currentLink, currentLabel])

    def evaluateClustering(self, labels, sentences, numberOfCluster, topics):
        topicsDic = {key: [] for key in topics}
        topicClusterId = {key: {} for key in topics}
        mapTopicClusterId = {}
        for i in range(len(labels)):
            currentTopic = sentences[i]["tags"][0].split(">")[0].strip()
            topicsDic[currentTopic].append(labels[i])
        keys = topicsDic.keys()
        for key in keys:
            currentList = topicsDic[key]
            topicClusterId[key] = {i: currentList.count(i) for i in currentList}
        unusedKey = [i for i in range(self.NUM_CLUSTER)]
        for key in keys:
            condition = True
            while condition:
                print(unusedKey)
                goalKey = max(topicClusterId[key], key=topicClusterId[key].get)
                if goalKey in unusedKey:
                    mapTopicClusterId[goalKey] = key
                    unusedKey.remove(goalKey)
                    condition = False
                else:
                    topicClusterId[key].pop(goalKey, None)
                    if len(topicClusterId[key].keys()) == 0:
                        mapTopicClusterId[unusedKey[0]] = key
                        del unusedKey[0]
                        break
        labels = self.kMeansCluster.labels_
        c1 = 0
        c2 = 0
        with open('innovators1.csv', 'w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["link", "label", "topic"])
            for i in range(len(labels)):
                currentLink = self.sentences[i]["link"]
                currentTopic = self.sentences[i]["tags"][0].split(">")[0].strip()
                currentLabel = labels[i]
                print(currentLink)
                print(currentLabel)
                if currentTopic == mapTopicClusterId[currentLabel]:
                    c1 += 1
                else:
                    c2 += 1
                writer.writerow([currentLink, mapTopicClusterId[currentLabel], currentTopic])
        print(topicClusterId)
        print(topicsDic)
        print(mapTopicClusterId)
        print(c1)
        print(c2)

    @staticmethod
    def extractTopicsFromSentence(sentence):
        topic = []
        for sen in sentence:
            currentTopic = sen["tags"][0].split(">")[0].strip()
            if currentTopic not in topic:
                topic.append(currentTopic)
        return topic


k = K_means(14)
k.setSentencesFromFile("C:\\Users\\Bill\\PycharmProjects\\MIR_project\\Phase3\\data\\hamshahri.json")
k.cluster(K_means.WORD2VEC_MODE)
k.writeToFile()
k.evaluateClustering(k.kMeansCluster.labels_, k.sentences, k.NUM_CLUSTER, K_means.extractTopicsFromSentence(k.sentences))
