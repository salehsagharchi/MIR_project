import math


class Score:
    def __init__(self):
        self.id = 0
        self.baseLog = 10

    def query(self, terms):
        n = self.getN()
        vectors = [[0 for i in range(len(terms))] for j in range(n)]
        for i in range(len(terms)):
            df, documentIndex, frequencyTerm = self.collectInformationAboutTerm(terms[i])
            for j in range(len(documentIndex)):
                vectors[documentIndex[j]][i] = (1 + math.log(frequencyTerm[j], self.baseLog)) * math.log(n / df,
                                                                                                         self.baseLog)
        for i in range(len(terms)):
            vectors[i] = self.normalizeVector(vectors[i])
        scores = self.calculateScore(self.createQueryVector(terms), vectors)
        sorted(scores, key=lambda x : x[1])
        result = []
        for i in range(len(scores)):
            result.append(scores[i][0])
        return result

    def calculateScore(self, qVector, dVectors):
        result = [0, 0] * len(dVectors)
        for i in range(dVectors):
            result[i] = [i, self.dotVectors(qVector, dVectors[i])]
        return result

    def dotVectors(self, firstVector, secondVector):
        result = 0
        for i in range(len(firstVector)):
            result = firstVector[i] * secondVector[i]
        return result

    def normalizeVector(self, vector):
        sum = 0
        for i in range(len(vector)):
            sum += math.pow(vector[i], 2)
        if sum == 0:
            return vector
        normal = 1 / math.sqrt(sum)
        for i in range(len(vector)):
            vector[i] = vector[i] * normal
        return vector

    def createQueryVector(self, terms):
        vector = [0] * len(terms)
        for i in range(terms):
            tf = self.findFrequency(terms, terms[i])
            vector[i] = 1 + math.log(tf, self.baseLog)
        return self.normalizeVector(vector)

    def findFrequency(self, terms, term):
        count = 0
        for i in range(len(terms)):
            if term == terms[i]:
                count += 1
        return count

    def getN(self):
        #TODO
        return 10
    def collectInformationAboutTerm(self, term):
        #TODO
        return 4, [], []
        #retunr df, documentIndex, frequencyTerm

