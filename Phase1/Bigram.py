import pickle
from Phase1 import Constants


class Bigram:
    bigram_set = {}

    @classmethod
    def load_file(cls):
        try:
            with open(Constants.bigram_file_path, 'rb') as f:
                cls.bigram_set = pickle.load(f)
        except FileNotFoundError:
            with open(Constants.bigram_file_path, 'wb') as f:
                pickle.dump({}, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def add_term_to_bigram(cls, term):
        for i in range(len(term) - 1):
            bi = term[i:i + 2]
            if bi in cls.bigram_set:
                if term not in cls.bigram_set[bi]:
                    cls.bigram_set[bi].append(term)
            else:
                cls.bigram_set[bi] = [term]

    @classmethod
    def delete_term_from_bigram(cls, term):
        for i in range(len(term) - 1):
            bi = term[i:i + 2]
            if bi in cls.bigram_set:
                if term in cls.bigram_set[bi]:
                    cls.bigram_set[bi].remove(term)
                    if len(cls.bigram_set[bi]) == 0:
                        cls.bigram_set.pop(bi)

    @classmethod
    def save_bigram(cls):
        with open(Constants.bigram_file_path, 'wb') as f:
            pickle.dump(cls.bigram_set, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def get_terms_of_bigram(cls, bi):
        if bi in cls.bigram_set:
            return cls.bigram_set[bi]
        return []

    @classmethod
    def jaccard_measure(cls, base_bigrams: set, term_bigrams: set):
        if len(base_bigrams) == 0 or len(term_bigrams) == 0:
            return 0
        intersection = 0
        for bi in term_bigrams:
            intersection += int(bi in base_bigrams)
        return intersection / (len(base_bigrams) + len(term_bigrams) - intersection)

    @classmethod
    def nearest_terms_wrt_jaccard_measure(cls, term, threshold=0.2):
        set_of_terms = set()
        for i in range(len(term) - 1):
            bi = term[i:i + 2]
            for t in cls.get_terms_of_bigram(bi):
                set_of_terms.add(t)

        res = []
        base_bigrams = set([term[i:i + 2] for i in range(len(term) - 1)])
        for t in set_of_terms:
            t_bigrams = set([t[i:i + 2] for i in range(len(t) - 1)])
            if cls.jaccard_measure(base_bigrams, t_bigrams) >= threshold:
                res.append(t)
        return res

    @classmethod
    def edit_distance_measure(cls, term1, term2):
        # Levenshtein distance algorithm
        dp = [[i + j for i in range(len(term2) + 1)] for j in range(len(term1) + 1)]
        for i in range(len(term1)):
            for j in range(len(term2)):
                replace = dp[i][j] + int(term1[i] != term2[j])
                delete = dp[i][j + 1] + 1
                insert = dp[i + 1][j] + 1
                dp[i + 1][j + 1] = min(replace, delete, insert)
        return dp[-1][-1]

    @classmethod
    def get_best_alternative(cls, term):
        nearest_terms = cls.nearest_terms_wrt_jaccard_measure(term)
        if len(nearest_terms) == 0:
            return term
        distances = [0 for _ in range(len(nearest_terms))]
        for i in range(len(nearest_terms)):
            distances[i] = cls.edit_distance_measure(term, nearest_terms[i])

        min_distance = min(distances)
        best = []
        for i in range(len(nearest_terms)):
            if distances[i] == min_distance:
                best.append(nearest_terms[i])

        if len(best) == 1:
            return best[0]
        return best
