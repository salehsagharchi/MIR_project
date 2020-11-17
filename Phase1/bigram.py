import pickle


class Bigram:
    bigram_set = {}
    FILE_NAME = 'bigrams'

    @classmethod
    def load_file(cls):
        try:
            with open(cls.FILE_NAME + '.pkl', 'rb') as f:
                cls.bigram_set = pickle.load(f)
                f.close()
        except FileNotFoundError:
            with open(cls.FILE_NAME + '.pkl', 'wb') as f:
                pickle.dump({}, f, pickle.HIGHEST_PROTOCOL)
                f.close()

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
    def save_to_file(cls):
        with open(cls.FILE_NAME + '.pkl', 'wb') as f:
            pickle.dump(cls.bigram_set, f, pickle.HIGHEST_PROTOCOL)
            f.close()

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
    def nearest_terms_wrt_jaccard_measure(cls, term, threshold=0.5):
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
            return None
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

    # @classmethod
    # def get_nearest_terms(cls, term, distance=3):
    #     set_of_terms = {}
    #     for i in range(len(term) - 1):
    #         bi = term[i:i + 2]
    #         for t in cls.get_terms_of_bigram(bi):
    #             if t in set_of_terms:
    #                 set_of_terms[t] += 1
    #             else:
    #                 set_of_terms[t] = 1
    #     res = []
    #     for t in set_of_terms:
    #         if set_of_terms[t] >= (len(term) - 1 - distance):
    #             res.append(t)
    #     return res


# if __name__ == '__main__':
#     pass
#     Bigram.load_file()
#
    # str = """Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the
    # industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled
    # it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic
    # typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets
    # containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker
    # including versions of Lorem Ipsum. It is a long established fact that a reader will be distracted by the readable
    # content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal
    # distribution of letters, as opposed to using 'Content here, content here', making it look like readable English.
    # Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text,
    # and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved
    # over the years, sometimes by accident, sometimes on purpose (injected humour and the like). """.lower()
#     for t in str.split(' '):
#         Bigram.add_term_to_bigram(t)
#     Bigram.save_to_file()

