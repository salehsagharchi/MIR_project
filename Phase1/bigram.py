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
    def add_word_to_bigram(cls, word):
        for i in range(len(word) - 1):
            bi = word[i:i + 2]
            if bi in cls.bigram_set:
                if word not in cls.bigram_set[bi]:
                    cls.bigram_set[bi].append(word)
            else:
                cls.bigram_set[bi] = [word]

    @classmethod
    def delete_word_from_bigram(cls, word):
        for i in range(len(word) - 1):
            bi = word[i:i + 2]
            if bi in cls.bigram_set:
                if word in cls.bigram_set[bi]:
                    cls.bigram_set[bi].remove(word)
                    if len(cls.bigram_set[bi]) == 0:
                        cls.bigram_set.pop(bi)

    @classmethod
    def save_to_file(cls):
        with open(cls.FILE_NAME + '.pkl', 'wb') as f:
            pickle.dump(cls.bigram_set, f, pickle.HIGHEST_PROTOCOL)
            f.close()

    @classmethod
    def get_words_with_bigram(cls, bi):
        if bi in cls.bigram_set:
            return cls.bigram_set[bi]
        return []

