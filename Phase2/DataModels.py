class Document:
    def __init__(self, docid, title, raw_text, tokens, is_test: bool, view: int):
        self.docid = docid
        self.title = title
        self.raw_text = raw_text
        self.tokens = tokens
        self.is_test = is_test
        self.view = view


class kNNData:
    def __init__(self, tokens: list, label):
        self.label = label
        self.vector = dict()
        for term in tokens:
            if self.vector.get(term) is None:
                self.vector[term] = 1
            else:
                self.vector[term] += 1

    def __str__(self):
        return str(self.label) + " : " + str(self.vector)
