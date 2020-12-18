class Document:
    def __init__(self, docid, title, raw_text, tokens, is_test: bool, view: int):
        self.docid = docid
        self.title = title
        self.raw_text = raw_text
        self.tokens = tokens
        self.is_test = is_test
        self.view = view


class kNNData:
    def __init__(self, vector: list, label):
        self.vector = vector
        self.label = label
