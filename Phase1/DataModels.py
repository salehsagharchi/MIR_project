class Document:
    def __init__(self, docid, title, normalized_title, raw_text, tokens, lang):
        self.docid = docid
        self.title = title
        self.normalized_title = normalized_title
        self.raw_text = raw_text
        self.tokens = tokens
        self.lang = lang
