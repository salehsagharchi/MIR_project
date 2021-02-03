# {
# 	"id": "Unique ID of the paper",
# 	"title": "Title of the paper",
# 	"abstract": "Abstract of the paper",
# 	"date": "Publication year",
# 	"authors": ["Name of the first author", ..., "Name of the last author"],
# 	"references": ["ID of the first reference", ..., "ID of the tenth reference"]
# }

class Paper:
    def __init__(self, paper_id, title, abstract, date, authors, references):
        self.id = paper_id
        self.title = title
        self.abstract = abstract
        self.date = date
        self.authors = authors
        self.references = references
