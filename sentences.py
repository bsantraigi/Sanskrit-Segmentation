#Loading of SKT Pickles
class word_new:
    def __init__(self,names):
        self.lemmas=[]
        self.names=names
        self.urls=[]
        self.forms=[]

class chunks:
    def __init__(self,chunk_name):
        self.chunk_name=chunk_name
        self.chunk_words={}

class sentences:
    def __init__(self,sent_id,sentence):
        self.sent_id=sent_id
        self.sentence=sentence
        self.chunk=[]
