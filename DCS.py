import sys
import warnings
class DCS:
    def __init__(self,sent_id,sentence):
        self.sent_id=sent_id
        self.sentence=sentence
        self.dcs_chunks=[]
        self.lemmas=[]
        self.cng=[]

def SeeDCS(dcsObj):
    print(dcsObj.sentence)
    print(dcsObj.lemmas)
    print(dcsObj.cng)