#Loading of SKT Pickles
from romtoslp import rom_slp
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

def SeeSentence(sentenceObj):
    print(sentenceObj.sentence)
    for chunk in sentenceObj.chunk:
        print("Analyzing ", rom_slp(chunk.chunk_name))
        for pos in chunk.chunk_words.keys():
            for word_sense in chunk.chunk_words[pos]:
                print(pos, ": ", rom_slp(word_sense.names), list(map(rom_slp, word_sense.lemmas)), word_sense.forms)