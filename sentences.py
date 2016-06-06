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

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
from wordTypeCheckFunction import *
import pickle

"""
SentencePreprocess:
-------------------
    Read a sentence obj and create + return the following objects 

    -> chunkDict: chunk_id -> position -> index in wordlist (nested dictionary)
    -> wordList: list of possible words as a result of word segmentation
    -> revMap2Chunk: Map word in wordlist to (cid, position) in chunkDict
    -> qu: Possible query nodes
"""
def SentencePreprocess(sentenceObj):
    """
    Considering word names only
    ***{Word forms or cngs can also be used}
    """
    chunkDict = {}
    wordList = []
    cngList = []
    revMap2Chunk = []
    qu = []

    cid = -1
    for chunk in sentenceObj.chunk:        
        cid = cid+1
        chunkDict[cid] = {}
        canBeQuery = 0
        if len(chunk.chunk_words.keys()) == 1:
            canBeQuery = 1 # Unsegmentable Chunk
        for pos in chunk.chunk_words.keys():
            chunkDict[cid][pos] = []
            if(canBeQuery == 1) and (len(chunk.chunk_words[pos]) == 1):
                canBeQuery = 2 # No cng alternative for the word
            for word_sense in chunk.chunk_words[pos]:
                if(len(word_sense.lemmas) > 0 and len(word_sense.forms) > 0):
                    wordList.append(rom_slp(word_sense.lemmas[0].split('_')[0]))
                    # print(word_sense.forms)
                    for thing in word_sense.forms:
                        cng = None
                        for form, config in thing.items():
                            if(form != 'verbform'):
                                if(type(config[0]) == list):
                                    cng = wordTypeCheck(form, config[0][0])
                                else:
                                    cng = wordTypeCheck(form, config[0])
                        if(cng != None):
                            cngList.append(cng)
                            break
                    
                    k = len(wordList) - 1
                    chunkDict[cid][pos].append(k)
                    revMap2Chunk.append((cid, pos))
                    if canBeQuery == 2:
                        # The word has a lemma available - in some pickle file it's not
                        # Make this word query
                        qu.append(k)
    # print(len(cngList))
    # print(len(wordList))
    verbs = []
    v2t = pickle.load(open('extras/verbs_vs_cngs_matrix_countonly.p', 'rb'), encoding=u'utf8')
    i = -1
    for w in wordList:
        i += 1
        if w in v2t:
            verbs.append(i)
    
    return (chunkDict, wordList, revMap2Chunk, qu, cngList, verbs)