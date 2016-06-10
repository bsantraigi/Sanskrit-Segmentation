#Loading of SKT Pickles
from romtoslp import rom_slp
from json import *
class word_new():
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
                print(pos, ": ", rom_slp(word_sense.names), list(map(rom_slp,word_sense.lemmas)), word_sense.forms)

def getWord(sentenceObj, cid, pos,kii):
    ch = sentenceObj.chunk[cid]
    word = ch.chunk_words[pos][kii]
    return {'lemmas': word.lemmas, 'forms':word.forms, 'names':word.names}

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

    -> chunkDict: chunk_id -> position -> index in lemmaList (nested dictionary)
    -> lemmaList: list of possible words as a result of word segmentation
    -> revMap2Chunk: Map word in wordlist to (cid, position) in chunkDict
    -> qu: Possible query nodes
"""
v2t = pickle.load(open('extras/verbs_vs_cngs_matrix_countonly.p', 'rb'), encoding=u'utf8')
def wtc_recursive(form, c):
    if type(c) ==list:
        for cc in c:
            return wtc_recursive(form, cc)
    else:
        return wordTypeCheck(form, c)

def SentencePreprocess(sentenceObj):
    """
    Considering word names only
    ***{Word forms or cngs can also be used}
    """
    def getCNGs(formsDict):
        l = []
        for form, configs in formsDict.items():
            for c in configs:
                if(form == 'verbform'):                
                    continue
                else:
                    l.append(wtc_recursive(form, configs))
        return list(set(l))

    chunkDict = {}
    lemmaList = []
    wordList = []
    cngList = []
    revMap2Chunk = []
    qu = []
    tuplesMain = []

    cid = -1
    tidExclusive = 0
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
                nama = rom_slp(word_sense.names)
                if(len(word_sense.lemmas) > 0 and len(word_sense.forms) > 0):
                    tuples = []
                    for lemmaI in range(len(word_sense.lemmas)):
                        lemma = rom_slp(word_sense.lemmas[lemmaI].split('_')[0])
                        tempCNGs = getCNGs(word_sense.forms[lemmaI])
                        for cng in tempCNGs:
                            # UPDATE LISTS
                            tuples.append((tidExclusive, nama, lemma, cng)) # Remember the order
                            lemmaList.append(lemma)
                            wordList.append(nama)
                            cngList.append(cng)

                            revMap2Chunk.append((cid, pos, len(tuplesMain)))
                            tidExclusive += 1
                    
                    k = len(tuplesMain)
                    chunkDict[cid][pos].append(k)
                    tuplesMain.append(tuples)
                    if canBeQuery == 2:
                        # The word has a lemma available - in some pickle file it's not
                        # Make this word query
                        # print("Query: ", len(tuples))
                        qu.append(tuples[0][0])

    verbs = []
    i = -1
    for w in lemmaList:
        i += 1
        if w in list(v2t.keys()):
            verbs.append(i)


    # print(tuplesMain)
    
    return (chunkDict, lemmaList, wordList, revMap2Chunk, qu, cngList, verbs, tuplesMain)