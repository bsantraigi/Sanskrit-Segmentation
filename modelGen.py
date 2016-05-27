from gensim.models.word2vec import Word2Vec
#from DCS import DCS
from utilities import printProgress, validatePickleName, pickleFixLoad
import pprint
from progressbar import printProgress
from romtoslp import rom_slp


import re
def isPickle(name):
    m = re.search("^[\w]+.p$", name)
    if(m != None):
        return True
    return False

isPickle('123.p')

class DCS:
	def __init__(self,sent_id,sentence):
		self.sent_id=sent_id
		self.sentence=sentence
		self.dcs_chunks=[]
		self.lemmas=[]
		self.cng=[]

import pickle
from os import listdir
from os.path import isfile, join
picklePath = "../Text Segmentation/DCS_pick/"
# pickleFiles = [f for f in listdir(picklePath) if (isfile(join(picklePath, f)) and isPickle(f))]
# len(pickleFiles)
pickleFiles = pickle.load(open("./dcsFileList.p", "rb"))
print("Total File Count: ",len(pickleFiles))

badPickles = []
class SKTSentences(object):
    def __init__(self, filePath, fileList):
        self.filePath = filePath
        self.fileList = fileList
        self.count = 0
        self.size = 6*len(fileList)
        
    def __iter__(self):
        printProgress (self.count, self.size, prefix = 'Progress', suffix = 'Complete', decimals = 2, barLength = 50)
        for f in self.fileList:
            if f not in badPickles:
                try:
                    dcsInstance = pickleFixLoad(self.filePath + f)
                except (KeyError, EOFError) as e:
                    badPickles.append(f)
                    #print(f)
                    yield [""]
                l = [rom_slp(item) for sublist in dcsInstance.lemmas for item in sublist]
                self.count = (self.count + 1)
                printProgress (self.count, self.size, prefix = 'Progress', suffix = 'Complete', decimals = 2, barLength = 50)
                yield l
            else:
                yield [""]


sentences = SKTSentences(picklePath, pickleFiles)
model = Word2Vec(sentences, size=100, window=10, workers=20)
model.save('model_100_10.p')
print(model)




