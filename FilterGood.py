from DCS import *
from sentences import *
import os
from utilities import *
import re
from romtoslp import rom_slp
import pickle
import datetime

def getName():
	dt = datetime.datetime.now()
	return('good/good_' + str(dt.hour) + str(dt.minute) + str(dt.second) + '.p')
	

dcsPath = '../Text Segmentation/DCS_pick/'
dcsList = os.listdir(dcsPath)
dcsList.sort()

try:
	sktPath_1 = sys.argv[3]
except IndexError:
	sktPath_1 = '../TextSegmentation/corrected_10to20/'
sktList_1 = os.listdir(sktPath_1)
sktList_1.sort()

preList = pickle.load(open('pvb.p', 'rb'))
def removePrefix(preList, lemma):
    for pre in preList:
        m = re.match(pre, lemma)
        if(m != None):
            return m.span()[1]
def pickleFixLoad(filename):
    return pickle.load(open(filename, 'rb'), encoding=u'utf8')

##------------------------------------------------------------------

def CanIUseIt(sntcObj, dcsObj):
#     print('-'*15)
#     CHECK 1 -> NUMBER OF CHUNKS/LEMMAS
    s = re.findall("[^ ]+", dcsObj.sentence)
    if((len(s) != len(dcsObj.lemmas)) or (len(s) != len(sntcObj.chunk))):
        return 1
    
#     CHECK 2 -> ALL LEMMAS PRESENT IN CORRESPONDING CHUNK
    sntc_lemma_packs = []
    for chunk in sntcObj.chunk:
        allwords = set()
        for pos, word_senses in chunk.chunk_words.items():
            for sense in word_senses:
                for lemma in sense.lemmas:
                    term = rom_slp(lemma.split('_')[0])
                    allwords.add(term)
#                     allwords.update([term, removePrefix(preList, term)])
                    
        sntc_lemma_packs.append(allwords)
        
    i = -1
    for chunk in dcsObj.lemmas:
        chunk = list(map(rom_slp, chunk))
        i += 1
#         print(" | ".join(chunk))
#         print("vs")
#         print(sntc_lemma_packs[i])
#         print()
        for lemma in chunk:
            if(lemma not in sntc_lemma_packs[i]):
                if lemma[removePrefix(preList, lemma)::] not in sntc_lemma_packs[i]:
#                     print(dcsObj.sentence)
                    return 2
#                     print('HUH')
    return 0
    
#     CHECK 3 -> NO CNG ERRORS ('VERBFORMS')


common = []
bads = []
count = 0
# print(len(sys.argv))
try:
	start = int(sys.argv[1])
	finish = int(sys.argv[2])
except IndexError:
	start = 0
	finish = 100
for dcsFile in dcsList[start:finish]:
    if(validatePickleName(dcsFile) != ""):
        if dcsFile in sktList_1:
            count += 1
            if(count % 100 == 0):
                print('Checkpoint: ', len(common), len(bads))                
                pass
            common.append(dcsFile)
            try:
                dcsObj = pickleFixLoad(dcsPath + dcsFile)
                sntcObj = pickleFixLoad(sktPath_1 + dcsFile)
            except (EOFError, pickle.UnpicklingError) as e:
                continue
            e = CanIUseIt(sntcObj, dcsObj)
            # IF zero returned then all test passed
            if e != 0:
                bads.append(dcsFile)
            else:
#                 print(dcsFile)
                pass
            if e == 2:
#                 print(dcsFile)
#                 print("FAIL 2")
                pass
                

good = set(common) - set(bads)
print(len(good))

## ----------------------------------------------------------------------------

pickle.dump(good, open(getName(), 'wb'))
