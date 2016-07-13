from DCS import *
from sentences import *
import os
from utilities import *
import re
from romtoslp import rom_slp
import pickle
import sys

goodDict = pickle.load(open('mergedGood.p', 'rb'))
badSet = set()
def checkBad(sentenceObj):
    for chunk in sentenceObj.chunk:
        for pos in chunk.chunk_words.keys():
            for word_sense in chunk.chunk_words[pos]:
                if(len(word_sense.lemmas) > 0 and len(word_sense.forms) == 0):
                    return True
    return False
# print(int(sys.argv[1]), int(sys.argv[2]))

count = 0
for fName in list(goodDict.keys()):
	count += 1
	if count % 200 == 0:
		print('Checkpoint: ', count)
		
	try:
		dcsObj = pickleFixLoad('../Text Segmentation/DCS_pick/' + fName)           
		if goodDict[fName] == "C1020" :
		    sentenceObj = pickleFixLoad('../TextSegmentation/corrected_10to20/' + fName)
		else:
		    sentenceObj = pickleFixLoad('../TextSegmentation/Pickle_Files/' + fName)
		if(checkBad(sentenceObj)):
		    badSet.add(fName)
		        
	except (EOFError, KeyError) as e:
		pass
print(len(badSet))

pickle.dump(badSet, open('temp/badSet.p', 'wb'))

print("Found all bad guys")
badSet = pickle.load(open('temp/badSet.p', 'rb'))
for f in badSet:
	goodDict.pop(f, None)
pickle.dump(goodDict, open('mergedGood_v2.p', 'wb'))
