import pickle
from utilities import *
from DCS import *
from sentences import *
import csv



goodDict = pickle.load(open('mergedGood_v4.p', 'rb'))


fList = list(goodDict.keys())

def HaveSolution(fi, nbcsv):
    
    f = fList[fi]
    skt, dcs  = loadSentence(f, goodDict[f])
    
    if skt==None:
        return
#     print()
#     print('=='*20)
#     print(f.upper())
#     SeeSentence(skt)
    (chunkDict, lemmaList, wordList, revMap2Chunk, qu, cngList, verbs, tuplesMain) = SentencePreprocess(skt)

    toSearch = []
    
    if len(chunkDict) != len(dcs.lemmas):
        nbcsv.writerow([f, 'Length Mismatch', ' ', ' ', ' ', '', ''])
        return
        
    
    arr_lemm_dcs = []
    arr_cng_dcs = []
    
    arr_lemm_skt = []
    arr_cng_skt = []

    for i in range(len(dcs.lemmas)):
        lls = dcs.lemmas[i]
        for j in range(len(lls)):
            toSearch.append((i, rom_slp(lls[j]), int(dcs.cng[i][j])))
            arr_lemm_dcs.append(rom_slp(lls[j]))
            arr_cng_dcs.append(int(dcs.cng[i][j]))

    presence = [0]*len(toSearch)
    
    
    for qi in range(len(toSearch)):
        qtup = toSearch[qi]
#         print('[QUERY]', qtup)
        qcid = qtup[0]
        qlem = qtup[1]
        qcng = qtup[2]
        activeChunk = chunkDict[qcid]
        matchFound = False

        for pos in activeChunk.keys():
            for i in activeChunk[pos]:
                for tup in tuplesMain[i]:
    #                 print(tup)
                    if (tup[2] == qtup[1]) and (tup[3] == qtup[2]):
#                         print('Pair Match:', tup)
                        matchFound = True
                        arr_lemm_skt.append(tup[2])
                        arr_cng_skt.append(tup[3])
                        presence[qi] = 1
                        break
                if(matchFound):
                    break
            if(matchFound):
                break

        if not matchFound:
            for pos in activeChunk.keys():
                for i in activeChunk[pos]:
                    for tup in tuplesMain[i]:
                        if tup[2] == qtup[1]:
#                             print('Lemma Match:', tup)
                            matchFound = True
                            presence[qi] = -1
                            arr_lemm_skt.append(tup[2])
                            arr_cng_skt.append(tup[3])
                            break
                    if(matchFound):
                        break
                if(matchFound):
                    break
                    
    status = 'Bad' if 0 in presence else 'Good'
    nbcsv.writerow([f, presence, ' '.join(arr_lemm_dcs), arr_cng_dcs, ' '.join(arr_lemm_skt), arr_cng_skt, status])


# In[10]:

with open('NewBads.csv', 'w') as bfh:
    nbcsv = csv.writer(bfh)
    nbcsv.writerow(['File', 'presence', 'DCS_lemm', 'DCS_cng', 'SKT_lemm', 'SKT_cng', 'status'])
#     for i in range(len(goodDict)):
    for i in range(len(fList)):
        if (i % 1000 == 0):
            print('Checkpoint:', i)
        HaveSolution(i, nbcsv)
        

len(fList)


# In[ ]:



