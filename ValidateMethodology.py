
# coding: utf-8

# In[1]:

import pickle
from utilities import *
from DCS import *
from sentences import *


goodDict = pickle.load(open('mergedGood_v3.p', 'rb'))


fList = list(goodDict.keys())

def ValidateSandhi(fi):
    
    f = fList[fi]
    skt, dcs  = loadSentence(f, goodDict[f])
    
    if skt==None:
        return
    print()
    print('=='*20)
    print(f.upper())
#     SeeSentence(skt)
    (chunkDict, lemmaList, wordList, revMap2Chunk, qu, cngList, verbs, tuplesMain) = SentencePreprocess(skt)

    toSearch = []

    for i in range(len(dcs.lemmas)):
        lls = dcs.lemmas[i]
        for j in range(len(lls)):
            toSearch.append((i, rom_slp(lls[j]), int(dcs.cng[i][j])))

    deactTuple = [False]*len(tuplesMain)
    for qtup in toSearch:
        print('[QUERY]', qtup)
        qcid = qtup[0]
        qlem = qtup[1]
        qcng = qtup[2]
        activeChunk = chunkDict[qcid]
        matchFound = False

        for pos in activeChunk.keys():
            for i in activeChunk[pos]:
                if not deactTuple[i]:
                    for tup in tuplesMain[i]:
        #                 print(tup)
                        if (tup[2] == qtup[1]) and (tup[3] == qtup[2]):
                            print('Pair Match:', tup)
                            matchFound = True
                            deactTuple[i] = True
                            srch = (pos, i)
                            break
                if(matchFound):
                    break
            if(matchFound):
                break

        if not matchFound:
            for pos in activeChunk.keys():
                for i in activeChunk[pos]:
                    if not deactTuple[i]:
                        for tup in tuplesMain[i]:
                            if tup[2] == qtup[1]:
                                print('Lemma Match:', tup)
                                matchFound = True
                                deactTuple[i] = True
                                srch = (pos, i)
                                break
                    if(matchFound):
                        break
                if(matchFound):
                    break

        if matchFound:
#             print(srch)
            n1 = tuplesMain[srch[1]][0][1]
#             print(n1)
            p1 = srch[0]
            for pos in activeChunk.keys():
                if(pos == srch[0]):
                    for i in activeChunk[pos]:
                        # Remove all
                        deactTuple[i] = True
                else:
                    if(pos < p1):
                        for i in activeChunk[pos]:
                            # Deactivate Tuple
                            if not deactTuple[i]:
                                n2 = tuplesMain[i][0][2]
                                if not CanCoExist_sandhi(pos, p1, n2, n1):
                                    deactTuple[i] = True
                                    print('[REMOVED]:', tuplesMain[i])
                    else:
                        for i in activeChunk[pos]:
                            # Deactivate Tuple
                            if not deactTuple[i]:
                                n2 = tuplesMain[i][0][2]
                                if not CanCoExist_sandhi(p1, pos, n1, n2):
                                    deactTuple[i] = True
                                    print('[REMOVED]:', tuplesMain[i])

    print('Words remaining in skt[%s]:' % f, len(tuplesMain) - sum(deactTuple))

for i in range(100):
    ValidateSandhi(i)




