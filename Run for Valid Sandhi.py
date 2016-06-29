
# coding: utf-8

# In[1]:

import pickle
from utilities import *
from DCS import *
from sentences import *


# In[2]:

goodDict = pickle.load(open('mergedGood_v3.p', 'rb'))


# In[3]:

# fList = list(goodDict.keys())


# In[4]:

# SeeDCS(dcsO)


# In[5]:

# skt, dcs  = loadSentence(fList[7], goodDict[fList[7]])
# SeeSentence(skt)


# In[ ]:




# In[6]:

fList = list(goodDict.keys())

def ValidateSandhi(fi = -1, fName = '', fPath='', verbose=False):
    
    if(fi >= 0):
        f = fList[fi]
        skt, dcs  = loadSentence(f, goodDict[f])
    else:
        f = fName
        skt, dcs  = loadSentence(fName, fPath)
    
    if skt==None:
        return 0
    if verbose:
        print()
        print('=='*20)
        print(f.upper())
#     SeeSentence(skt)
    (chunkDict, lemmaList, wordList, revMap2Chunk, qu, cngList, verbs, tuplesMain) = SentencePreprocess(skt)

    toSearch = []

    for i in range(len(dcs.lemmas)):
        lls = dcs.lemmas[i]
        for j in range(len(lls)):
            # (chunk, lemma, cng)
            toSearch.append((i, rom_slp(lls[j]), int(dcs.cng[i][j])))

    deactTuple = [False]*len(tuplesMain)
    for qtup in toSearch:
        if verbose:
            print('\n\n[QUERY]', qtup)
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
                            if(verbose):
                                print('[PAIR Match] chunk_%d, pos_%d, [%s], cng(%d)' % (qcid, pos, tup[2], tup[3]))
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
                                if(verbose):
                                    print('[LEMMA Match] chunk_%d, pos_%d, [%s], cng(%d)' % (qcid, pos, tup[2], tup[3]))
                                matchFound = True
                                deactTuple[i] = True
                                srch = (pos, i)
                                break
                    if(matchFound):
                        break
                if(matchFound):
                    break

        if matchFound:
            if verbose:
                print('[REMOVED] (Pos, Lemma, Names)')
#             print(srch)
            n1 = tuplesMain[srch[1]][0][1]
#             print(n1)
            p1 = srch[0]
            for pos in activeChunk.keys():
                if(pos == srch[0]):
                    for i in activeChunk[pos]:
                        # Remove all
                        deactTuple[i] = True
                        if verbose:
                            for t_rem in tuplesMain[i]:
                                print('(pos_%d, [%s], [%s]) ' % (pos, t_rem[2], t_rem[1]))
                else:
                    if(pos < p1):
                        for i in activeChunk[pos]:
                            # Deactivate Tuple
                            if not deactTuple[i]:
                                n2 = tuplesMain[i][0][2]
                                if not CanCoExist_sandhi(pos, p1, n2, n1):
                                    deactTuple[i] = True
                                    if(verbose):
                                        for t_rem in tuplesMain[i]:
                                            print('(pos_%d, [%s], [%s]) ' % (pos, t_rem[2], t_rem[1]))
                    else:
                        for i in activeChunk[pos]:
                            # Deactivate Tuple
                            if not deactTuple[i]:
                                n2 = tuplesMain[i][0][2]
                                if not CanCoExist_sandhi(p1, pos, n1, n2):
                                    deactTuple[i] = True
                                    if(verbose):
                                        for t_rem in tuplesMain[i]:
                                            print('(pos_%d, [%s], [%s]) ' % (pos, t_rem[2], t_rem[1]))
        else:
            if verbose:
                print('___[NOT FOUND]___')
    if verbose:
        print('\nWords remaining in skt[%s]:' % f, len(tuplesMain) - sum(deactTuple))
    return len(tuplesMain) - sum(deactTuple)


# In[7]:

# ValidateSandhi(fName = '305755.p', fPath= '../TextSegmentation/Pickle_Files/305755.p', verbose=True)


# In[9]:

got = 0
for i in range(3000):
    r = ValidateSandhi(i)
    if r > 0:
        got += 1
        ValidateSandhi(i, verbose=True)
    if got >= 100:
        break
print('-'*50, '\nGot', got, 'problems')


# In[ ]:



