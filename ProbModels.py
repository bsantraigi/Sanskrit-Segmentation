import pickle
from utilities import printProgress, validatePickleName, pickleFixLoad
import numpy as np
import math
import pickle
from wordTypeCheckFunction import *
from collections import defaultdict
import pprint

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sandhiRules = pickle.load(open('extras/sandhiRules.p','rb'))    
def CanCoExist_sandhi(p1, p2, name1, name2):
    # P1 must be less than P2
    # Just send it in the proper order
    if(p1 < p2):
        overlap = max((p1 + len(name1)) - p2, 0)
        if overlap == 0:
            return True
        if overlap == 1 or overlap == 2:
            p1 = (name1[len(name1) - overlap:len(name1):], name2[0])
            p2 = (name1[-1], name2[0:overlap:])
            # print(name1, name2, p1, p2)
            # print(p1, p2)
            if p1 in sandhiRules:
                if(sandhiRules[p1]['length'] < len(p1[0]) + len(p1[1])):
                    return True
            if p2 in sandhiRules:
                if(sandhiRules[p2]['length'] < len(p2[0]) + len(p2[1])):
                    return True

    return False

"""
These models are count based probabilistic model
created using the DCS_Pick corpus

-----------------------------------

Uses KN smoothing
"""
class ProbModels():
    def __init__(self, **kwargs):
        kwargs = defaultdict(lambda: None, kwargs)        
        
        """WORD2WORD CO-OCCURENCE DATA"""

        fullCo_oc_mat = kwargs['fullCo_oc_mat']
        unigram_counts = kwargs['unigram_counts']        

        context_count = defaultdict(int)
        for word in fullCo_oc_mat.keys():
            context_count[word] = len(fullCo_oc_mat[word])

        # Each bigram is repeated; a-b is same as b-a
        total_context = int(sum(context_count.values())/2)

        total_co_oc = sum(
            [sum(fullCo_oc_mat[word].values()) for word in fullCo_oc_mat.keys()])
        
        self.fullCo_oc_mat = fullCo_oc_mat
        self.unigram_counts = unigram_counts
        self.context_count = context_count
        self.total_context = total_context
        self.total_co_oc = total_co_oc

        """TYPE2TYPE DATA"""

        cng2cngFullMat = kwargs['cng2cngFullMat']
        cng2index_dict = kwargs['cng2index_dict']

        t2t_context_count = np.sum(cng2cngFullMat > 0, axis = 1) # Row-wise sum
        t2t_total_co_oc = int(np.sum(cng2cngFullMat)/2)
        t2t_total_contexts = np.sum(t2t_context_count)

        self.cng2cngFullMat = cng2cngFullMat
        self.cng2index_dict = cng2index_dict

        self.t2t_context_count = t2t_context_count
        self.t2t_total_contexts = t2t_total_contexts
        self.t2t_total_co_oc = t2t_total_co_oc


        """VERB2TYPE DATA"""
        self.v2c_fullMat = kwargs['v2c_fullMat']


        """W2W_SAME_CNG DATA"""
        w2w_samecng_fullmat = kwargs['w2w_samecng_fullmat']
        samecng_unigram_counts = kwargs['samecng_unigram_counts']

        samecng_context_count = defaultdict(int)
        for word in w2w_samecng_fullmat.keys():
            samecng_context_count[word] = len(w2w_samecng_fullmat[word])

        # Each bigram is repeated; a-b is same as b-a
        samecng_total_context = int(sum(samecng_context_count.values())/2)
        
        samecng_total_co_oc = sum(
            [sum(w2w_samecng_fullmat[word].values()) for word in w2w_samecng_fullmat.keys()])
        
        self.w2w_samecng_fullmat = w2w_samecng_fullmat
        self.samecng_unigram_counts = samecng_unigram_counts
        self.samecng_context_count = samecng_context_count
        self.samecng_total_context = samecng_total_context
        self.samecng_total_co_oc = samecng_total_co_oc
        return

    def RemoveCompetingEdges(self, TransitionMat, tuplesMain, chunkDict):
        lastTuple = tuplesMain[len(tuplesMain) - 1]
        nodeCount = lastTuple[len(lastTuple) - 1][0] + 1
        wordList = ['']*nodeCount
        for i in range(0, len(tuplesMain)):
            # print(tuplesMain[i])
            for tup in tuplesMain[i]:
                wordList[tup[0]] = tup[1]
        # REMOVE EDGES FROM COMPETETING NODES
        # IN THE SAME CHUNK
        cDict2 = {}
        for cid in chunkDict.keys():
            chunk = chunkDict[cid]
            cDict2[cid] = {}
            for pos in chunk.keys():
                wids = []
                for zz in chunk[pos]:
                    for tup in tuplesMain[zz]:
                        wids.append(tup[0])
                cDict2[cid][pos]=wids

        for cid in chunkDict.keys():
            chunk = chunkDict[cid]
            for pos in chunk.keys():
                wids = chunk[pos]
                # Remove edge b/w nodes at same location
                for u in range(len(wids) - 1):
                    for v in range(u + 1, len(wids)):
                        # print('Remvoe b/w', wordList[wids[u]], wordList[wids[v]])
                        TransitionMat[wids[u], wids[v]] = 0
                        TransitionMat[wids[v], wids[u]] = 0
                # Remove edge b/w competing nodes from diff location
                for _pos in chunk.keys():
                    wids2 = chunk[_pos]
                    if(pos < _pos):
                        for wi1 in wids:
                            for wi2 in wids2:
                                name1 = wordList[wi1]
                                name2 = wordList[wi2]
                                if not CanCoExist_sandhi(pos, _pos, name1, name2):
                                    # print('Remvoe b/w', name1, name2)
                                    TransitionMat[wi1, wi2] = 0

                    elif(_pos < pos):
                        for wi1 in wids:
                            for wi2 in wids2:
                                name1 = wordList[wi1]
                                name2 = wordList[wi2]
                                if not CanCoExist_sandhi(_pos, pos, name2, name1):
                                    # print('Remvoe b/w', name2, name1)
                                    TransitionMat[wi1, wi2] = 0                                        
        
    def get_cng2cng_mat(self, tuplesMain, chunkDict, kn_smooth = True):
        # pprint.pprint(tuplesMain)
        # pprint.pprint(chunkDict)

        lastTuple = tuplesMain[len(tuplesMain) - 1]
        nodeCount = lastTuple[len(lastTuple) - 1][0] + 1

        TransitionMat = np.zeros((nodeCount, nodeCount))
        if kn_smooth:
            for i in range(0, len(tuplesMain) - 1):
                for j in range(i + 1, len(tuplesMain)):
                    tSet1 = tuplesMain[i]
                    tSet2 = tuplesMain[j]
                    for tup1 in tSet1:
                        for tup2 in tSet2:
                            row = tup1[0]
                            col = tup2[0]
                            # row != col, always
                            TransitionMat[row][col] = self.kn_cng2cng(tup1[3], tup2[3])
                            TransitionMat[col][row] = self.kn_cng2cng(tup2[3], tup1[3])

            self.RemoveCompetingEdges(TransitionMat, tuplesMain, chunkDict)
            for row in range(nodeCount):
                row_sum = np.sum(TransitionMat[row, :])
                if(row_sum == 0):
                    print("Report ROW SUM ZERO CNG2CNG")
                TransitionMat[row, :] /= row_sum
                TransitionMat[row, row] = 0
        else:
            # FIXME: DOESN'T SUPPORT TUPLESMAIN
            cng2cngFullMat = self.cng2cngFullMat
            cng2index_dict = self.cng2index_dict
            cngList = []
            for tupleSet in tuplesMain:
                for tup in tupleSet:
                    cngList.append(tup[3])
            cngIndexList = []
            for cng in cngList:
                try:
                    ci = cng2index_dict[str(cng)]
                    cngIndexList.append(ci)
                except:
                    cngIndexList.append(None)
            for row in range(nodeCount):
                for col in range(nodeCount):
                    if row != col:
                        try:
                            # print(cngIndexList[row])
                            TransitionMat[row][col] = cng2cngFullMat[cngIndexList[row],cngIndexList[col]]
                        except KeyError:
                            TransitionMat[row][col] = 0 #WHAT TO DO HERE??
                    else:
                        TransitionMat[row][col] = 0
                
                row_sum = np.sum(TransitionMat[row, :])
                if(row_sum > 0):
                    TransitionMat[row, :] /= row_sum
                else:
                    TransitionMat[row, :] = 1/(nodeCount - 1)
                    pass
                
                TransitionMat[row, row] = 0
            
            # print((TransitionMat[row, :]))
        # MakeRowStochastic(TransitionMat)
        return TransitionMat

    

    def get_w2w_mat(self, tuplesMain, chunkDict, kn_smooth = True):
        lastTuple = tuplesMain[len(tuplesMain) - 1]
        nodeCount = lastTuple[len(lastTuple) - 1][0] + 1
        
        TransitionMat = np.zeros((nodeCount, nodeCount))
        if kn_smooth:
            for i in range(0, len(tuplesMain) - 1):
                for j in range(i + 1, len(tuplesMain)):
                    tSet1 = tuplesMain[i]
                    tSet2 = tuplesMain[j]
                    for tup1 in tSet1:
                        for tup2 in tSet2:
                            row = tup1[0]
                            col = tup2[0]
                            # row != col, always
                            TransitionMat[row][col] = self.kn_word2word(tup1[2], tup2[2])
                            TransitionMat[col][row] = self.kn_word2word(tup2[2], tup1[2])
            self.RemoveCompetingEdges(TransitionMat, tuplesMain, chunkDict)
            
            for row in range(nodeCount):
                row_sum = np.sum(TransitionMat[row, :])
                if(row_sum == 0):
                    print("Report ROW SUM ZERO W2W")
                TransitionMat[row, :] /= row_sum
                TransitionMat[row, row] = 0
        else:
            # FIXME:
            wordList = []
            for tupleSet in tuplesMain:
                for tup in tupleSet:
                    wordList.append(tup[2])
            for row in range(nodeCount):
                for col in range(nodeCount):
                    if row != col:
                        try:
                            TransitionMat[row][col] = self.fullCo_oc_mat[wordList[row]][wordList[col]]
                        except KeyError:
                            TransitionMat[row][col] = 0
                    else:
                        TransitionMat[row][col] = 0
                

                row_sum = np.sum(TransitionMat[row, :])
                if row_sum > 0:
                    TransitionMat[row, :] /= row_sum
                else:
                    TransitionMat[row, :] = 1/(nodeCount - 1)
                TransitionMat[row, row] = 0
            # print((TransitionMat[row, :]))
        # MakeRowStochastic(TransitionMat)
        return TransitionMat

    def get_w2w_samecng_mat(self, tuplesMain, chunkDict, kn_smooth = True):
        lastTuple = tuplesMain[len(tuplesMain) - 1]
        nodeCount = lastTuple[len(lastTuple) - 1][0] + 1
        
        TransitionMat = np.zeros((nodeCount, nodeCount))
        if kn_smooth:
            for i in range(0, len(tuplesMain) - 1):
                for j in range(i + 1, len(tuplesMain)):
                    tSet1 = tuplesMain[i]
                    tSet2 = tuplesMain[j]
                    for tup1 in tSet1:
                        for tup2 in tSet2:
                            row = tup1[0]
                            col = tup2[0]
                            # row != col, always
                            TransitionMat[row][col] = self.kn_word2word_samecng(tup1[2], tup2[2])
                            TransitionMat[col][row] = self.kn_word2word_samecng(tup2[2], tup1[2])
            
            self.RemoveCompetingEdges(TransitionMat, tuplesMain, chunkDict)
            
            for row in range(nodeCount):
                row_sum = np.sum(TransitionMat[row, :])
                if(row_sum == 0):
                    print("Report ROW SUM ZERO W2W")
                TransitionMat[row, :] /= row_sum
                TransitionMat[row, row] = 0
        else:
            wordList = []
            for tupleSet in tuplesMain:
                for tup in tupleSet:
                    wordList.append(tup[2])
            for row in range(nodeCount):
                for col in range(nodeCount):
                    if row != col:
                        try:
                            TransitionMat[row][col] = self.w2w_samecng_fullmat[wordList[row]][wordList[col]]
                        except KeyError:
                            TransitionMat[row][col] = 0
                    else:
                        TransitionMat[row][col] = 0
                
                row_sum = np.sum(TransitionMat[row, :])
                if row_sum != 0:
                    TransitionMat[row, :] /= row_sum
                else:
                    TransitionMat[row, :] /= 1/(nodeCount - 1)
                TransitionMat[row, row] = 0
            
        return TransitionMat

    def get_v2c_ranking(self, wordList, cngList, verbs):
        # Higher is better
        v2c_fullMat = self.v2c_fullMat
        ranks = np.zeros(len(cngList))
        for vi in verbs:
            tempRank = np.zeros(len(cngList))
            v = wordList[vi]
            # print(v)
            for i in range(len(cngList)):
                if i not in verbs:
                    c = str(cngList[i])
                    if c in v2c_fullMat[v]:
                        tempRank[i] = v2c_fullMat[v][c]
                    else:
                        tempRank[i] = 0
            ranks = np.max((tempRank, ranks), axis = 0)
            # print(ranks)
            
        s = np.sum(ranks)
        if(s > 0):
            ranks /= s
        
        return ranks

    def kn_word2word(self, word_a, word_b):
        fullCo_oc_mat = self.fullCo_oc_mat
        total_co_oc = self.total_co_oc
        total_context = self.total_context
        context_count = self.context_count
        
        delta = 0.5
        normalization = delta*total_context/total_co_oc

        if word_a in fullCo_oc_mat[word_b]:
            c_ab = max((fullCo_oc_mat[word_a][word_b] - delta), 0)/total_co_oc
            p_a = context_count[word_a]/total_context
            p_b = context_count[word_b]/total_context
            return c_ab + normalization*p_a*p_b
        else:
            p_a = max(context_count[word_a], 1)/total_context
            p_b = max(context_count[word_b], 1)/total_context

            return normalization*p_a*p_b



    def kn_cng2cng(self, cng_a, cng_b):
        # print(cng_a, cng_b)
        cng2index_dict = self.cng2index_dict
        cng2cngFullMat = self.cng2cngFullMat
        t2t_context_count = self.t2t_context_count
        t2t_total_contexts = self.t2t_total_contexts
        t2t_total_co_oc = self.t2t_total_co_oc

        delta = 0.5
        normalization = delta*t2t_total_contexts/t2t_total_co_oc

        try:
            index_a = cng2index_dict[str(cng_a)]
            index_b = cng2index_dict[str(cng_b)]
        except KeyError:
            return 1/440000

        if cng2cngFullMat[index_a, index_b] > 0:
            c_ab = max((cng2cngFullMat[index_a, index_b] - delta), 0)/t2t_total_co_oc
            # print(cng_a, cng_b, c_ab)
            p_a = t2t_context_count[index_a]/t2t_total_contexts
            p_b = t2t_context_count[index_b]/t2t_total_contexts
            return c_ab + normalization*p_a*p_b
        else:
            p_a = max(t2t_context_count[index_a], 1)/t2t_total_contexts
            p_b = max(t2t_context_count[index_b], 1)/t2t_total_contexts
            # print(p_a, p_b)
            return normalization*p_a*p_b

    def kn_word2word_samecng(self, word_a, word_b):
        w2w_samecng_fullmat = self.w2w_samecng_fullmat
        samecng_total_co_oc = self.samecng_total_co_oc
        samecng_total_context = self.samecng_total_context
        samecng_context_count = self.samecng_context_count
        
        delta = 0.5
        normalization = delta*samecng_total_context/samecng_total_co_oc

        if word_a in w2w_samecng_fullmat[word_b]:
            c_ab = max((w2w_samecng_fullmat[word_a][word_b] - delta), 0)/samecng_total_co_oc
            p_a = samecng_context_count[word_a]/samecng_total_context
            p_b = samecng_context_count[word_b]/samecng_total_context
            return c_ab + normalization*p_a*p_b
        else:
            p_a = max(samecng_context_count[word_a], 1)/samecng_total_context
            p_b = max(samecng_context_count[word_b], 1)/samecng_total_context
            return normalization*p_a*p_b