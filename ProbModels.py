import pickle
from utilities import printProgress, validatePickleName, pickleFixLoad
import numpy as np
import math
import pickle
from wordTypeCheckFunction import *
from collections import defaultdict

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


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

        total_sentences = 441735
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

        """W2W_SAME_CNG DATA"""

        
        return

    def get_cng2cng_no_KN(self, cngList):
        cng2cngFullMat = self.cng2cngFullMat
        cng2index_dict = self.cng2index_dict
        nodeCount = len(cngList)
        # print(cngList)
        cngIndexList = list(map(lambda x:cng2index_dict[str(x)], cngList))
        # print(cngIndexList)
        TransitionMat = np.zeros((nodeCount, nodeCount))
        
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

    def get_cng2cng_mat(self, cngList):
        nodeCount = len(cngList)
        TransitionMat = np.zeros((nodeCount, nodeCount))
        
        for row in range(nodeCount):
            for col in range(nodeCount):
                if row != col:
                    TransitionMat[row][col] = self.kn_cng2cng(cngList[row], cngList[col])
                else:
                    TransitionMat[row][col] = 0
            
            row_sum = np.sum(TransitionMat[row, :])
            TransitionMat[row, :] /= row_sum
            TransitionMat[row, row] = 0
            # print((TransitionMat[row, :]))
        # MakeRowStochastic(TransitionMat)
        return TransitionMat

    

    def get_w2w_mat(self, wordList):    
        nodeCount = len(wordList)
        TransitionMat = np.zeros((nodeCount, nodeCount))
        
        for row in range(nodeCount):
            for col in range(nodeCount):
                if row != col:
                    TransitionMat[row][col] = self.kn_word2word(wordList[row], wordList[col])
                else:
                    TransitionMat[row][col] = 0
            
            row_sum = np.sum(TransitionMat[row, :])
            if(row_sum > 0):
                TransitionMat[row, :] /= row_sum
            else:
                TransitionMat[row, :] = 1/(nodeCount - 1)
            
            TransitionMat[row, row] = 0
            # print((TransitionMat[row, :]))
        # MakeRowStochastic(TransitionMat)
        return TransitionMat

    def get_w2w_no_KN(self, wordList):    
        nodeCount = len(wordList)
        TransitionMat = np.zeros((nodeCount, nodeCount))
        
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
            TransitionMat[row, :] /= row_sum
            TransitionMat[row, row] = 0
            # print((TransitionMat[row, :]))
        # MakeRowStochastic(TransitionMat)
        return TransitionMat

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
            p_a = context_count[word_a]/total_context
            p_b = context_count[word_b]/total_context
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

        index_a = cng2index_dict[str(cng_a)]
        index_b = cng2index_dict[str(cng_b)]        

        if cng2cngFullMat[index_a, index_b] > 0:
            c_ab = max((cng2cngFullMat[index_a, index_b] - delta), 0)/t2t_total_co_oc
            # print(cng_a, cng_b, c_ab)
            p_a = t2t_context_count[index_a]/t2t_total_contexts
            p_b = t2t_context_count[index_b]/t2t_total_contexts
            return c_ab + normalization*p_a*p_b
        else:
            p_a = t2t_context_count[index_a]/t2t_total_contexts
            p_b = t2t_context_count[index_b]/t2t_total_contexts
            # print(p_a, p_b)
            return normalization*p_a*p_b