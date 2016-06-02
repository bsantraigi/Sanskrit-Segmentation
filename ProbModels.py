import pickle
from utilities import printProgress, validatePickleName, pickleFixLoad
import numpy as np
import math
import pickle
from wordTypeCheckFunction import *

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


class ProbModels():
    def __init__(self):
        print("Loading Prob. Models")
        self.model_cbow = pickleFixLoad('extras/modelpickle10.p')
        
        self.fullCo_ocMat = pickle.load(open('extras/all_dcs_lemmas_matrix.p', 'rb'))
        self.word2IndexDict = pickle.load(open('dcsLemma2index.p', 'rb'))

        self.cng2cngFullMat = np.mat(pickleFixLoad('extras/all_dcs_cngs_matrix_countonly.p'))
        self.cng2index_dict = pickle.load(open('cng2index_dict.p', 'rb'))

        print("Prob. Models loaded")
        return
    def get_cng2cng_mat(self, cngList):
        cng2cngFullMat = self.cng2cngFullMat
        cng2index_dict = self.cng2index_dict
        nodeCount = len(cngList)
        cngIndexList = list(map(lambda x:cng2index_dict[str(x)], cngList))
        # print(cngIndexList)
        TransitionMat = np.zeros((nodeCount, nodeCount))
        
        """
        FIXME:
        1. HOW TO DO SMOOTHING?
        2. HOW TO CONVERT WORD2VEC SIM. TO PROB.
        """
        
        for row in range(nodeCount):
            for col in range(nodeCount):
                if row != col:
                    try:
    #                     print(cngIndexList[row])
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
            
            TransitionMat[row, row] = 0
            # print((TransitionMat[row, :]))
        # MakeRowStochastic(TransitionMat)
        return TransitionMat

    def get_word2vec_mat(self, wordList):
        model_cbow = self.model_cbow
        nodeCount = len(wordList)
        TransitionMat = np.zeros((nodeCount, nodeCount))
        
        """
        FIXME:
        1. HOW TO DO SMOOTHING?
        2. HOW TO CONVERT WORD2VEC SIM. TO PROB.
        """
        
        for row in range(nodeCount):
            for col in range(nodeCount):
                if row != col:
                    try:
                        TransitionMat[row][col] = sigmoid(model_cbow.similarity(wordList[row], wordList[col]))
                    except KeyError:
                        TransitionMat[row][col] = 0 #WHAT TO DO HERE??
                else:
                    TransitionMat[row][col] = 0
            
            row_sum = np.sum(TransitionMat[row, :])
            if(row_sum > 0):
                delta = 0.1
                NZ = np.sum(TransitionMat[row, :] != 0)
                if(NZ < nodeCount - 1):
                    Z = nodeCount - 1 - NZ
                    TransitionMat[row, :] -= delta
                    TransitionMat[row, :] = TransitionMat[row, :].clip(min = 0)
                    TransitionMat[row, :] /= row_sum
                    filler = delta*NZ/(row_sum*Z)
                    TransitionMat[row, :] += filler * (TransitionMat[row, :] == 0)
                else:
                    TransitionMat[row, :] /= row_sum
            else:
                TransitionMat[row, :] = 1/(nodeCount - 1)
            
            TransitionMat[row, row] = 0
            # print((TransitionMat[row, :]))
        # MakeRowStochastic(TransitionMat)
        return TransitionMat

    def getCo_occurMat(self, wordList):    
        fullCo_ocMat = self.fullCo_ocMat
        word2IndexDict = self.word2IndexDict
        nodeCount = len(wordList)
        wordIndexList = [-1]*nodeCount
        i = -1
        for w in wordList:
            i += 1
            try:
                wordIndexList[i] = word2IndexDict[w]
            except KeyError:
                continue
        TransitionMat = np.zeros((nodeCount, nodeCount))
        
        """
        FIXME:
        1. HOW TO DO SMOOTHING?
        2. HOW TO CONVERT WORD2VEC SIM. TO PROB.
        """
        
        for row in range(nodeCount):
            for col in range(nodeCount):
                if row != col:
                    try:
                        TransitionMat[row][col] = fullCo_ocMat[wordIndexList[row]][wordIndexList[col]]
                    except KeyError:
                        TransitionMat[row][col] = 0 #WHAT TO DO HERE??
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
