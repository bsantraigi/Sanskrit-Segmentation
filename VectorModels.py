import pickle
from utilities import printProgress, validatePickleName, pickleFixLoad
import numpy as np
import math
import pickle
from wordTypeCheckFunction import *
from collections import defaultdict

"""
These are based on gensim package 

CBOW word2vec models

-----------------------------------

Uses Discounting followed by uniform distribution 
for smoothing
"""
class VectorModels():
    def __init__(self, **kwargs):
        kwargs = defaultdict(lambda: None, kwargs)

        print("Loading Word2Vec Models")
        
        # WORD2WORD DATA
        self.model_cbow = 

        # TYPE2TYPE DATA

        # VERB2TYPE DATA

        # W2W_SAME_CNG DATA

        print("Word2Vec models loaded")
        return

    def get_w2w_mat(self, wordList):
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