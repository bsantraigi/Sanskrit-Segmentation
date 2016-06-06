import os, sys
import pickle
from DCS import *
from sentences import *
from utilities import printProgress, validatePickleName, pickleFixLoad
import re
import numpy as np
import math
import pickle
from romtoslp import rom_slp
from wordTypeCheckFunction import *
import multiprocessing
from ProbModels import *
from graph import *

class Method():
    word2vec = 0
    word2word = 1
    type2type = 2
    verb2type = 3

np.set_printoptions(precision=2, suppress= True)

def MakeRowStochastic(matrix):
    rowCount = matrix.shape[0]
    for row in range(rowCount):
        s = np.sum(matrix[row, :])
        if(s!=0):
            matrix[row, :] = matrix[row, :]/s

def RWR(prioriVec, transMat, restartP, maxIteration, queryList, deactivated, allowRPModify = True):
    """
    Run Random walk with restart
    until 
    we reach steady state or max iteration steps
    """
#     print(prioriVec)
    transMat = np.copy(transMat)
    
#     MERGE THE NEW QUERY NODE(IF ANY), CHANGES IN TRANSMAT AND PRIORI-VEC
    doMax = True
    if(len(queryList) > 1):
        dest = queryList[0]
        if(doMax):
            # Using the max probability logic
            transMat[dest, :] = np.max(transMat[queryList, :], axis=0)
            transMat[queryList[1:], :] = 0   
            
            transMat[:, dest] = np.max(transMat[:, queryList], axis=1)
            transMat[:, queryList[1:]] = 0
        else:
            # TODO - Using the sum probability logic
            transMat[dest, :] = np.sum(transMat[queryList, :], axis=0)
            transMat[queryList[1:], :] = 0
            
            transMat[:, dest] = np.sum(transMat[:, queryList], axis=1)
            transMat[:, queryList[1:]] = 0

    transMat[:, deactivated] = 0
    transMat[deactivated, :] = 0
    # MakeRowStochastic(transMat)


    
    eps = 0.0000000001    # the error difference, which should ideally be zero but can never be attained.
    
    n = prioriVec.shape[1]
    papMat = np.copy(prioriVec)
    
    rVec = np.zeros((1, n))    
#     print(n)
    for i in queryList:
#         print(i)
        rVec[0, i] = 1/len(queryList)

    nodes = []
    for i in range(prioriVec.shape[1]):
        if(prioriVec[0,i] > 0):
            nodes.append(i)

    if(allowRPModify):
        """
        Find Dia of graph using Floyd Warshall
        Dynamically decide restart probability
        """
        dia = GetDiaFromTransmat(nodes, transMat)
        rp_new = 1 - math.pow(.045, 1/dia)
        # print( "Dia: ", dia, " RP: ", rp_new)
    else:
        rp_new = restartP
    
    
    # print(papMat)
    for i in range(maxIteration):
        newMat = (1 - rp_new) * np.dot(papMat, transMat) + rp_new * np.mat(rVec)        
        diff = np.absolute(papMat - newMat)
        diffMax = np.max(diff)
        # print(diffMax)
        papMat = np.copy(newMat)
        # print(papMat)
#         print(diffMax)
        if  abs(diffMax) < eps and i > 25:
            break
    
    return(papMat)

"""
Loads the Model_CBOW from file
Keeps a full list of train files and target sentences
Test on a single sentence of a set of sentences
"""
class SktWsegRWR(object):
    # print("Loaded: ", model_cbow)
    def __init__(self, w2w_modelFunc, t2t_modelFunc, v2t_modelFunc, sameCng_modelFunc, partition = [1/4, 1/4, 1/4, 1/4]):
        self.w2w_modelFunc = w2w_modelFunc
        self.t2t_modelFunc = t2t_modelFunc
        self.sameCng_modelFunc = sameCng_modelFunc
        self.v2t_modelFunc = v2t_modelFunc
        self.partition = partition

    def predict(self, sentenceObj, dcsObj):
        (chunkDict, wordList, revMap2Chunk, qu, cngList) = SentencePreprocess(sentenceObj)

        if(len(wordList) == 0):
            return

        TransitionMat_t2t = t2t_modelFunc(cngList)
        TransitionMat_w2w = w2w_modelFunc(wordList)

        if(len(qu) > 0):
            solution = [w for w in dcsObj.dcs_chunks]

            # INITIALIZATION OF RWR VECTORS/MATRICES
            nodeCount = len(wordList)
            deactivated = []
            prioriVec = np.ones((1, nodeCount))
            for q in qu:
                prioriVec[0, q] = 0
            prioriVec[0, qu[0]] = 1

            def deactivate(index):    
                deactivated.append(index)
                prioriVec[0,index] = 0

            while((len(qu) + len(deactivated)) <= (nodeCount - 1)):
                try:
                    uniform_prob = 1/(nodeCount - len(qu) + 1 - len(deactivated))
                    prioriVec = (prioriVec != 0) * uniform_prob

                    restartP = 0.1 # This is to be set based on graph diameter
                    # print(deactivated)
                    weights_w2w = RWR(
                        prioriVec = prioriVec, transMat = TransitionMat_w2w, 
                        restartP = restartP, maxIteration = 500, queryList = qu, 
                        deactivated = deactivated, allowRPModify = False)
                    ranking_w2w = np.asarray(weights_w2w.argsort()).reshape(-1)

                    weights_t2t = RWR(
                        prioriVec = prioriVec, transMat = TransitionMat_t2t, 
                        restartP = restartP, maxIteration = 500, queryList = qu, 
                        deactivated = deactivated, allowRPModify = False)
                    ranking_t2t = np.asarray(weights_t2t.argsort()).reshape(-1)

                    cid = -1
                    pos = -1
                    
                    # FIND OUT THE WINNER
                    for r in ranking[::-1]:
                        if(r in qu or r in deactivated):
                            continue
                        qu.append(r)
                        prioriVec[0,r] = 0
                        # Remove overlapping competitors
                        cid, pos = revMap2Chunk[r]
                        break

                    # Remove overlapping words
                    activeChunk = chunkDict[cid]
                    r = qu[len(qu) - 1]
                    for _pos in activeChunk:
                        if(_pos < pos):
                            for index in activeChunk[_pos]:
                                if index not in deactivated and index not in qu:
                                    w = wordList[index]
                                    if(_pos+len(w)-1 > pos):
                                        deactivate(index)                    
                        elif(_pos > pos):
                            winwin = wordList[r]
                            for index in activeChunk[_pos]:
                                if index not in deactivated and index not in qu:
                                    if(_pos < pos + len(winwin) - 1):
                                        deactivate(index)                    
                        else:
                            for index in activeChunk[_pos]:
                                if(index != r):
                                    if index not in deactivated:
                                        deactivate(index)                    

                except KeyError:
                    break

                # print([wordList[i] for i in qu])
                # print(solution)

            result = list(map(lambda x: wordList[x], qu))
            # print(result)
            # ac = 100*sum(list(map(lambda x: wordList[x] in solution, qu)))/len(solution)
            return(result)

