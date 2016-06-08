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

Pass weights for each method in partition array in the following order
[w2w_cooccurrence, t2t, samecng, -]
"""
class SktWsegRWR(object):
    # print("Loaded: ", model_cbow)
    def __init__(self, w2w_modelFunc, t2t_modelFunc, v2c_modelFunc, sameCng_modelFunc, partition = [1/3, 1/3, 1/3, 0]):
        self.w2w_modelFunc = w2w_modelFunc
        self.t2t_modelFunc = t2t_modelFunc
        self.sameCng_modelFunc = sameCng_modelFunc
        self.v2c_modelFunc = v2c_modelFunc

        partition = np.array(partition)
        partition /= sum(partition)
        self.partition = partition


    def predict(self, sentenceObj, dcsObj):
        partition = self.partition
        (chunkDict, wordList, revMap2Chunk, qu, cngList, verbs) = SentencePreprocess(sentenceObj)

        if(len(wordList) <= 1):
            # print("ERROR: Zero or one word in sentence...")
            return None

        # ALL FUNC USES KN SMOOTHING
        # USE THE SECOND ARGUMENT TO TURN IN OFF BY PASSING FALSE
        TransitionMat_t2t = self.t2t_modelFunc(cngList)
        TransitionMat_w2w = self.w2w_modelFunc(wordList)
        TransitionMat_w2w_samecng = self.sameCng_modelFunc(wordList)

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
                    prioriVec1 = np.copy((prioriVec != 0) * uniform_prob)
                    prioriVec2 = np.copy(prioriVec1)
                    prioriVec3 = np.copy(prioriVec1)

                    restartP = 0.4 # This is to be set based on graph diameter
                    # print(deactivated)
                    weights_w2w = RWR(
                        prioriVec = prioriVec1, transMat = TransitionMat_w2w, 
                        restartP = restartP, maxIteration = 500, queryList = qu, 
                        deactivated = deactivated, allowRPModify = False)
                    # weights_w2w = 1/weights_w2w # to sort it in ascending order
                    # print(weights_w2w)
                    weights_w2w = 1/(1+weights_w2w)
                    ranking_w2w = np.asarray(weights_w2w.argsort())
                    # print(ranking_w2w)

                    weights_t2t = RWR(
                        prioriVec = prioriVec2, transMat = TransitionMat_t2t, 
                        restartP = restartP, maxIteration = 500, queryList = qu, 
                        deactivated = deactivated, allowRPModify = False)
                    # print(weights_t2t)
                    weights_t2t = 1/(1+weights_t2t)
                    ranking_t2t = np.asarray(weights_t2t.argsort())
                    # print(ranking_t2t)

                    weights_w2w_samecng = RWR(
                        prioriVec = prioriVec3, transMat = TransitionMat_w2w_samecng, 
                        restartP = restartP, maxIteration = 500, queryList = qu,
                        deactivated = deactivated, allowRPModify = False)
                    # print(weights_w2w_samecng)
                    weights_w2w_samecng = 1/(1+weights_w2w_samecng)
                    ranking_w2w_samecng = np.asarray(weights_w2w_samecng.argsort())
                    # print(ranking_w2w_samecng)

                    v2c_scores = self.v2c_modelFunc(wordList, cngList, verbs)
                    # print(v2c_scores)
                    v2c_scores = 1/(1+v2c_scores)
                    ranking_v2cng = np.asarray(v2c_scores.argsort())
                    # print(ranking_v2cng)

                    # NORMALIZE THE WEIGHT VECTORS
                    weights_w2w /= np.sum(weights_w2w)
                    weights_t2t /= np.sum(weights_t2t)
                    weights_w2w_samecng /= np.sum(weights_w2w_samecng)
                    v2c_scores /= np.sum(v2c_scores)

                    weights_combined = partition[0]*weights_w2w + partition[1]*weights_t2t + partition[2]*weights_w2w_samecng + partition[3]*v2c_scores
                    ranking_combined = np.asarray(weights_combined.argsort()).reshape(-1)

                    # print("W2W: ", ranking_w2w)
                    # print("W2W CNG: ", ranking_t2t)
                    # print("W2W SAMECNG: ", ranking_w2w_samecng)
                    # print("V2C Ranking: ", ranking_v2cng)
                    # print("*********************************")
                    # print("COMBINED: ", ranking_combined)
                    # print()


                    cid = -1
                    pos = -1
                    
                    # FIND OUT THE WINNER
                    for r in ranking_combined[::-1]:
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
            # if(result == None):
            #     print("NONE")

            # print(result)
            # ac = 100*sum(list(map(lambda x: wordList[x] in solution, qu)))/len(solution)
            return(result)
        else:
            # print("No unsegmentable chunks")
            return None

