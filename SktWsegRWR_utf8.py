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

def CanCoExist_simple(p1, p2, n1, n2):
    # Make sure p1 is < p2, always
    if(p1 < p2):
        if(p1 + len(n1) - 1 < p2):
            return True
    return False

sandhiRules = pickle.load(open('extras/sandhiRules.p','rb'))    
def CanCoExist_sandhi(p1, p2, name1, name2):
    # P1 must be less than P2
    # Just send it in the proper order
    if(p1 < p2):
        overlap = max((p1 + len(name1)) - p2, 0)
        if overlap == 0:
            return True
        p1 = (name1[len(name1) - overlap:len(name1):], name2[0])
        p2 = (name1[-1], name2[0:overlap:])
        # print(p1, p2)
        if p1 in sandhiRules:
            # print(name1, name2, p1, ' = ', sandhiRules[p1])
            if(sandhiRules[p1]['length'] == 1):
                return True
        if p2 in sandhiRules:
            # print(name1, name2, p2, ' = ', sandhiRules[p2])
            if(sandhiRules[p2]['length'] == 1):
                return True
    return False

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
        (chunkDict, lemmaList, wordList, revMap2Chunk, qu, cngList, verbs, tuplesMain) = SentencePreprocess(sentenceObj)

        if(len(lemmaList) <= 1):
            # print("ERROR: Zero or one word in sentence...")
            return None

        # ALL FUNC USES KN SMOOTHING
        # USE THE SECOND ARGUMENT TO TURN IN OFF BY PASSING FALSE
        TransitionMat_t2t = self.t2t_modelFunc(tuplesMain)
        TransitionMat_w2w = self.w2w_modelFunc(tuplesMain)
        TransitionMat_w2w_samecng = self.sameCng_modelFunc(tuplesMain)

        if(len(qu) > 0):
            solution = [w for w in dcsObj.dcs_chunks]

            # INITIALIZATION OF RWR VECTORS/MATRICES
            lastTuple = tuplesMain[len(tuplesMain) - 1]
            nodeCount = lastTuple[len(lastTuple) - 1][0] + 1
            # print(nodeCount)
            deactivated = []
            prioriVec = np.ones((1, nodeCount))
            for q in qu:
                prioriVec[0, q] = 0
            prioriVec[0, qu[0]] = 1

            def deactivate(index):    
                # print("Remove:", wordList[index])
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

                    # v2c_scores = self.v2c_modelFunc(lemmaList, cngList, verbs)
                    # # print(v2c_scores)
                    # v2c_scores = 1/(1+v2c_scores)
                    # ranking_v2cng = np.asarray(v2c_scores.argsort())
                    # print(ranking_v2cng)

                    # NORMALIZE THE WEIGHT VECTORS
                    weights_w2w /= np.sum(weights_w2w)
                    weights_t2t /= np.sum(weights_t2t)
                    weights_w2w_samecng /= np.sum(weights_w2w_samecng)
                    # v2c_scores /= np.sum(v2c_scores)

                    # FIXME:
                    weights_combined = partition[0]*weights_w2w + partition[1]*weights_t2t + partition[2]*weights_w2w_samecng # + partition[3]*v2c_scores
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
                    for r in ranking_combined:
                        if(r in qu or r in deactivated):
                            continue
                        qu.append(r)
                        prioriVec[0,r] = 0
                        # Remove overlapping competitors
                        cid, pos, tid = revMap2Chunk[r]
                        # print("Result", r)
                        # print("Winner:", wordList[r], '<-', lemmaList[r])
                        # print(cid, pos, chunkDict[cid][pos])
                        break

                    for tup in tuplesMain[tid]:
                        if tup[0] not in qu:
                            deactivate(tup[0])

                    # Remove overlapping words
                    activeChunk = chunkDict[cid]
                    r = qu[len(qu) - 1]
                    winwin = wordList[r]
                    for _pos in activeChunk:
                        if(_pos < pos):
                            for indexDummy in activeChunk[_pos]:
                                tupSet = tuplesMain[indexDummy]
                                for tup in tupSet:
                                    index = tup[0]
                                    if index not in deactivated and index not in qu:
                                        w = wordList[index]
                                        if CanCoExist_sandhi(_pos, pos, w, winwin):
                                            deactivate(index)
                        elif(_pos > pos):
                            for indexDummy in activeChunk[_pos]:
                                tupSet = tuplesMain[indexDummy]
                                for tup in tupSet:
                                    index = tup[0]
                                    if index not in deactivated and index not in qu:
                                        w = wordList[index]
                                        if CanCoExist_sandhi(pos, _pos, winwin, w):
                                            deactivate(index)                    
                        else:
                            for indexDummy in activeChunk[_pos]:
                                tupSet = tuplesMain[indexDummy]
                                for tup in tupSet:
                                    index = tup[0]
                                    if(index != r):
                                        if index not in deactivated:
                                            deactivate(index)                    

                except KeyError:
                    break

                # print([lemmaList[i] for i in qu])
                # print(solution)

            result = list(map(lambda x: lemmaList[x], qu))
            # print(result)
            # if(result == None):
            #     print("NONE")

            # print(result)
            # ac = 100*sum(list(map(lambda x: lemmaList[x] in solution, qu)))/len(solution)
            return(result)
        else:
            # print("No unsegmentable chunks")
            return None

    def predictVerbose(self, sentenceObj, dcsObj):
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

            statFull = {}
            stepID = -1
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
                    stat = {}
                    for r in ranking_combined:
                        if(r in qu or r in deactivated):
                            continue
                        qu.append(r)
                        prioriVec[0,r] = 0
                        # Remove overlapping competitors
                        cid, pos = revMap2Chunk[r]
                        # print(r, chunkDict[cid][pos])
                        kii = np.where(r == chunkDict[cid][pos])[0][0]
                        stat['winner'] = getWord(sentenceObj, cid, pos, kii)
                        break

                    # Remove overlapping words
                    stat['removed'] = []
                    activeChunk = chunkDict[cid]
                    r = qu[len(qu) - 1]
                    for _pos in activeChunk:
                        # print('Checking Pos:', _pos)
                        if(_pos < pos):
                            for index in activeChunk[_pos]:
                                if index not in deactivated and index not in qu:
                                    w = wordList[index]
                                    if(_pos+len(w)-1 > pos):
                                        deactivate(index)
                                        goneCid, gonePos = revMap2Chunk[index]
                                        # try:
                                        kii = chunkDict[goneCid][gonePos].index(index)
                                        # except IndexError:
                                            # print(r, chunkDict[goneCid][gonePos])
                                        stat['removed'].append(getWord(sentenceObj, goneCid, gonePos, kii))


                        elif(_pos > pos):
                            winwin = wordList[r]
                            for index in activeChunk[_pos]:
                                if index not in deactivated and index not in qu:
                                    if(_pos < pos + len(winwin) - 1):
                                        deactivate(index)
                                        goneCid, gonePos = revMap2Chunk[index]
                                        kii = chunkDict[goneCid][gonePos].index(index)
                                        stat['removed'].append(getWord(sentenceObj, goneCid, gonePos, kii))
                        else:
                            for index in activeChunk[_pos]:
                                if(index != r):
                                    if index not in deactivated:
                                        deactivate(index)
                                        goneCid, gonePos = revMap2Chunk[index]
                                        kii = chunkDict[goneCid][gonePos].index(index)
                                        stat['removed'].append(getWord(sentenceObj, goneCid, gonePos, kii))

                    stepID += 1
                    statFull[str(stepID)] = stat

                except KeyError:
                    break

                # print([wordList[i] for i in qu])
                # print(solution)

            result = list(map(lambda x: wordList[x], qu))
            # if(result == None):
            #     print("NONE")

            # print(result)
            # ac = 100*sum(list(map(lambda x: wordList[x] in solution, qu)))/len(solution)
            statFull['sentence'] = dcsObj.sentence
            statFull['DCSLemma'] = dcsObj.lemmas
            return (result, statFull)
        else:
            # print("No unsegmentable chunks")
            return None

