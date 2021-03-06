import os, sys
import pickle
from DCS import *
from sentences import *
from utilities import *
import re
import numpy as np
import math
import pickle
from romtoslp import rom_slp
from wordTypeCheckFunction import *
import multiprocessing
from ProbModels import *
from graph import *
import csv
from IPython.display import display
import pandas as pd

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

def RWR(prioriVec, transMat, restartP, maxIteration, queryList, deactivated, allowRPModify = True, oldQueryList = np.array([])):

    """
    Run Random walk with restart
    until 
    we reach steady state or max iteration steps
    """
    # print('Priori SUM:', np.sum(prioriVec))
    transMat = np.copy(transMat)
    
    # FIRST TAKE CARE OF THE DEACTIVATED NODES
    transMat[:, deactivated] = 0
    transMat[deactivated, :] = 0
    
    # MERGE THE NEW QUERY NODE(IF ANY), CHANGES IN TRANSMAT AND PRIORI-VEC
    # THIS SHOULD CREATE A PROPER ROW-STOCHASTIC TRANSITION MATRIX
    
    # print(np.sum(transMat, axis = 1))
    if(len(queryList) > 1):

        # TODO - Using the sum probability logic
        
        '''
        # USING SIMPLE SUM
        
        # transMat[dest, :] = np.sum(transMat[queryList, :], axis=0)
        transMat[dest, :] = np.sum(transMat[queryList, :], axis=0)
        transMat[dest, :] /= np.sum(transMat[dest, :])
        transMat[queryList[1:], :] = 0
        '''
        if oldQueryList.shape[0] == 0:
            dest = queryList[0] # New merged node
            # Add the columns
            transMat[:, dest] = np.sum(transMat[:, queryList], axis=1)
            transMat[:, queryList[1:]] = 0

            # USING BAYES
            newPs = np.multiply(transMat[:, dest], prioriVec)
            # transMat[dest, :] = np.sum(transMat[queryList, :], axis=0)
            transMat[dest, :] = newPs
            transMat[dest, :] /= np.sum(transMat[dest, :])
            transMat[queryList[1:], :] = 0

        else:
            # USING WEIGHTED SUM
            newQueryList = list(set(queryList) - set(oldQueryList))
            
            dest = queryList[0] # New merged node

            # ADD THE COLUMNS
            c1 = np.sum(transMat[:, oldQueryList], axis=1)
            c2 = np.sum(transMat[:, newQueryList], axis=1)

            transMat[:, dest] = c1 + 0.1*c2
            transMat[:, queryList[1:]] = 0


            # ADD THE ROWS
            v1 = np.sum(transMat[oldQueryList, :], axis=0)
            v2 = np.sum(transMat[newQueryList, :], axis=0)

            transMat[dest, :] = v1 + 0.1*v2
            transMat[queryList[1:], :] = 0
            
    MakeRowStochastic(transMat)
    # print('Transum',np.sum(transMat, axis=1))



    
    eps = 0.0000000001    # the error difference, which should ideally be zero but can never be attained.
    
    n = prioriVec.shape[1]
    papMat = np.copy(prioriVec)
    
    rVec = np.zeros((1, n))

    '''FIXME: WHICH NODE(S) SHOULD BE THE QUERY NODES??
    '''
    # rVec[0, queryList[0]] = 1
    for i in queryList:
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
    # print(i, 'iterations')
    
    return(papMat)

# ONLY THE ONE WITH POSITIVE WEIGHTS HAVE BEEN TAKEN HERE
# goodCols = ['112', '113', '122', '123', '131', '213', '223', '231', '232', '311', '312', '322', '400', '500', '600', '700', '800']
# goodCols = ['111', '112', '113', '122', '123', '131', '132', '211', '212', '213', '222', '223', '231', '232', '311', '312', '322', '400', '500', '600', '700', '800']


goodCols = ['111', '112', '113', '122', '123', '131', '132', '211', '212', '213', '222', '223', '231', '232', '311', '312', '322', '400', '500', '600', '700', '800', '011', '022', '033']


'''
goodWeights = [ 0.00288289,  0.11299414,  0.02034902,  0.05426418,  0.3139211 ,
        0.09675109,  0.07450028,  0.04045034,  0.12735949,  0.08601451,
        0.00226412,  0.0114856 ,  0.03033762,  0.00314999,  0.00367026,
        0.01618957,  0.00341584]
'''
'''
goodWeights = [ 0.0383487 ,  0.00542029,  0.23419246,  0.04553042,  0.11657666,
        0.7726488 , -0.0310248 , -0.2273512 , -0.0942556 ,  0.2111668 ,
       -0.07864587,  0.18102852,  0.09075549,  0.27696369,  0.20928265,
        0.00544038,  0.02693159,  0.0686089 ,  0.00674246,  0.00769334,
        0.03832224,  0.00840885]
        
'''
#'''
goodWeights = [ -2.30184849,   0.15205544,  11.37284794,   2.22866016,
         2.5926891 ,  27.34128227,  -2.72856657, -15.95316347,
        -4.30608648,   9.95769945,  -3.72684989,   5.79208033,
         2.16235142,  13.69485912,   6.77834282,   0.43567416,
         1.20300337,   2.95172869,   0.24279471,   0.30121553,
         1.29004227,   0.27179539,  30.28146594,  -0.21266843,   9.93318708]
#'''
def Get_PCRW_ranking(qu, candNodes, qc_pairs, tMain_unrolled, df_pcrw_f):
    scores = np.zeros((len(tMain_unrolled), len(goodCols)));
    for cn in candNodes:
        try:
            df_picked = pd.DataFrame()
            for q in qu:
                ql = tMain_unrolled[q][2]
                cl = tMain_unrolled[cn][2]
                qcng = int(tMain_unrolled[q][3])
                ccng = int(tMain_unrolled[cn][3])
                df_picked = df_picked.append(df_pcrw_f.loc[(df_pcrw_f.ln_lemma == ql) & (df_pcrw_f.rn_lemma == cl) & (df_pcrw_f.ln_cng == qcng) & (df_pcrw_f.rn_cng == ccng), goodCols])
            scores[cn, :] = np.sum(df_picked[goodCols], axis = 0)
        except TypeError:
            pass
        
    for c in range(scores.shape[1]):
        scores[:, c] *= goodWeights[c]
    return np.sum(scores, axis = 1)

def CanCoExist_simple(p1, p2, n1, n2):
    # Make sure p1 is < p2, always
    if(p1 < p2):
        if(p1 + len(n1) - 1 < p2):
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
    def __init__(self, w2w_modelFunc, t2t_modelFunc, v2c_modelFunc, sameCng_modelFunc, df_PCRW, partition = np.array([1/3, 1/3, 1/3, 0])):
        self.w2w_modelFunc = w2w_modelFunc
        self.t2t_modelFunc = t2t_modelFunc
        self.sameCng_modelFunc = sameCng_modelFunc
        self.v2c_modelFunc = v2c_modelFunc
        self.df_pcrw = df_PCRW

        partition = np.array(partition)
        # partition /= sum(partition)
        self.partition = partition

    def predict(self, sentenceObj, dcsObj, verbose = False, supervised = False, eta = 0.1, algoname = '3RWR', **kwargs):
        if 'weightCollectorCSV' in kwargs:
            wcsv = kwargs['weightCollectorCSV']
        # eta = 0.1
        partition = self.partition
        # print(self.partition)
        try:
            (chunkDict, lemmaList, wordList, revMap2Chunk, qu, cngList, verbs, tuplesMain, qc_pairs) = SentencePreprocess(sentenceObj)
            tMain_unrolled = [t for ts in tuplesMain for t in ts]
            oldQueryList = np.copy(qu)
            df_pcrw_f = self.df_pcrw[self.df_pcrw.f == sentenceObj.sent_id + '.p2']
        except SentenceError as e:
            print('Empty name in file', sentenceObj.sent_id)
            if verbose:
                return None, None
            return None

        if(len(tuplesMain) <= 1):
            if verbose:
                return None, None
            return None

        if(len(qu) > 0):
            if(len(lemmaList) <= 1):
                # print("ERROR: Zero or one word in sentence...")
                if verbose:
                    return None, None
                return None

            if dcsObj != None:
                solution, solution_no_pvb = GetSolutions(dcsObj)
                
                solTuples = []
                for i in range(len(dcsObj.lemmas)):
                    for j in range(len(dcsObj.lemmas[i])):
                        solTuples.append((rom_slp(dcsObj.lemmas[i][j]), int(dcsObj.cng[i][j])))

            if verbose:
                # print(solTuples)
                runDetails = {}
                runDetails['sentence'] = sentenceObj.sentence
                runDetails['DCSLemmas'] = []
                if dcsObj != None:
                    for a in dcsObj.lemmas:
                        runDetails['DCSLemmas'].append([rom_slp(c) for c in a])


            # ALL FUNC USES KN SMOOTHING
            # USE THE SECOND ARGUMENT TO TURN IN OFF BY PASSING FALSE
            TransitionMat_t2t = self.t2t_modelFunc(tuplesMain, chunkDict)
            TransitionMat_w2w = self.w2w_modelFunc(tuplesMain, chunkDict)
            TransitionMat_w2w_samecng = self.sameCng_modelFunc(tuplesMain, chunkDict)

            if verbose:
                runDetails['TransitionMat_w2w'] = TransitionMat_w2w
                runDetails['TransitionMat_t2t'] = TransitionMat_t2t
                runDetails['TransitionMat_w2w_samecng'] = TransitionMat_w2w_samecng
                runDetails['nodeList'] = tuplesMain
                runDetails['initialQuery'] = str(qu)


            # INITIALIZATION OF RWR VECTORS/MATRICES
            lastTuple = tuplesMain[len(tuplesMain) - 1]
            nodeCount = lastTuple[len(lastTuple) - 1][0] + 1
            # print(nodeCount)
            deactivated = []
            prioriVec = np.ones((1, nodeCount))
            for q in qu:
                prioriVec[0, q] = 0
            prioriVec[0, qu[0]] = 1


                    
            if verbose:
                stepCount = -1
            while((len(qu) + len(deactivated)) <= (nodeCount - 1)):
                if verbose:
                    stepCount += 1
                    stepDetails = {}
                    stepDetails['removed'] = []

                def deactivate(index):
                    # print("Remove:", wordList[index], '<-', lemmaList[index])
                    if(index not in qu):
                        deactivated.append(index)
                        prioriVec[0,index] = 0
                        if verbose:
                            stepDetails['removed'].append((index, wordList[index], lemmaList[index], cngList[index]))
                try:
                    if algoname == '3RWR':
                        uniform_prob = 1/(nodeCount - len(qu) + 1 - len(deactivated))
                        prioriVec1 = np.copy((prioriVec != 0) * uniform_prob)
                        prioriVec2 = np.copy(prioriVec1)
                        prioriVec3 = np.copy(prioriVec1)

                        restartP = 0.4 # This is to be set based on graph diameter
                        weights_t2t = RWR(
                            prioriVec = prioriVec2, transMat = TransitionMat_t2t, 
                            restartP = restartP, maxIteration = 500, queryList = qu, 
                            deactivated = deactivated, allowRPModify = False, oldQueryList=oldQueryList)
                        ranking_t2t = weights_t2t[0].argsort()[::-1]


                        weights_w2w = RWR(
                            prioriVec = prioriVec1, transMat = TransitionMat_w2w, 
                            restartP = restartP, maxIteration = 500, queryList = qu, 
                            deactivated = deactivated, allowRPModify = False, oldQueryList=oldQueryList)
                        ranking_w2w = weights_w2w[0].argsort()[::-1]


                        weights_w2w_samecng = RWR(
                            prioriVec = prioriVec3, transMat = TransitionMat_w2w_samecng, 
                            restartP = restartP, maxIteration = 500, queryList = qu,
                            deactivated = deactivated, allowRPModify = False, oldQueryList=oldQueryList)
                        ranking_w2w_samecng = weights_w2w_samecng[0].argsort()[::-1]
                        
                        # NORMALIZE THE WEIGHT VECTORS
                        if np.sum(weights_w2w) > 0:
                            weights_w2w /= np.sum(weights_w2w)
                        if np.sum(weights_t2t) > 0:
                            weights_t2t /= np.sum(weights_t2t)
                        if np.sum(weights_w2w_samecng) > 0:
                            weights_w2w_samecng /= np.sum(weights_w2w_samecng)

                        # FIXME:
                        weights_combined = partition[0]*weights_w2w + partition[1]*weights_t2t + partition[2]*weights_w2w_samecng
                        ranking_combined = weights_combined[0].argsort()[::-1]

                    elif algoname == '25Path':
                        # SCORING OF NODES USING PATHS WITH LENGTHS MORE THAN 1
                        
                        # TODO: ADD NEW CODE HERE FOR PCRW
                        candNodes = set(range(nodeCount))
                        
                        weights_pcrw = Get_PCRW_ranking(qu, candNodes - set(qu) - set(deactivated), qc_pairs, tMain_unrolled, df_pcrw_f)
                        ranking_pcrw = weights_pcrw.argsort()[::-1]
                        
                        # NORMALIZE THE WEIGHT VECTORS
                        if np.sum(weights_pcrw) > 0:
                            weights_pcrw /= np.sum(weights_pcrw)

                        # FIXME:
                        weights_combined = weights_pcrw                        
                        ranking_combined = weights_combined.argsort()[::-1]
                        
                        # print(weights_combined.astype(list))
                        # print(ranking_combined)

                    if verbose:
                        if algoname == '3RWR':
                            stepDetails['w2w_score'] = weights_w2w
                            stepDetails['t2t_score'] = weights_t2t
                            stepDetails['w2w_samecng_score'] = weights_w2w_samecng
                            stepDetails['final_score'] = weights_combined
                            stepDetails['w2w_rank'] = ranking_w2w
                            stepDetails['t2t_rank'] = ranking_t2t
                            stepDetails['w2w_samecng_rank'] = ranking_w2w_samecng
                            
                        stepDetails['final_rank'] = ranking_combined


                    #print("W2W: ", ranking_w2w)
                    #print("W2W CNG: ", ranking_t2t)
                    #print("W2W SAMECNG: ", ranking_w2w_samecng)
                    #print(weights_pcrw[ranking_pcrw])
                    #print("PCRW Ranking: ", ranking_pcrw)
                    #print("*********************************")
                    #print("COMBINED: ", ranking_combined)
                    #print()


                    cid = -1
                    pos = -1

                    # if verbose or True:
                    if verbose or supervised:
                        if algoname == '3RWR':
                            winner_w2w = -1
                            for r in ranking_w2w:
                                if(r in qu or r in deactivated):
                                    continue
                                winner_w2w = r
                                break


                            winner_t2t = -1
                            for r in ranking_t2t:
                                if(r in qu or r in deactivated):
                                    continue
                                winner_t2t = r
                                break

                            winner_w2w_samecng = -1
                            for r in ranking_w2w_samecng:
                                if(r in qu or r in deactivated):
                                    continue
                                winner_w2w_samecng = r
                                break
                        elif algoname == '25Path':
                            winner_pcrw = -1
                            for r in ranking_pcrw:
                                if(r in qu or r in deactivated):
                                    continue
                                winner_pcrw = r
                                break
                        


                    # FIND OUT THE WINNER
                    winner_combined = -1
                    
                    dcsScores = [-1]*4
                    wrongLemmaScores = [-1]*4
                    # print('Ws', weights_combined)
                    # print('Rank List:', ranking_combined)
                    

                    for r in ranking_combined:
                        if(r in qu or r in deactivated):
                            continue
                        if winner_combined == -1:
                            winner_combined = r
                            qu.append(r)
                            prioriVec[0,r] = 0
                            # Remove overlapping competitors
                            cid, pos, tid = revMap2Chunk[r]

                            if verbose:
                                # print(winner_w2w, winner_t2t, winner_w2w_samecng, r)
                                stepDetails["winner"] = (r, wordList[r], lemmaList[r], cngList[r])
                                wt = (lemmaList[r], cngList[r])

                            if supervised:
                                if (lemmaList[r] in solution) or (lemmaList[r] in solution_no_pvb):
                                    #It's a correct weight, don't change anything
                                    break
                            else:
                                break

                        if supervised:
                            if (lemmaList[r] in solution) or (lemmaList[r] in solution_no_pvb):
                                # DCS lemma found
                                if dcsScores[0] < 0:
                                    dcsScores[0] = weights_w2w[0,r]
                                    dcsScores[1] = weights_t2t[0,r]
                                    dcsScores[2] = weights_w2w_samecng[0,r]
                            else:
                                # WRONG PREDICTION FOUND
                                if wrongLemmaScores[0] < 0:
                                    wrongLemmaScores[0] = weights_w2w[0,r]
                                    wrongLemmaScores[1] = weights_t2t[0,r]
                                    wrongLemmaScores[2] = weights_w2w_samecng[0,r]
                            if wrongLemmaScores[0] >= 0 and dcsScores[0] >= 0:
                                    # Both have been set
                                    diff_w2w = dcsScores[0] - wrongLemmaScores[0]
                                    diff_t2t = dcsScores[1] - wrongLemmaScores[1]
                                    diff_w2w_samecng = dcsScores[2] - wrongLemmaScores[2]

                                    if supervised:
                                        #==============================================
                                        # Supervised Learning of Weights
                                        #==============================================

                                        # HOW THE LAST SET OF PARTITION VALUES PERFORMED
                                        if wcsv != None:
                                            wcsv.writerow(np.append(partition,[diff_w2w, diff_t2t, diff_w2w_samecng]))
                                        
                                        partition[0] = partition[0] + eta*diff_w2w
                                        partition[1] = partition[1] + eta*diff_t2t
                                        partition[2] = partition[2] + eta*diff_w2w_samecng
                                        partition = partition/np.sum(partition)
                                        self.partition = partition
                                        # print(partition)
                                    break



                    for tup in tuplesMain[tid]:
                        if tup[0] not in qu:
                            deactivate(tup[0])

                    # Remove overlapping words
                    activeChunk = chunkDict[cid]
                    r = qu[len(qu) - 1]
                    winwin = wordList[r]
                    # print('Winner:', winwin)
                    for _pos in activeChunk:
                        if(_pos < pos):
                            for indexDummy in activeChunk[_pos]:
                                tupSet = tuplesMain[indexDummy]
                                for tup in tupSet:
                                    index = tup[0]
                                    if index not in deactivated and index not in qu:
                                        w = wordList[index]
                                        '''
                                        # FIXME: REMOVE TRY-CATCH
                                        '''
                                        try:
                                            if not CanCoExist_sandhi(_pos, pos, w, winwin):
                                                deactivate(index)
                                        except IndexError:
                                            print('Sandhi Related IndexError Occurred:', sentenceObj.sent_id)
                                            raise IndexError



                        elif(_pos > pos):
                            for indexDummy in activeChunk[_pos]:
                                tupSet = tuplesMain[indexDummy]
                                for tup in tupSet:
                                    index = tup[0]
                                    if index not in deactivated and index not in qu:
                                        w = wordList[index]


                                        # '''
                                        # # FIXME: REMOVE TRY-CATCH
                                        # '''
                                        # try:
                                        if not CanCoExist_sandhi(pos, _pos, winwin, w):
                                            deactivate(index)
                                        # except IndexError:
                                        #     print('Sandhi Related IndexError Occurred:', sentenceObj.sent_id)
                                        #     raise IndexError



                        else:
                            for indexDummy in activeChunk[_pos]:
                                tupSet = tuplesMain[indexDummy]
                                for tup in tupSet:
                                    index = tup[0]
                                    if(index != r):
                                        if index not in deactivated:
                                            deactivate(index)
                                        

                    ## Find QUERY nodes now
                    # print(qu)
                    tuples = []
                    for pos in activeChunk.keys():
                        tupIds = chunkDict[cid][pos]
                        for tupId in tupIds:
                            for tup in tuplesMain[tupId]:
                                if tup[0] not in qu and tup[0] not in deactivated:
                                    # POS, ID, NAME
                                    tuples.append((pos, tup[0], tup[1]))

                    for u in range(len(tuples)):
                        tup1 = tuples[u]
                        quFlag = True
                        for v in range(len(tuples)):
                            if(u == v):
                                continue
                            tup2 = tuples[v]


                            # '''
                            # # FIXME: REMOVE TRY-CATCH
                            # '''
                            # try:
                            if(tup1[0] < tup2[0]):
                                if not CanCoExist_sandhi(tup1[0], tup2[0], tup1[2], tup2[2]):
                                    ## Found a competing node - hence can't be a query
                                    quFlag = False
                                    break
                            elif(tup1[0] > tup2[0]):
                                if not CanCoExist_sandhi(tup2[0], tup1[0], tup2[2], tup1[2]):
                                    ## Found a competing node - hence can't be a query
                                    quFlag = False
                                    break
                            else:
                                quFlag = False
                                break
                            # except IndexError:
                            #     print('Sandhi Related IndexError Occurred:', sentenceObj.sent_id)
                            #     raise IndexError


                        if quFlag:
                            qu.append(tup1[1])
                    
                    ## NEW CODE - END
                    # print(qu)

                    if verbose:
                        stepDetails['updated_query'] = list(qu)

                except KeyError:
                    break

                if verbose:
                    runDetails[str(stepCount)] = stepDetails

                # print([lemmaList[i] for i in qu])
                # print(solution)

            result = list(map(lambda x: lemmaList[x], qu))

            if verbose:
                runDetails['prediction'] = result
                if dcsObj != None:
                    runDetails['accuracy'] = Accuracy(result, dcsObj)
                runDetails['steps'] = stepCount
                return(result, runDetails)
            else:
                return(result)
        else:
            # print("No unsegmentable chunks")
            if verbose:
                return None, None
            return None

    
