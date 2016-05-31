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
import multiprocessing

class Method():
    word2vec = 0
    word2word = 1
    type2type = 2
    verb2type = 3

np.set_printoptions(precision=2, suppress= True)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def GetDiaFromTransmat(nodes, transMat):
    """
    Transmat is probability matrix
    Convert it to a edge weight adjacency matrix
    before calling Floyd_Warshall
    """
#     print('Transition Prob. Matrix: ')
#     print(transMat)
    adjMat = np.copy(transMat[nodes, :])
    adjMat = adjMat[:, nodes]
    with np.errstate(divide='ignore'):
        adjMat = 1/adjMat
#     print('Graph Adj. Matrix: ')
#     print(adjMat)
    sp = Floyd_Warshall(adjMat)
    dia = np.max(sp)
    # print(sp)
    return dia

def Floyd_Warshall(adjMat):
    l = adjMat.shape[0]
    D = np.copy(adjMat)
    for i in range(l):
        D[i, i] = 0
    for k in range(l):
        D_new = np.zeros(D.shape)
        for i in range(l):
            for j in range(l):
                D_new[i, j] = np.min([D[i,j], D[i, k] + D[k, j]])
        D = D_new
#     print("Shortest Path Matrix: ")
#     print(D)
    return(D)

def getCo_occurMat(wordList, fullCo_ocMat, word2IndexDict):    
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

def getTransMat(wordList, model_cbow):
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
    # print(prioriVec)
    
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


    
    eps = 0.0000000000001    # the error difference, which should ideally be zero but can never be attained.
    
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
    else:
        rp_new = restartP
    # print( "Dia: ", dia, " RP: ", rp_new)

    for i in range(maxIteration):        
        newMat = (1 - rp_new) * np.dot(papMat, transMat) + rp_new * np.mat(rVec)        
        diff = np.absolute(papMat - newMat)
        diffMax = np.argmax(diff)
        papMat = newMat
        # print(papMat)
        if  abs(diffMax) < eps and maxIteration/100.0 > 1:
            break
                  
    return(papMat)


class AlgoTestFactory(multiprocessing.Process):
    def __init__(self, testRange, processID, method = Method.word2vec, sentencesPath = '../TextSegmentation/Pickles/', dcsPath = '../Text Segmentation/DCS_pick/', storeAccuracies = False):
        multiprocessing.Process.__init__(self)
        if(sys.version_info < (3, 0)):
            warnings.warn("\nPython version 3 or greater is required. Python 2.x is not tested.\n")

        """
           Folder @ sentencesPath contains pickle files for "sentences" object
           Folder @ path2 contains pickle files for the same sentences
           as in Folder @ sentencesPath but its DCS equivalent
        """
        self.sentencesPath = sentencesPath
        self.dcsPath = dcsPath
        self.testRange = testRange
        self.processID = processID
        self.method = method
        self.storeAccuracies = storeAccuracies


        """
        Get common dcs and sentences files        
        Uncomment to refresh fileLists
        """
        # self.sentenceFiles=set(sorted(os.listdir(sentencesPath)))
        # self.dcsFiles=set(sorted(os.listdir(dcsPath)))
        # self.commonFiles = []
        
        # for sPickle in self.sentenceFiles:
        #     if sPickle in self.dcsFiles:
        #         sPickle = validatePickleName(sPickle)
        #         if sPickle != "":                
        #             self.commonFiles.append(sPickle)

        # self.commonFiles = list(set(self.commonFiles))
        # pickle.dump(self.commonFiles, open('commonFiles.p', 'wb'))

        """
        Load file list from pickle
        """
        # self.commonFiles = pickle.load(open("commonFiles.p", 'rb'))

        # print("Current folder contains: ",len(self.commonFiles), " Files")

        self.algo = SktWsegRWR(method=self.method)

    def loadSentence(self, fName):
        try:
            sentenceObj = pickleFixLoad(self.sentencesPath + fName)
            dcsObj = pickleFixLoad(self.dcsPath + fName)
        except (KeyError, EOFError) as e:
            return None, None
        return(sentenceObj, dcsObj)

    def run(self):
        accuracies = []
        # print(self.testRange[0])
        # print(AlgoTestFactory.commonFiles[self.testRange[0]])
        # return
        for f in AlgoTestFactory.commonFiles[self.testRange[0]:self.testRange[1]]:
        # f = self.commonFiles[33]
            sentenceObj, dcsObj = self.loadSentence(f)
            if(sentenceObj != None):
                try:
                    result = self.algo.predict(sentenceObj, dcsObj)
                except ZeroDivisionError:
                    result = None
                    pass
                solution = [rom_slp(c) for c in dcsObj.dcs_chunks]
                if result != None:
                    ac = 100*sum(list(map(lambda x: x in solution, result)))/len(solution)
                    accuracies.append(ac)
                    if not self.storeAccuracies:
                        print(ac)
                    # print("Solution: ", solution)
                    # print("Prediction: ", result)
        
        # AlgoTestFactory.allAccuracies.append(accuracies)
        if(self.storeAccuracies):
            pickle.dump(accuracies, open('.temp/' + str(self.processID) + '_out.p', 'wb'))
            print('Process Finished (accuracies saved to disk)')
        else:
            print('Process Finished (accuracies not saved to disk)')


AlgoTestFactory.commonFiles = pickle.load(open("commonFiles.p", 'rb'))
print("Current folder contains: ",len(AlgoTestFactory.commonFiles), " Files")
AlgoTestFactory.allAccuracies = []

"""
SentencePreprocess:
-------------------
    Read a sentence obj and create + return the following objects 

    -> chunkDict: chunk_id -> position -> index in wordlist (nested dictionary)
    -> wordList: list of possible words as a result of word segmentation
    -> revMap2Chunk: Map word in wordlist to (cid, position) in chunkDict
    -> qu: Possible query nodes
"""
def SentencePreprocess(sentenceObj):
    """
    Considering word names only
    ***{Word forms or cngs can also be used}
    """
    chunkDict = {}
    wordList = []
    revMap2Chunk = []
    qu = []

    cid = -1
    for chunk in sentenceObj.chunk:
        # print()
        cid = cid + 1
        chunkDict[cid] = {}
        canBeQuery = 0
        if len(chunk.chunk_words.keys()) == 1:
            canBeQuery = 1 # Unsegmentable Chunk
        for pos in chunk.chunk_words.keys():
            chunkDict[cid][pos] = []
            if(canBeQuery == 1) and (len(chunk.chunk_words[pos]) == 1):
                canBeQuery = 2 # No cng alternative for the word
            for word_sense in chunk.chunk_words[pos]:
                if(len(word_sense.lemmas) > 0):
                    wordList.append(rom_slp(word_sense.lemmas[0]))
                    k = len(wordList) - 1
                    chunkDict[cid][pos].append(k)
                    revMap2Chunk.append((cid, pos))
                    if canBeQuery == 2:
                        # The word has a lemma available - in some pickle file it's not
                        # Make this word query
                        qu.append(k)
    return (chunkDict, wordList, revMap2Chunk, qu)

"""
Loads the Model_CBOW from file
Keeps a full list of train files and target sentences
Test on a single sentence of a set of sentences
"""
class SktWsegRWR(object):
    print("Loading Prob. Models")
    modelFilePath = 'extras/modelpickle10.p'
    model_cbow = pickleFixLoad(modelFilePath)
    fullCo_ocMat = pickle.load(open('extras/all_dcs_lemmas_matrix.p', 'rb'))
    word2IndexDict = pickle.load(open('dcsLemma2index.p', 'rb'))
    print("Prob. Models loaded")
    # print("Loaded: ", model_cbow)
    def __init__(self, method = Method.word2vec, modelFilePath = 'extras/modelpickle10.p'):
        """
        Load the CBOW pickle
        """
        # modelFilePath = 'extras/model_100_10.p'
        self.model_cbow = SktWsegRWR.model_cbow
        self.fullCo_ocMat = SktWsegRWR.fullCo_ocMat
        self.word2IndexDict = SktWsegRWR.word2IndexDict
        self.method = method

        if(self.method == Method.word2word):
            print("Using word2word")
        if(self.method == Method.word2vec):
            print("Using word2vec")

            

        # print("Loaded: ", self.model_cbow)


    def predict(self, sentenceObj, dcsObj):
        """-----------------------------------------------------------
        PART 2
            1. LOAD A SENTENCE
            2. UNIFORM PRIOR PROB.
            3. SET QUERY NODE
            3. RUN RANDOM WALK
            4. CHOOSE WINNER
            5. MERGE QUERY NODES
            6. RERUN FROM 2
        -----------------------------------------------------------"""
        
        """
        Considering word names only
        ***{Word forms or cngs can also be used}
        """
        (chunkDict, wordList, revMap2Chunk, qu) = SentencePreprocess(sentenceObj)
        
        if(len(wordList) == 0):
            return

        if(self.method == Method.word2vec):
            TransitionMat = getTransMat(wordList, self.model_cbow)
        elif(self.method == Method.word2word):
            TransitionMat = getCo_occurMat(wordList, self.fullCo_ocMat, self.word2IndexDict)
        if(len(qu) > 0):
            # for q in qu:
            #     print(q,wordList[q], end=" ")
                
            
            # print("------------DCS--------------")
            # print(dcsObj.sentence)
            solution = [w for w in dcsObj.dcs_chunks]
            # print(solution)

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

                    restartP = 0.4 # This is to be set based on graph diameter
                    # print(deactivated)
                    weights = RWR(
                        prioriVec = prioriVec, transMat = TransitionMat, 
                        restartP = restartP, maxIteration = 500, queryList = qu, 
                        deactivated = deactivated, allowRPModify = False)
                    ranking = np.asarray(weights.argsort()).reshape(-1)            
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

            # print()
            result = list(map(lambda x: wordList[x], qu))
            # ac = 100*sum(list(map(lambda x: wordList[x] in solution, qu)))/len(solution)                
            return(result)

if __name__ == "__main__":
    if(len(sys.argv) > 1):
        prCount = int(sys.argv[1])
    else:
        prCount = 1
    print("Using", prCount, "Processes")
    upto = 10
    filePerProcess = upto/prCount
    testerProcesses = [None]*prCount
    for thId in range(0,prCount):
        testerProcesses[thId] = AlgoTestFactory([int(thId*filePerProcess), int((thId + 1)*filePerProcess)], processID = thId, method=Method.word2vec)
        testerProcesses[thId].start()
    
    for p in testerProcesses:
        p.join()

    # print("Results: ")
    # # print(AlgoTestFactory.allAccuracies)
    # accuracies = [ac for acList in AlgoTestFactory.allAccuracies for ac in acList]
    # # print(accuracies)

    # accuracies = np.array(accuracies)
    # print("Mean: ", accuracies.mean())
    # print("Percentiles: ", np.percentile(accuracies, [0, 25, 50, 75, 100]))