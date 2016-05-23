import os, sys
import pickle
from DCS import DCS
from sentences import word_new, chunks, sentences
from utilities import printProgress, validatePickleName, pickleFixLoad
import re
import numpy as np
import math
import pickle

np.set_printoptions(precision=2, suppress= True)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def Floyd_Warshall(nodes, transMat):
    # D = np.copy(1/(transMat + 0.00001))
    D = np.copy(transMat)
    D[D > 0] = 1
    D[D == 0] = np.Inf
    for i in nodes:
        D[i, i] = 0    
    for k in nodes:
        D_new = np.zeros(D.shape)
        for i in nodes:
            for j in nodes:
                D_new[i, j] = np.min([D[i,j], D[i, k] + D[k, j]])
        D = D_new

    return(D)


def GetDiaFromTransmat(nodes, transMat):
    sp = Floyd_Warshall(nodes, transMat)
    # sp[sp == np.Inf] = 0
    # print(nodes)
    sp = sp[nodes, :]
    sp = sp[:, nodes]
    dia = np.max(sp)
    # print(sp)
    return dia

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
                TransitionMat[row][col] = 0 #WHAT TO DO HERE??            
        
    MakeRowStochastic(TransitionMat)
    return TransitionMat


def MakeRowStochastic(matrix):
    rowCount = matrix.shape[0]
    for row in range(rowCount):
        s = np.sum(matrix[row, :])
        if(s!=0):
            matrix[row, :] = matrix[row, :]/s

def RWR(prioriVec, transMat, restartP, maxIteration, queryList, deactivated):
    """
    Run Random walk with restart
    until 
    we reach steady state or max iteration steps
    """
    # print(transMat)
    
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
    papMat = np.array(prioriVec)
    
    rVec = np.zeros((1, n))    
#     print(n)
    for i in queryList:
#         print(i)
        rVec[0, i] = 1/len(queryList)
    


    """
    Find Dia of graph using BFS
    """
    nodes = []
    for i in range(prioriVec.shape[1]):
        if(prioriVec[0,i] > 0):
            nodes.append(i)

    dia = GetDiaFromTransmat(nodes, transMat)
    # print( "Dia: ", dia)

    for i in range(maxIteration):        
#        print('shapes',papMat.shape,va.shape,prevMat.shape)
        newMat = (1 - restartP) * np.dot(papMat, transMat) + restartP * np.mat(rVec)
        diff = np.absolute(papMat - newMat)
        diffMax = np.argmax(diff)
        papMat = newMat
        if  abs(diffMax) < eps and maxIteration/10 > 10:
            break
                  
    return(papMat)


class AlgoTestFactory(object):
    def __init__(self, sentencesPath = '../TextSegmentation/Pickles/', dcsPath = '../Text Segmentation/DCS_pick/'):
        if(sys.version_info < (3, 0)):
            warnings.warn("\nPython version 3 or greater is required. Python 2.x is not tested.\n")

        """
           Folder @ sentencesPath contains pickle files for "sentences" object
           Folder @ path2 contains pickle files for the same sentences
           as in Folder @ sentencesPath but its DCS equivalent
        """
        self.sentencesPath = sentencesPath
        self.dcsPath = dcsPath


        """
        Get common dcs and sentences files
        """
        # print()        

        """
        Uncommet to refresh fileLists
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
        self.commonFiles = pickle.load(open("commonFiles.p", 'rb'))

        print("Current folder contains: ",len(self.commonFiles), " Files")

        self.algo = SktWsegRWR()

    def loadSentence(self, fName):
        try:
            sentenceObj = pickleFixLoad(self.sentencesPath + fName)
            dcsObj = pickleFixLoad(self.dcsPath + fName)
        except (KeyError, EOFError) as e:
            return None, None
        return(sentenceObj, dcsObj)

    def test(self):
        accuracies = []
        for f in self.commonFiles[1:10]:
        # f = self.commonFiles[33]
            sentenceObj, dcsObj = self.loadSentence(f)
            if(sentenceObj != None):
                result = self.algo.predict(sentenceObj, dcsObj)
                solution = dcsObj.dcs_chunks
                if result != None:
                    ac = 100*sum(list(map(lambda x: x in solution, result)))/len(solution)
                    accuracies.append(ac)
                    print(ac)
                    # print("Solution: ", solution)
                    # print("Prediction: ", result)
        # print("Results: ")
        # accuracies = np.array(accuracies)
        # print("Mean: ", accuracies.mean())
        # print("Percentiles: ", np.percentile(accuracies, [0, 25, 50, 75, 100]))


"""
Loads the Model_CBOW from file
Keeps a full list of train files and target sentences
Test on a single sentence of a set of sentences
"""
class SktWsegRWR(object):
    def __init__(self, modelFilePath = 'extras/NewModel_utf8.p'):
        """
        Load the CBOW pickle
        """
        # print()
        self.model_cbow = pickleFixLoad(modelFilePath)
        print("Loaded: ", self.model_cbow)


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
        ***{Word forms can also be used}
        """
        chunkDict = {}
        wordList = []
        revMap2Chunk = []
        qu = []
        
        cid = -1
        for chunk in sentenceObj.chunk:
            # print()
            cid = cid+1
            chunkDict[cid] = {}
            canBeQuery = 0
            if len(chunk.chunk_words.keys()) == 1:
                canBeQuery = 1
            # print("Analyzing ", chunk.chunk_name)
            for pos in chunk.chunk_words.keys():
                chunkDict[cid][pos] = []
                if(canBeQuery == 1) and (len(chunk.chunk_words[pos]) == 1):
                    canBeQuery = 2
                for word_sense in chunk.chunk_words[pos]:
                    if(len(word_sense.lemmas) > 0):
                        wordList.append(word_sense.lemmas[0])
                        k = len(wordList) - 1
                        chunkDict[cid][pos].append(k)
                        revMap2Chunk.append((cid, pos))
                        # print(pos, ": ", rom_slp(word_sense.names), word_sense.lemmas, word_sense.forms)
                        if canBeQuery == 2:
                            qu.append(k)
        
        if(len(wordList) == 0):
            return

        TransitionMat = getTransMat(wordList, self.model_cbow)
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

                    restartP = 0.045 # This is to be set based on graph diameter
                    weights = RWR(prioriVec, TransitionMat, restartP, 1000, qu, deactivated)
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
            return(result)
            # ac = 100*sum(list(map(lambda x: wordList[x] in solution, qu)))/len(solution)                

if __name__ == "__main__":
    tester = AlgoTestFactory()
    tester.test()
    

            
