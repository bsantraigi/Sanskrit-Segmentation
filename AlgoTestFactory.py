from SktWsegRWR_utf8 import *
import pickle
# badFiles = pickle.load(open('extras/badFiles.p', 'rb'))
# badFiles = []
# print(badFiles)
class AlgoTestFactory(multiprocessing.Process):
    def __init__(self, testRange, processID, method = Method.word2vec, sentencesPath = '../TextSegmentation/Pickles/', dcsPath = '../Text Segmentation/DCS_pick/', storeAccuracies = False, savePath = None):
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
        self.savePath = savePath

        

        self.algo = SktWsegRWR(method=self.method)

    def loadSentence(self, fName, folderTag):
        # print('File: ', fName)
        try:
            dcsObj = pickleFixLoad(self.dcsPath + fName)           
            if folderTag == "C1020" :
                sentenceObj = pickleFixLoad('../TextSegmentation/corrected_10to20/' + fName)
            else 
                sentenceObj = pickleFixLoad('../TextSegmentation/Pickle_Files/' + fName)

        except (KeyError, EOFError) as e:
            return None, None
        return(sentenceObj, dcsObj)

    def run(self):

        accuracies = []
        newBad = False

        for f in list(AlgoTestFactory.goodDict.keys())[self.testRange[0]:self.testRange[1]]:
            sentenceObj, dcsObj = self.loadSentence(f, goodDict[f])
            if(sentenceObj != None):
                try:
                    result = self.algo.predict(sentenceObj, dcsObj)
                except (ZeroDivisionError, IndexError) as e:
                    # SeeSentence(sentenceObj)
                    # print(f)
                    # newBad = True
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
        # if(newBad):
        #     # print('BAD BAD')
        #     pickle.dump(badFiles, open('extras/badFiles.p', 'wb'))
        #     # badFiles = pickle.load(open('extras/badFiles.p', 'rb'))
        # AlgoTestFactory.allAccuracies.append(accuracies)
        savePath = self.savePath
        if(self.storeAccuracies):
            if(savePath == None):
                savePath = '.temp/' + str(self.processID) + '_out.p'
            else:
                savePath = '.temp/' + savePath + '/' + str(self.processID) + '_out.p'

            pickle.dump(accuracies, open(savePath, 'wb'))
            print('Process Finished (accuracies saved to disk){0}'.format(
                savePath))
        else:
            print('Process Finished (accuracies not saved to disk)')


AlgoTestFactory.goodFileDict = pickle.load(open('mergedGood.p', 'rb'))
AlgoTestFactory.allAccuracies = []