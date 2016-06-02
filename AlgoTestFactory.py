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
        # print('File: ', fName)
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
        newBad = False
        for f in AlgoTestFactory.commonFiles[self.testRange[0]:self.testRange[1]]:
        # f = self.commonFiles[33]
            # if(f in badFiles):
            #     # print(f, 'is a badfile')
            #     continue
            sentenceObj, dcsObj = self.loadSentence(f)
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


AlgoTestFactory.commonFiles = pickle.load(open("commonFiles.p", 'rb'))
print("Current folder contains: ",len(AlgoTestFactory.commonFiles), " Files")
AlgoTestFactory.allAccuracies = []