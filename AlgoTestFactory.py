from SktWsegRWR_utf8 import *
import pickle
import ProbData
from ProbModels import *
import multiprocessing
import math
# badFiles = pickle.load(open('extras/badFiles.p', 'rb'))
# badFiles = []
# print(badFiles)
class AlgoTestFactory():
    def __init__(self, testRange, processCount, sentencesPath = '../TextSegmentation/Pickles/', dcsPath = '../Text Segmentation/DCS_pick/', storeAccuracies = False, savePath = None):
        
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
        self.processCount = processCount
        self.storeAccuracies = storeAccuracies
        self.savePath = savePath

        
        self.algo = SktWsegRWR(
            w2w_modelFunc = AlgoTestFactory.pb.get_w2w_mat, 
            t2t_modelFunc = AlgoTestFactory.pb.get_cng2cng_mat,
            v2c_modelFunc = AlgoTestFactory.pb.get_v2c_ranking,
            sameCng_modelFunc = AlgoTestFactory.pb.get_w2w_samecng_mat,
            partition=[0.25, 0.25, 0.25, 0.1]
        )

    def loadSentence(self, fName, folderTag):
        # print('File: ', fName)
        try:
            dcsObj = pickleFixLoad(self.dcsPath + fName)           
            if folderTag == "C1020" :
                sentenceObj = pickleFixLoad('../TextSegmentation/corrected_10to20/' + fName)
            else:
                sentenceObj = pickleFixLoad('../TextSegmentation/Pickle_Files/' + fName)

        except (KeyError, EOFError, pickle.UnpicklingError) as e:
            return None, None
        return(sentenceObj, dcsObj)

    def processTarget(self, processID, start, finish):
        accuracies = []
        for f in list(AlgoTestFactory.goodFileDict.keys())[start:finish]:
            sentenceObj, dcsObj = self.loadSentence(f, AlgoTestFactory.goodFileDict[f])
            if(sentenceObj != None):
                try:
                    result = self.algo.predict(sentenceObj, dcsObj)
                except RuntimeWarning:
                    print(f)
                

                if result != None:
                    ac = Accuracy(result, dcsObj)
                    accuracies.append(ac)
                    if not self.storeAccuracies and not self.processCount > 1:
                        print(ac)
                # else:
                    # print(f)
                    # print("BAD HIT")
        savePath = self.savePath
        if(self.storeAccuracies):
            if(savePath == None):
                savePath = '.temp/' + str(processID) + '_out.p'
            else:
                directory = '.temp/' + savePath + '/'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                savePath = directory + str(processID) + '_out.p'

            pickle.dump(accuracies, open(savePath, 'wb'))
            print('Process Finished ({1} accuracies and saved to disk){0}'.format(savePath, len(accuracies)))
        else:
            accuracies = np.array(accuracies)
            print("Results: ")
            print("Mean: ", accuracies.mean())
            print("Percentiles: ", np.percentile(accuracies, [0, 25, 50, 75, 100]))
            print('Process Finished ({0} accuracies and not saved to disk)'.format(len(accuracies)))

    def run(self):
        processCount = self.processCount
        start = self.testRange[0] 
        finish = self.testRange[1]
        if processCount == 1:
            self.processTarget(0, start, finish)
        else:
            step = math.ceil((finish - start)/processCount)
            processes = []
            for processID in range(0, processCount):
                processes.append(multiprocessing.Process(target = self.processTarget, args = (processID, start + step*processID, min(step*(processID + 1), finish))))
            for p in processes:
                p.start()
            for p in processes:
                p.join()
                



def Accuracy(prediction, dcsObj):
    solution = [rom_slp(c) for arr in dcsObj.lemmas for c in arr]
    ac = 100*sum(list(map(lambda x: x in prediction, solution)))/len(solution)
    return ac

AlgoTestFactory.goodFileDict = pickle.load(open('mergedGood_v3.p', 'rb'))
AlgoTestFactory.allAccuracies = []
AlgoTestFactory.pb = ProbModels(fullCo_oc_mat = ProbData.fullCo_oc_mat, unigram_counts = ProbData.unigram_counts,
               cng2cngFullMat = ProbData.cng2cngFullMat, cng2index_dict = ProbData.cng2index_dict,
               w2w_samecng_fullmat=ProbData.w2w_samecng_fullmat, samecng_unigram_counts=ProbData.samecng_unigram_counts,
               v2c_fullMat = ProbData.v2c_fullMat)