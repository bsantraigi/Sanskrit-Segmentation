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
    def __init__(self, testRange, processCount, partition = [0, 0, 0, 1], sentencesPath = '../TextSegmentation/Pickles/', dcsPath = '../Text Segmentation/DCS_pick/', storeAccuracies = False, savePath = None, algoname = '3RWR'):
        
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
        self.algoname = algoname

        
        self.algo = SktWsegRWR(
            w2w_modelFunc = AlgoTestFactory.pb.get_w2w_mat, 
            t2t_modelFunc = AlgoTestFactory.pb.get_cng2cng_mat,
            v2c_modelFunc = AlgoTestFactory.pb.get_v2c_ranking,
            df_PCRW = ProbData.df_pcrw,
            sameCng_modelFunc = AlgoTestFactory.pb.get_w2w_samecng_mat,
            partition = partition
        )

    def processTarget(self, processID, start, finish):
        accuracies = {}
        fileCount = 0
        for f in list(AlgoTestFactory.loaded_SKT.keys())[start:finish]:
        # for f in list(AlgoTestFactory.undone)[start:finish]:
            sentenceObj = AlgoTestFactory.loaded_SKT[f]
            dcsObj = AlgoTestFactory.loaded_DCS[f]
            
            if(sentenceObj != None):
                try:
                    result = self.algo.predict(sentenceObj, dcsObj, algoname = self.algoname)
                except RuntimeWarning:
                    print('RuntimeWarning', f)
                

                if result != None:
                    solution, solution_no_pvb = GetSolutions(dcsObj)
                    ac = 0
                    for x in range(len(solution)):
                        if(solution[x] in result):
                            ac += 1

                    accuracies[f] = (ac, len(solution), len(result))
                    if not self.storeAccuracies and not self.processCount > 1:
                        print(ac)
            if len(accuracies) >= 100:
                savePath = self.savePath
                if(self.storeAccuracies):
                    if(savePath == None):
                        savePath = '.temp/' + str(processID) + 'f' + str(fileCount) + '_out.p'
                    else:
                        directory = '.temp/' + savePath + '/'
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        savePath = directory + str(processID) + 'f' + str(fileCount) + '_out.p'

                    pickle.dump(accuracies, open(savePath, 'wb'))
                    print('Process Finished ({1} accuracies and saved to disk){0}'.format(savePath, len(accuracies)))
                    accuracies = {}
                    fileCount += 1
                    
                    
        # save the rest of the accuracies
        savePath = self.savePath
        if(self.storeAccuracies) and len(accuracies) > 0:
            if(savePath == None):
                savePath = '.temp/' + str(processID) + 'f' + str(fileCount) + '_out.p'
            else:
                directory = '.temp/' + savePath + '/'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                savePath = directory + str(processID) + 'f' + str(fileCount) + '_out.p'

            pickle.dump(accuracies, open(savePath, 'wb'))
            print('Process Finished ({1} accuracies and saved to disk){0}'.format(savePath, len(accuracies)))
            accuracies = {}
            fileCount += 1

        if not self.storeAccuracies:
            accuracies = np.array(accuracies)
            print("Results: ")
            print("Mean: ", accuracies.mean())
            print("Percentiles: ", np.percentile(accuracies, [0, 25, 50, 75, 100]))
            print('Process Finished ({0} accuracies and not saved to disk)'.format(len(accuracies)))

    def run(self):
        print('Using', self.algoname)
        processCount = self.processCount
        start = self.testRange[0] 
        finish = self.testRange[1]
        if processCount == 1:
            self.processTarget(0, start, finish)
        else:
            step = math.ceil((finish - start)/processCount)
            processes = []
            for processID in range(0, processCount):
                print('Range', self.savePath, (processID, start + step*processID, min(step*(processID + 1), finish)))
                processes.append(multiprocessing.Process(target = self.processTarget, args = (processID, start + step*processID, min(start + step*(processID + 1), finish))))
            for p in processes:
                p.start()
            for p in processes:
                p.join()
                


print('ATF loading files...')
#AlgoTestFactory.loaded_SKT = pickle.load(open('../Simultaneous_CompatSKT_10K.p', 'rb'))
#AlgoTestFactory.loaded_DCS = pickle.load(open('../Simultaneous_DCS_10K.p', 'rb'))
AlgoTestFactory.loaded_SKT = pickle.load(open('../Simultaneous_CompatSKT.p', 'rb'))
AlgoTestFactory.loaded_DCS = pickle.load(open('../Simultaneous_DCS.p', 'rb'))

AlgoTestFactory.undone = list(pickle.load(open('.temp/undone13K.p', 'rb')))
print('ATF file loading [COMPLETE]...')

AlgoTestFactory.allAccuracies = []
AlgoTestFactory.pb = ProbModels(fullCo_oc_mat = ProbData.fullCo_oc_mat, unigram_counts = ProbData.unigram_counts,
               cng2cngFullMat = ProbData.cng2cngFullMat, cng2index_dict = ProbData.cng2index_dict,
               w2w_samecng_fullmat=ProbData.w2w_samecng_fullmat, samecng_unigram_counts=ProbData.samecng_unigram_counts,
               v2c_fullMat = ProbData.v2c_fullMat)
