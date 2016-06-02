from SktWsegRWR_utf8 import *
from AlgoTestFactory import *


if __name__ == "__main__":
    if(len(sys.argv) > 1):
        prCount = int(sys.argv[1])
    else:
        prCount = 1
    print("Using", prCount, "Processes")
    upto = 1
    filePerProcess = upto/prCount
    testerProcesses = [None]*prCount

    for thId in range(0,prCount):
        testerProcesses[thId] = AlgoTestFactory([int(thId*filePerProcess), int((thId + 1)*filePerProcess)], processID = thId, method = Method.word2vec)
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