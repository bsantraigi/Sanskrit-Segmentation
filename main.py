from AlgoTestFactory import *
import os

def printAccuracies(path):    
    fNames = [validatePickleName(f) for f in os.listdir(path)]
    fNames = list(filter(len, fNames))
    accuracies = []
    for fName in fNames:
        accuracies.append(pickle.load(open(path + fName, 'rb')))
    print(len(accuracies))
    accuracies = [ac for acList in accuracies for ac in acList]
    print(len(accuracies))
    accuracies = np.array(accuracies)
    print("Results: ")
    print("Mean: ", accuracies.mean())
    print("Percentiles: ", np.percentile(accuracies, [0, 25, 50, 75, 100]))

if __name__ == "__main__":
    altf1 = AlgoTestFactory([0, 100], 4, savePath="Combined_longrun_10K", storeAccuracies=True)
    altf1.run()
    printAccuracies('.temp/Combined_longrun_10K/')