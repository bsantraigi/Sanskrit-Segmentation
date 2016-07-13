import pandas as pd
from utilities import *
import os

def printAccuracies(path):    
    print(path)
    fNames = [validatePickleName(f) for f in os.listdir(path)]
    fNames = list(filter(len, fNames))
    accuracies = []
    for fName in fNames:
        act = pickle.load(open(path + fName, 'rb'))
        for f, pair in act.items():
            accuracies.append(pair[0]*100/pair[1])
            
    print(len(accuracies), 'files processed')
    accuracies = np.array(accuracies)
    print("Results: ")
    print("Mean: ", accuracies.mean())
    print("Percentiles: ", np.percentile(accuracies, [0, 25, 50, 75, 100]))
    print('Macro Accuracy %:', 100*np.sum(accuracies >= 95)/accuracies.shape[0])

if __name__ == "__main__":
    printAccuracies('.temp/pcrw_include_20and3/')
    printAccuracies('.temp/pcrw_include_22paths/')
    printAccuracies('.temp/pcrw_include_17paths/')
