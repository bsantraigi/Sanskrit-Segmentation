# Stats for using only word2vec Transition matrix
import numpy as np
import pickle
import os

def printAccuracies(path):
    fNames = os.listdir(path)
    fNames = list(filter(len, fNames))
    accuracies = {}
    for fName in fNames:
        act = pickle.load(open(path + fName, 'rb'))
        for f, pair in act.items():
            accuracies[f] = pair[0]*100/pair[1]

    print(len(accuracies), 'files processed')
    accuracies = np.array(list(accuracies.values()))
    print("Results: ")
    print("Mean: ", accuracies.mean())
    print("Percentiles: ", np.percentile(accuracies, [0, 25, 50, 75, 100]))
    print('Macro Accuracy %:', 100*np.sum(accuracies >= 95)/accuracies.shape[0])
    
def CheckFor(path):
    try:
        fNames = os.listdir(path)
        print(len(fNames), 'files collected')
    except FileNotFoundError:
        print('NO files collected')
    with open('nohup.out', 'r') as f:
        for line in f:
            if path in line and 'Process Finished' in line:
                print(line)
                
                
print('\n\n===============================================================================')
print('Method: 25Path          87K Files             Query Select: OLD Heuristic')
print('=================================================================================\n')
printAccuracies('.temp/25Path_1L/')

print('\n\n===============================================================================')
print('Method: 25Path          13K Files:RUNNING             Query Select: NEW Heuristic')
print('=================================================================================\n')
printAccuracies('.temp/25Path_13K_FAST/')

print('\n\n===============================================================================')
print('Method: 3RWR           87K Files             Query Select: OLD Heuristic')
print('=================================================================================\n')
printAccuracies('.temp/baseline1L/')
                
print('\n\n===============================================================================')
print('Method: 3RWR           13K Files             Query Select: NEW Heuristic')
print('=================================================================================\n')
printAccuracies('.temp/3RWR_13K_qless/')

print('\n\n===============================================================================')
print('Method: 25Path          1L Files:RUNNING             Query Select: NEW Heuristic')
print('=================================================================================\n')
printAccuracies('.temp/25Path_1L_QNEW/')

print('\n\n===============================================================================')
print('Method: 3RWR          1L Files             Query Select: NEW Heuristic')
print('=================================================================================\n')
printAccuracies('.temp/3RWR_20Kgoodf_qless1L/')











