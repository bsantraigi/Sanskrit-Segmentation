import numpy as np

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
