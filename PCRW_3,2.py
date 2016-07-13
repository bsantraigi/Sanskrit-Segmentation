#===================================================================================
# IMPORT
#===================================================================================
from SktWsegRWR_utf8 import *
import pickle
import ProbData
from ProbModels import *
import multiprocessing
import math
import json
import pprint
import csv
from utilities import *
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd

#============================================================
# LOAD AND PREPROCESS MATRICES
#============================================================
# DO THIS IN PROBDATA_2

mat_lem2cng = pickle.load(open('../NewData/mat_lem2cng.p', 'rb'), encoding='utf-8')
mat_cng2lem = pickle.load(open('../NewData/mat_cng2lem.p', 'rb'), encoding='utf-8')
mat_tup2cng = pickle.load(open('../NewData/mat_tup2cng.p', 'rb'), encoding='utf-8')
mat_tup2lem = pickle.load(open('../NewData/mat_tup2lem.p', 'rb'), encoding='utf-8')
mat_selfLemCng_evidence = pickle.load(open('../NewData/mat_selfLemCng_evidence.p', 'rb'), encoding='utf-8')
mat_selfLemCLASS_evidence = pickle.load(open('../NewData/mat_selfLemCLASS_evidence.p', 'rb'), encoding='utf-8')

# Get count of each key in the matrices
mat_lem2cng_1D = {}
for lem in mat_lem2cng.keys():
    mainset = []
    for fs in mat_lem2cng[lem].values():
         mainset.extend(fs)
    mainset = set(mainset)
    mat_lem2cng_1D[lem] = len(mainset)
    

mat_cng2lem_1D = {}
for cng in mat_cng2lem.keys():
    mainset = []
    for fs in mat_cng2lem[cng].values():
         mainset.extend(fs)
    mainset = set(mainset)
    mat_cng2lem_1D[cng] = len(mainset)
    

mat_tup2cng_1D = {}
for tup in mat_tup2cng.keys():
    mainset = []
    for fs in mat_tup2cng[tup].values():
         mainset.extend(fs)
    mainset = set(mainset)
    mat_tup2cng_1D[tup] = len(mainset)
    

mat_tup2lem_1D = {}
for tup in mat_tup2lem.keys():
    mainset = []
    for fs in mat_tup2lem[tup].values():
         mainset.extend(fs)
    mainset = set(mainset)
    mat_tup2lem_1D[tup] = len(mainset)
    
    
#===================================================================================
# Load the skt/dcs
#===================================================================================
loaded_SKT = pickle.load(open('../Simultaneous_CompatSKT_10K.p', 'rb'))
loaded_DCS = pickle.load(open('../Simultaneous_DCS_10K.p', 'rb'))
    
#===================================================================================
# THE SCORES AND SHORT SCORE FUNCTIONS
#===================================================================================
def TheScores(q, v, c):
#     print(q, v, c)
    try:
        if type(q) == str:
            if type(v) == str:
                p1 = ProbData.fullCo_oc_mat[q][v]/mat_lem2cng_1D[q]
            elif type(v) == int:
                p1 = len(mat_lem2cng[q][str(v)])/mat_lem2cng_1D[q]
            elif type(v) == tuple:
                z = v[0] + '_' + str(v[1])
                p1 = len(mat_tup2lem[z][q])/mat_lem2cng_1D[q]
        elif type(q) == int:
            if type(v) == str:
                p1 = len(mat_cng2lem[str(q)][v])/mat_cng2lem_1D[str(q)]
            elif type(v) == tuple:
                p1 = len(mat_tup2cng[v[0] + '_' + str(v[1])][str(q)])/mat_cng2lem_1D[str(q)]
            elif type(v) == int:
                ia = ProbData.cng2index_dict[str(q)]
                ib = ProbData.cng2index_dict[str(v)]
                p1 = ProbData.cng2cngFullMat[ia, ib]/mat_cng2lem_1D[str(q)]
        elif type(q) == tuple:
            z = q[0] + '_' + str(q[1])
            if type(v) == str:
                p1 = len(mat_tup2lem[z][v])/mat_tup2lem_1D[z]
            elif type(v) == int:
                p1 = len(mat_tup2cng[z][str(v)])/mat_tup2lem_1D[z]
            
        if type(c) == str:
            if type(v) == str:
                p2 = ProbData.fullCo_oc_mat[v][c]/mat_lem2cng_1D[v]
            elif type(v) == int:
                p2 = len(mat_cng2lem[str(v)][c])/mat_cng2lem_1D[v]
            elif type(v) == tuple:
                z = v[0] + '_' + str(v[1])
                p2 = len(mat_tup2lem[z][c])/mat_tup2lem_1D[z]
        elif type(c) == int:
            if type(v) == str:
                p2 = len(mat_lem2cng[v][str(c)])/mat_lem2cng_1D[v]
            elif type(v) == int:
                ia = ProbData.cng2index_dict[str(v)]
                ib = ProbData.cng2index_dict[str(c)]
                p2 = ProbData.cng2cngFullMat[ia, ib]/mat_cng2lem_1D[str(v)]
            elif type(v) == tuple:
                z = v[0] + '_' + str(v[1])
                p2 = len(mat_tup2cng[z][str(c)])/mat_tup2lem_1D[z]
        elif type(c) == tuple:
            if type(v) == int:
                p2 = len(mat_tup2cng[c[0] + '_' + str(c[1])][str(v)])/mat_cng2lem_1D[str(v)]
            elif type(v) == str:
                p2 = len(mat_tup2lem[c[0] + '_' + str(c[1])][v])/mat_lem2cng_1D[v]
                
        return p1*p2
    except KeyError:
        return 0
    except UnboundLocalError:
        print('UnboundLocalError[TheScore]: ', fn, q, v, c)
        pass
    return 0

## COPY IT BACK
def ShortScore(q, c, code):
    try:
        if code == 'n-n':
            if ProbData.fullCo_oc_mat[q][c] == 0:
                return 0
            p1 = ProbData.w2w_samecng_fullmat[q][c]/ProbData.fullCo_oc_mat[q][c]
        else:
            s1 = mat_selfLemCLASS_evidence[q]['verbs'] - mat_selfLemCLASS_evidence[q][code]
            s2 = mat_selfLemCLASS_evidence[c][code]
            if len(s1) == 0:
                return 0
            p1 = len(s1 & s2)/len(s1)
    except KeyError:
        p1 = 0
    except UnboundLocalError:
        print('UnboundLocalError[TheScore]: ', fn, q, c, code)
        pass
    return p1


#===================================================================================
# Form NON-competitor dictionary - Query - Candidate Pairs
#===================================================================================

def Get_QCs():

    qc_pairs = {}
    for ni in range(len(nodeList)):
        qc_pairs[ni] = set(range(len(nodeList))) - set([ni])

    for cid in chunkDict.keys():
        # Neighbours
        for pos1 in chunkDict[cid].keys():
            for pos2 in chunkDict[cid].keys():
                if pos1 <= pos2:
                    nList1 = []
                    for ti1 in chunkDict[cid][pos1]:
                        for tup1 in tuplesMain[ti1]:
                            nList1.append(tup1[0])
                    nList2 = []
                    for ti2 in chunkDict[cid][pos2]:
                        for tup2 in tuplesMain[ti2]:
                            nList2.append(tup2[0])
                    nList1 = set(nList1)
                    nList2 = set(nList2)
                    for n1 in nList1:
                        qc_pairs[n1] = qc_pairs[n1] - nList1

                    for n2 in nList2:
                        qc_pairs[n2] = qc_pairs[n2] - nList2

                    if pos1 < pos2:
                        for n1 in nList1:
                            for n2 in nList2:
                                if not CanCoExist_sandhi(pos1, pos2, nodeList[n1][1], nodeList[n2][1]):
                                    qc_pairs[n1] = qc_pairs[n1] - set([n2])
                                    qc_pairs[n2] = qc_pairs[n2] - set([n1])
                                    
    return qc_pairs


#===================================================================================
# FIND THE SPECIEAL WORDS (REQUIRED IN CERTAIN PATHS) - TO BE USED/CHECKED AGAINST LATER
#===================================================================================

# Pass before converting the nodeList
def Splitter(nodeList):
    nouns = set()
    verbs = set()
    # adverbs = set() # IGNORE ADVERB
    gerund = set()
    ppp = set()
    ppa = set()
    inf = set()
    absol = set()

    
    for n in nodeList:
        try:
            if n[3] == -190:
                ppp.add(n[0])
            if n[3] == -200:
                ppp.add(n[0])
            if n[3] == -210: # Compare cng
                gerund.add(n[0]) # Add id
            if n[3] == -220:
                inf.add(n[0])
            if n[3] == -230:
                absol.add(n[0])


            # CONSIDERING ADV, GERUND, ABSOL ALSO IN VERB
            if n[3] < 0:
                verbs.add(n[0])
            if n[3] > 3:
                nouns.add(n[0])
        except TypeError:
            continue
    return (nouns, verbs, gerund, ppp, ppa, inf, absol)


#===================================================================================
# FIND EACH POSSIBLE QUERY-CANDIDATE PAIR AND CSV ALL THE AVAILABLE SCORES FOR THAT PAIR
#===================================================================================

csvf = open('pcrw_01.csv', 'w')
pcrw_csv = csv.writer(csvf)

hList = ['f', 'ln_lemma', 'rn_lemma', 'ln_cng', 'rn_cng']
for u in range(1,10):
    if u < 4:
        for v in [1,2,3]:
            for w in [1,2,3]:
                hList.append(u*100 + 10*v + w)
    else:
        hList.append(u*100)

hList.append('flag')

pcrw_csv.writerow(hList)

# Loop through files
for fn in list(loaded_SKT.keys()):
    skt  = loaded_SKT[fn]
    dcs  = loaded_DCS[fn]

    try:
        (chunkDict, lemmaList, wordList, revMap2Chunk, qu, cngList, verbs, tuplesMain) = SentencePreprocess(skt)
    except SentenceError:
        continue
    
    nodeList = [t for ts in tuplesMain for t in ts]
    sol, solNoPvb = GetSolutions(dcs)
    
    (nouns, verbs, gerund, ppp, ppa, inf, absol) = Splitter(nodeList)
    code_set = {'ger':(gerund, 500), 'absol':(absol, 600), 'ppp':(ppp, 700),
                'inf':(inf, 800), 'ppa':(ppa, 900)}
    
    qc_pairs = Get_QCs()
    
    #Change nodelist id, lemma, cng, word
    nodeList = [(t[0], t[2], t[3], (t[2], t[3])) for ts in tuplesMain for t in ts]

    # def Get_lvl_score(nodeList, nouns, verbs, adverbs, path):
    # 1 for lemma
    # 2 for cng
    # 3 for (lemma, cng)
    for ri in range(len(nodeList)):
        rn = nodeList[ri]

        for li in qc_pairs[ri]:
            scores = {}
            for u in range(1,4):
                for v in [1,2,3]:
                    for w in [1,2,3]:
                        scores[u*100 + 10*v + w] = 0
            scores[400] = 0
            scores[500] = 0
            scores[600] = 0
            scores[700] = 0
            scores[800] = 0
            scores[900] = 0
            if ri > li: # Otherwise it will be measured twice - duplicate entries in csv
                flag = 0
                ln = nodeList[li]
                if rn[1] in sol and ln[1] in sol:
                    flag = 1
                # FORM the paths here - OF LENGTH 3
                for mi in qc_pairs[ri]:
                    if mi != li  and mi in verbs:
                        mn = nodeList[mi]
                        for nt1 in [1, 2, 3]:
                            for nt2 in [1, 2, 3]:
                                for nt3 in [1, 2, 3]:
                                    nts = [nt1, nt2, nt3]
                                    if nts.count(3) > 1:
                                        continue
                                    scores[nt1*100 + nt2*10 + nt3] += TheScores(ln[nt1], mn[nt2], rn[nt3])

                # FORM the paths of length 2
                if ri in nouns and li in nouns:
                    # 400 is code for noun-noun paths
                    scores[400] += ShortScore(ln[1], rn[1], 'n-n')
                
                
                
                for key, pair in code_set.items():
                    spSet = pair[0]
                    code = pair[1]
                    if li in verbs and ri in spSet and li not in spSet:
                        scores[code] += ShortScore(ln[1], rn[1], key)

                rowlist = [fn, ln[1], rn[1], ln[2], rn[2]]
                for u in range(1,10):
                    if u < 4:
                        for v in [1,2,3]:
                            for w in [1,2,3]:
                                rowlist.append(scores[u*100 + 10*v + w])
                    else:
                        rowlist.append(scores[u*100])
                rowlist.append(flag)
                pcrw_csv.writerow(rowlist)



    #             print(rn, ln, flag)

            
# Get_lvl_score(nodeList, nouns, verbs, adverbs, [0,0,0])
csvf.close()
