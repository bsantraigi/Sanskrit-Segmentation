import pandas as pd
import numpy as np
import os
import pickle
from sentences import *
from DCS import *
from utilities import *
from IPython.display import display
from multiprocessing import Process

# OPEN FILES

df_ri_pf = pd.read_csv('extras/replace_new@pf.csv', header=None, names=['sktLemma', 'dcsLemma', 'file', 'chunk'])
df_ri_10to20 = pd.read_csv('extras/replace_new@skt.csv', header=None, names=['sktLemma', 'dcsLemma', 'file', 'chunk'])
df_ri_upd = pd.read_csv('extras/replace_new@upd.csv', header=None, names=['sktLemma', 'dcsLemma', 'file', 'chunk'])

# SET PATH

df_ri_pf['path'] = df_ri_pf.apply(lambda row: '../TextSegmentation/Pickle_Files/%d.p' % row['file'], axis=1)
df_ri_10to20['path'] = df_ri_10to20.apply(lambda row: '../TextSegmentation/corrected_10to20/%d.p' % row['file'], axis=1)
df_ri_upd['path'] = df_ri_upd.apply(lambda row: '../TextSegmentation/Updated Pickles/%d.p' % row['file'], axis=1)

# MERGE
frames = [df_ri_pf, df_ri_10to20, df_ri_upd]
df_ri = pd.concat(frames)

# FILE LIST

files = df_ri['file'].unique()
files

# FIX AND PICKLE SKTs

def run(start, finish):
    counter = 0
    tCounter = 0
    for fx in files[start:finish]:
        fn = '%d.p' % fx
        df_occur = df_ri[df_ri['file'] == fx]
        slemmas = {}   
        
        fp = df_occur.iloc[0].loc['path']
        try:
            skt, dcs = loadSentence(fn, fp)
        except IndexError:
            continue
        if(len(skt.chunk) != len(dcs.lemmas)):
            continue
    #     display(df_occur)
        #=====================================================================
        # FOLLOWING BLOCK OF CODE FIXES SKT OBJECTS WITH THE REPLACE 'i' ISSUE
        #=====================================================================
        for ci in df_occur['chunk']:
            slemmas[ci] = {}
            for index, row in df_occur.loc[df_occur['chunk'] == ci].iterrows():
                slemmas[ci][row['sktLemma']] = row['dcsLemma']
    #         print(slemmas)
            chunk = skt.chunk[ci]
            for pos in chunk.chunk_words.keys():
                for wsi in range(len(chunk.chunk_words[pos])):
                    ws = chunk.chunk_words[pos][wsi]
                    for li in range(len(ws.lemmas)):
                        lemma = ws.lemmas[li]
                        if lemma in slemmas[ci]:
    #                         print('Yes', lemma)
                            skt.chunk[ci].chunk_words[pos][wsi].lemmas[li] = slemmas[ci][lemma]

        # NOW CHECK FULL COVERAGE
        goodFlag = FullCoverage(skt, dcs)
    #     print(goodFlag)
        if goodFlag:
            counter += 1
            pickle.dump(skt, open('../TextSegmentation/CompatSKT/%d.p2' % fx, 'wb'))
        tCounter += 1
        if(tCounter % 200 == 0):
            print('Chekpoint', counter, 'of', tCounter)
            
# 157804 FILES
if __name__ == '__main__':
    prs = []

    p = Process(target = run, args = (0, 40000))
    prs.append(p)
    p.start()

    p = Process(target = run, args = (40000, 80000))
    prs.append(p)    
    p.start()    
    
    p = Process(target = run, args = (80000, 120000))
    prs.append(p)    
    p.start()    

    p = Process(target = run, args = (120000, 160000))
    prs.append(p)
    p.start()    
    
    
    for pr in prs:
        pr.join()
        
    print('ALL COMPLETE')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
