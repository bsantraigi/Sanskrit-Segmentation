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

df_m1_pf = pd.read_csv('extras/jigyasa@pf.csv', header=None, names=['file', 'cid', 'pos', 'lemma'])
df_m1_10to20 = pd.read_csv('extras/jigyasa@skt.csv', header=None, names=['file', 'cid', 'pos', 'lemma'])
df_m1_upd = pd.read_csv('extras/jigyasa@upd.csv', header=None, names=['file', 'cid', 'pos', 'lemma'])

df_m1_pf['path'] = df_m1_pf.apply(lambda row: '../TextSegmentation/Pickle_Files/%d.p' % row['file'], axis=1)
df_m1_10to20['path'] = df_m1_10to20.apply(lambda row: '../TextSegmentation/corrected_10to20/%d.p' % row['file'], axis=1)
df_m1_upd['path'] = df_m1_upd.apply(lambda row: '../TextSegmentation/Updated Pickles/%d.p' % row['file'], axis=1)

# MERGE
df_m1_pf = df_m1_pf[(df_m1_pf['lemma'] != 'tad') & (df_m1_pf['lemma'] != 'vE')]
df_m1_10to20 = df_m1_10to20[(df_m1_10to20['lemma'] != 'tad') & (df_m1_10to20['lemma'] != 'vE')]
df_m1_upd = df_m1_upd[(df_m1_upd['lemma'] != 'tad') & (df_m1_upd['lemma'] != 'vE')]

frames = [df_m1_pf, df_m1_10to20, df_m1_upd]
df_m1 = pd.concat(frames)
print(df_m1.shape[0])
print(len(df_m1['file'].unique()))

# REMOVE DUPLICATE ENTRIES
df_m1 = df_m1.drop_duplicates('file')

df_m1['fileN'] = df_m1.apply(lambda row: '%d.p' % row['file'] ,axis=1)
df_m1['status'] = 'rejected'

files = np.array(df_m1.loc[df_m1['status'] == 'rejected', 'file'])

# FIX AND PICKLE SKTs=-

def run(start, finish):
    counter = 0
    tCounter = 0

    for fx in files[start:finish]:
        fn = '%d.p' % fx
        fn_new = fn + '2'
        if os.path.isfile('../TextSegmentation/CompatSKT_1/' + fn_new) or os.path.isfile('../TextSegmentation/CompatSKT/' + fn_new):
    #         print(fn_new, 'already there')
            continue
    #     print(fn_new)
        df_occur = df_m1[df_m1['file'] == fx]
        slemmas = {}
        
        fp = df_occur.iloc[0].loc['path']
    #     display(df_occur)
        try:
            skt, dcs = loadSentence(fn, fp)
        except IndexError:
            continue

        if(len(skt.chunk) != len(dcs.lemmas)):
            continue
            
        try:
            for index, row in df_occur.iterrows():
                ci = row['cid']
                pos = row['pos']
                dlemmas = [rom_slp(l) for l in dcs.lemmas[ci]]


                # FIXME: ADD A PROPER NAME TO WS
                lemma = row['lemma']
                li = dlemmas.index(lemma)
                if li == len(dlemmas) - 1:
                    # Last lemma
                    name = skt.chunk[ci].chunk_name[pos:]
                else:
                    qlemma = dlemmas[li + 1]
        #             print('Look for next lemma', qlemma)
                    qpos = -1
                    for px in skt.chunk[ci].chunk_words:
                        for ws in skt.chunk[ci].chunk_words[px]:
                            if qlemma in ws.lemmas:
                                qpos = px
                                break
                        if qpos != -1:
                            break
                    name = skt.chunk[ci].chunk_name[pos:qpos]
        #         print('Name', (pos,qpos), name)
                cng = dcs.cng[ci][li]
                ws = word_new(name)
                ws.lemmas.append(lemma)
                ws.forms.append(cng)
                if pos in skt.chunk[ci].chunk_words:
                    skt.chunk[ci].chunk_words[pos].append(ws)
                else:
                    skt.chunk[ci].chunk_words[pos] = []
                    skt.chunk[ci].chunk_words[pos].append(ws)
        except (IndexError, ValueError) as e:
            # display(df_occur)
            continue
        # NOW CHECK FULL COVERAGE
        goodFlag = FullCoverage(skt, dcs)
        if goodFlag:
    #         df_m1.loc[df_occur.index, 'status'] = 'accepted'
            counter += 1
            pickle.dump(skt, open('../TextSegmentation/CompatSKT/' + fn_new, 'wb'))
        tCounter += 1
        if(tCounter % 400 == 0):
            print('Chekpoint', counter, 'of', tCounter)
            
# 157804 FILES
if __name__ == '__main__':
    prs = []

    p = Process(target = run, args = (0, 12000))
    prs.append(p)
    p.start()

    p = Process(target = run, args = (12000, 24000))
    prs.append(p)    
    p.start()    
    
    p = Process(target = run, args = (24000, 36000))
    prs.append(p)    
    p.start()    

    p = Process(target = run, args = (36000, 48000))
    prs.append(p)
    p.start()    
    
    
    for pr in prs:
        pr.join()
        
    print('ALL COMPLETE')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
