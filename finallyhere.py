
# coding: utf-8

# In[1]:

import os, sys
import pickle
from DCS import DCS
from sentences import word_new, chunks, sentences
from utilities import printProgress, validatePickleName, pickleFixLoad
import re
from romtoslp import rom_slp
import numpy as np
import csv


# In[ ]:

class dekho(object):
    def __init__(self,skt_path='15CS91R05/Bishal/TextSegmentation/Pickle_Files/',
                 dcs_path='15CS91R05/Bishal/Text Segmentation/DCS_pick/'):
       
        self.skt_path=skt_path
        self.dcs_path=dcs_path
        self.sentenceFiles=set(sorted(os.listdir(skt_path)))
        self.dcsFiles=set(sorted(os.listdir(dcs_path)))
        self.commonFiles=[]
        for sPickle in self.sentenceFiles:
            if sPickle in self.dcsFiles:
                sPickle = validatePickleName(sPickle)
                if sPickle != "":                
                    self.commonFiles.append(sPickle)

        self.commonFiles = list(set(self.commonFiles))

        csvfile = open('result_countlo.csv','w',encoding="utf8") 
        spamwriter = csv.writer(csvfile, delimiter=',')
        
        kitna=0
        for tFile in self.commonFiles:
            try:
                sentence_obj = pickleFixLoad(self.skt_path + tFile)
                dcs_obj= pickleFixLoad(self.dcs_path + tFile)
            except (KeyError, EOFError) as e:
                continue
            
            count=0
            for i in range(0,len(sentence_obj.chunk)):
                
                try:
                    list_skt=[]
                    chunk=sentence_obj.chunk[i]
                    for key,words in chunk.chunk_words.items():
                        for word in words:
                            list_skt.extend(word.lemmas)

                    list_dcs= dcs_obj.lemmas[i]

                    set_common=set(list_dcs).intersection(set(list_skt))
                    set_diff=set(list_dcs).difference(set_common)
                    count+=len(set_diff)
                    flag=0
                    
                except IndexError as e:
                    flag=1
                
                
                
            if(flag==0):
                spamwriter.writerow([sentence_obj.sent_id ,len(dcs_obj.lemmas),count])
            else:
                spamwriter.writerow([sentence_obj.sent_id,'Incomplete chunk',len(sentence_obj.chunk)-len(dcs_obj.lemmas)])
            
            kitna=kitna+1
            print(kitna)
                    
aan_de= dekho()

