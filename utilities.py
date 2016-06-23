import sys as Sys
import pickle, re
import numpy as np
from romtoslp import *

# Print iterations progress
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    Sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    Sys.stdout.flush()
    if iteration == total:
        print("\n")

def pickleFixLoad(filename):
    return pickle.load(open(filename, 'rb'), encoding=u'utf-8')

def validatePickleName(fName):
    m = re.search("^[\w]*.p$", fName)
    if m != None:
        return(m.group(0))
    return("")

sandhiRules = pickle.load(open('extras/sandhiRules.p','rb'))    
def CanCoExist_sandhi(p1, p2, name1, name2):
    # P1 must be less than P2
    # Just send it in the proper order
    if(p1 < p2):
        overlap = max((p1 + len(name1)) - p2, 0)
        if overlap == 0:
            return True
        if overlap == 1 or overlap == 2:
            p1 = (name1[len(name1) - overlap:len(name1):], name2[0])
            p2 = (name1[-1], name2[0:overlap:])
            # print(name1, name2, p1, p2)
            # print(p1, p2)
            if p1 in sandhiRules:
                if(sandhiRules[p1]['length'] < len(p1[0]) + len(p1[1])):
                    return True
            if p2 in sandhiRules:
                if(sandhiRules[p2]['length'] < len(p2[0]) + len(p2[1])):
                    return True

    return False

def fix_w_new(word_new_obj):    
    dicto= { 'asmad':'mad','yuzmad':'tvad','ayam':'idam','agn':'agni','ya':'yad','eza':'etad',
             'tad':'sa','vd':'vid','va':'vE','-tva':'tva','ptta':'pitta','mahat':'mahant','ndra':'indra',
             'ap':'api','h':'hi','t':'iti','tr':'tri','va':'iva'}

    for i in range(0,len(word_new_obj.lemmas)):
        word_new_obj.lemmas[i]= rom_slp(word_new_obj.lemmas[i])
        word_new_obj.lemmas[i]= word_new_obj.lemmas[i].split('_')[0]
        
        if word_new_obj.lemmas[i] in dicto:
            # print('CHANGED', word_new_obj.lemmas[i])
            word_new_obj.lemmas[i]= dicto[word_new_obj.lemmas[i]]
                
        if(word_new_obj.lemmas[i]== 'yad'):
            if word_new_obj.names== 'yadi':
                word_new_obj.lemmas[i]= 'yadi'
                
    return(word_new_obj)

def loadSentence(fName, folderTag):
    try:
        dcsObj = pickleFixLoad('../Text Segmentation/DCS_pick/' + fName)           
        if folderTag == "C1020" :
            sentenceObj = pickleFixLoad('../TextSegmentation/corrected_10to20/' + fName)
        else:
            sentenceObj = pickleFixLoad('../TextSegmentation/Pickle_Files/' + fName)

    except (KeyError, EOFError, pickle.UnpicklingError) as e:
        return None, None
    return(sentenceObj, dcsObj)

preList = pickle.load(open('pvb.p', 'rb'))
def removePrefix(lemma):
    for pre in preList:
        m = re.match(pre, lemma)
        if(m != None):
            s = m.span()
            pat = lemma[s[0]:s[1]]
            return (lemma.split(pat)[1])
    return lemma

def Accuracy(prediction, dcsObj):
    solution = [rom_slp(c) for arr in dcsObj.lemmas for c in arr]
    solution_no_pvb = [removePrefix(l) for l in solution]

    ac = 0
    for x in range(len(solution)):
        if(solution[x] in prediction):
            ac += 1
        elif(solution_no_pvb[x] in prediction):
            ac += 1

    ac = 100*ac/len(solution)
    return ac