import os, sys
import pickle
from DCS import DCS
from sentences import word_new, chunks, sentences
from utilities import printProgress, validatePickleName, pickleFixLoad
import re
from romtoslp import rom_slp



def RWR(prioriVec, simMat, restartP, restartNode, maxIteration):
    """
    Run Random walk with restart
    until 
    we reach steady state or max iteration steps
    """
    eps = 0.0000000000001    # the error difference, which should ideally be zero but can never be attained.
    n = len(prioriVec)
    papMat = np.array(prioriVec)
    for i in range(maxIteration):

        prevMat = papMat

#        print('shapes',papMat.shape,va.shape,prevMat.shape)
        papMat = ((1 - restart) *np.mat(ActorActor)*np.mat(papMat)) + restart*np.mat(va)
        diff = (np.mat(prevMat) - np.mat(papMat))
#        diff=np.transpose(diff)
        diff=np.absolute(diff)
        diffMax = np.argmax(diff)

        if  abs(diffMax) < eps and maxIteration/10 > 10:
            break
    return(papMat)


if __name__ == "__main__":
    if(sys.version_info < (3, 0)):
        warnings.warn("\nPython version 3 or greater is required. Python 2.x is not tested.\n")

    """
       Folder @ sentencesPath contains pickle files for "sentences" object
       Folder @ path2 contains pickle files for the same sentences
       as in Folder @ sentencesPath but its DCS equivalent
    """

    """-----------------------------------------------------------
    PART 
        1. CONFIGURATION (SET PATHS OF PICKLE FILES)
        2. LOAD THE Word2Vec MODEL
    -----------------------------------------------------------"""

    sentencesPath ='../TextSegmentation/Pickles/'
    dcsPath = '../Text Segmentation/DCS_pick/'

    sentenceFiles=set(sorted(os.listdir(sentencesPath)))
    dcsFiles=set(sorted(os.listdir(dcsPath)))

    """
    Get common dcs and sentences files
    """
    print()
    minSize = min(len(sentenceFiles), len(dcsFiles))
    commonFiles = []
    
    for sPickle in sentenceFiles:        
        if sPickle in dcsFiles:
            sPickle = validatePickleName(sPickle)
            if sPickle != "":                
                commonFiles.append(sPickle)

    commonFiles = list(set(commonFiles))

    print("Testing with: ",len(commonFiles), " Files")

    """
    Load the CBOW pickle
    """
    print()
    model_cbow = pickleFixLoad('extras/modelpickle10.p')
    print(model_cbow)

    """-----------------------------------------------------------
    PART 2
        1. LOAD A SENTENCE
        2. UNIFORM PRIOR PROB.
        3. SET QUERY NODE
        3. RUN RANDOM WALK
        4. CHOOSE WINNER
        5. MERGE QUERY NODES
        6. RERUN FROM 2
    -----------------------------------------------------------"""
    
    """
    Test with a sentence
    """

#     print(commonFiles[10])
    fName = commonFiles[99]
    sentenceObj = pickleFixLoad(sentencesPath + fName)
    dcsObj = pickleFixLoad(dcsPath + fName)
    
    print()
    print("------------SENTENCE--------------")
    print(sentenceObj.sentence)
    
    for chunk in sentenceObj.chunk:
        print()
        print("Analyzing ", chunk.chunk_name)
        for pos in chunk.chunk_words.keys():
            for word_sense in chunk.chunk_words[pos]:
                print(pos, ": ", word_sense.lemmas, word_sense.forms)
        
    

    
    print()
    print("------------DCS--------------")
    print(dcsObj.sentence)    
    solution = list(map(lambda x: rom_slp(x), dcsObj.dcs_chunks))
    print(solution)
    
