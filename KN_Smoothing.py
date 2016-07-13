import pickle
import numpy as np
from utilities import pickleFixLoad
from DCS import *
from sentences import *
from romtoslp import rom_slp
from collections import defaultdict

fullCo_ocMat = pickleFixLoad('extras/all_dcs_lemmas_matrix_countonly.p')
unigram_counts = pickle.load(open('extras/counts_of_uniq_lemmas.p', 'rb'))

all_word_list = list(unigram_counts.keys())

context_count = defaultdict(int)
for word in fullCo_ocMat.keys():
    context_count[word] = len(fullCo_ocMat[word])

# Each bigram is repeated as a-b is same as b-a
total_context = int(sum(context_count.values())/2)
print(total_context)
print(len(context_count.keys()))

total_sentences = 441735
total_co_oc = sum([sum(fullCo_ocMat[word].values()) for word in fullCo_ocMat.keys()])
print('Total Co oc Count: ', total_co_oc)

def kn(word_a, word_b):
    delta = 0.5
    if word_a in fullCo_ocMat[word_b]:
        c_ab = (fullCo_ocMat[word_a][word_b] - delta)/total_co_oc
        normalization = delta*total_context/total_co_oc
        p_a = context_count[word_a]/total_context
        p_b = context_count[word_b]/total_context
        return c_ab + normalization*p_a*p_b
    else:
        normalization = delta*total_context/total_co_oc
        p_a = context_count[word_a]/total_context
        p_b = context_count[word_b]/total_context
        return normalization*p_a*p_b


ps = 0
count = 0
for first in all_word_list:
    count += 1
    if(count%1000 == 0):
        print("Checkpoint 1K")
    for second in all_word_list:
        ps += kn(first, second)
print("Sum of PS:", ps)




