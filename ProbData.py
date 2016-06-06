# IMPORT THE DATALOADER FILES FIRST
from utilities import *
from collections import *
print("Dataloader Started[Prob]...")

fullCo_oc_mat = pickleFixLoad('extras/all_dcs_lemmas_matrix_countonly.p')
unigram_counts = pickleFixLoad('extras/counts_of_uniq_lemmas.p')

cng2cngFullMat = np.mat(pickle.load(open('extras/all_dcs_cngs_matrix_countonly.p','rb'), encoding='latin1'))
cng2index_dict = pickleFixLoad('cng2index_dict.p')

w2w_samecng_fullmat = pickle.load(open('extras/lemmas_with_same_cngs_matrix_countonly.p', 'rb'), encoding=u'utf8')
samecng_unigram_counts = pickle.load(open('extras/dictionary_for_lemmas_with_same_cng.p', 'rb'), encoding=u'utf8')

v2c_fullMat = pickle.load(open('extras/verbs_vs_cngs_matrix_countonly.p', 'rb'), encoding=u'utf8')

print("Dataloader Finished[Prob]...")
