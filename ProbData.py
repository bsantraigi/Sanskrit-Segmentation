# IMPORT THE DATALOADER FILES FIRST
from utilities import *
from collections import *
print("Dataloader Started[Prob]...")

fullCo_oc_mat = pickleFixLoad('extras/all_dcs_lemmas_matrix_countonly.p')
unigram_counts = pickle.load(open('extras/counts_of_uniq_lemmas.p', 'rb'))

cng2cngFullMat = np.mat(pickleFixLoad('extras/all_dcs_cngs_matrix_countonly.p'))
cng2index_dict = pickle.load(open('cng2index_dict.p', 'rb'))

print("Dataloader Finished[Prob]...")
