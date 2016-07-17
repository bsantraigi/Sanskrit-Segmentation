# IMPORT THE DATALOADER FILES FIRST
from utilities import *
from collections import *
import pandas as pd
import numpy as np
print("Dataloader Started[Prob]...")

fullCo_oc_mat = pickleFixLoad('extras/all_dcs_lemmas_matrix_countonly.p')
unigram_counts = pickleFixLoad('extras/counts_of_uniq_lemmas.p')

cng2cngFullMat = np.mat(pickle.load(open('extras/all_dcs_cngs_matrix_countonly.p','rb'), encoding = 'latin1'))
cng2index_dict = pickleFixLoad('cng2index_dict.p')

w2w_samecng_fullmat = pickle.load(open('extras/lemmas_with_same_cngs_matrix_countonly.p', 'rb'), encoding=u'utf-8')
samecng_unigram_counts = pickle.load(open('extras/dictionary_for_lemmas_with_same_cng.p', 'rb'), encoding=u'utf-8')

v2c_fullMat = pickle.load(open('extras/verbs_vs_cngs_matrix_countonly.p', 'rb'), encoding=u'utf-8')

print("Dataloader Finished[Prob]...")

print("Preprocessing PCRW Database...")

df_pcrw = pd.read_csv('pcrw_25_smooth.csv', usecols=['f', 'ln_lemma', 'rn_lemma', 'ln_cng', 'rn_cng', '111',
       '112', '113', '121', '122', '123', '131', '132', '133', '211', '212',
       '213', '221', '222', '223', '231', '232', '233', '311', '312', '313',
       '321', '322', '323', '331', '332', '333', '400', '500', '600', '700',
       '800', '900', '011', '022', '033', 'flag'])

df_pcrw = df_pcrw.drop(df_pcrw[np.isnan(df_pcrw.ln_cng) | np.isnan(df_pcrw.rn_cng)].index) # Add

df_pcrw.ln_cng = df_pcrw.ln_cng.astype(int) # Add
df_pcrw.rn_cng = df_pcrw.rn_cng.astype(int) # Add

print("Preprocessing PCRW Database [COMPLETE]...")
