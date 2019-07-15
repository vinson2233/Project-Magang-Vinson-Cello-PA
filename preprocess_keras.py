def train_test_split_sparse(df,train_size):
	#Pastikan df disini indexnya sudah di reset
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.utils import shuffle
	from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
	import random
	random.seed( 30 )
	M = max(df.itemid) + 1
	N = max(df.userid) + 1
	# split into train and test
	df = shuffle(df)
	cutoff = int(train_size*len(df))
	df_train = df.iloc[:cutoff]
	df_test = df.iloc[cutoff:]

	A = lil_matrix((N, M))
	print("Calling: update_train")
	print(A.shape)

	def update_train(row):
	    nonlocal A
	    i = row.userid
	    j = row.itemid
	    A[i,j] = row.event
	    return A

	df_train.apply(update_train, axis=1)

	# mask, to tell us which entries exist and which do not
	A = A.tocsr()

	# test ratings dictionary
	A_test = lil_matrix((N, M))
	print("Calling: update_test")

	def update_test(row):
	    nonlocal A_test
	    i = int(row.userid)
	    j = int(row.itemid)
	    A_test[i,j] = row.event

	df_test.apply(update_test, axis=1)
	A_test = A_test.tocsr()
	mask = (A > 0) * 1.0
	mask_test = (A_test > 0) * 1.0

	return A, A_test, mask, mask_test
