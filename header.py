''' Kumpulan Fungsi untuk Recommendation System E-Commerce '''

#  --- Fungsi Filtering ---

def filter_consistent_view_user(df) :
  '''
  Mengambil user yang pernah melakukan sejumlah view ke item yang sama
  Input :
   - df (Pandas.DataFrame) : Dataset yang digunakan untuk modelling
  '''
  
  # Menentukan minimal view untuk filtering
  cons_min_view = 2
  
  # Hitung berapa jumlah view yang dilakukan tiap user ke tiap item
  cons_view_user = df.groupby(['userid','itemid']).count().reset_index()
  
  # Filter user yang memenuhi minimal jumlah view
  bool_cons_user = cons_view_user['rating'] >= cons_min_view
  filter_cons_user = list(cons_view_user[bool_cons_user]['userid'].unique())
  
  # Filter datasetnya
  df = df[df['userid'].isin(filter_cons_user)]
  
  # Tampilkan informasi tentang dataset setelah filtering
  print('Besar dataset setelah filter consistent min view user :',df.shape)
  print('Jumlah user unique :',len(df['userid'].unique()))
  print('Jumlah item unique :',len(df['itemid'].unique()))
  print('')
  
  return df


def filter_event_maksimal(df) :
  '''
  Mengambil event terakhir dari dataset untuk tiap user untuk tiap item
  Input :
   - df (Pandas.DataFrame) : Dataset yang digunakan untuk modelling
  '''
  
  ### Buat mapping untuk mengganti value rating buat filtering
  ### Buat mapping untuk mengganti value rating kembali ke semula
  map_rating = {'view':1, 'addtocart':2, 'transaction':3}
  map_rating_inv = {u:i for i,u in map_rating.items()}
  
  # Ganti nilai rating pada dataset sesuai dengan mapping
  df['rating'] = df['rating'].replace(map_rating)
  
  # Simpan rating maksimum dari datasetnya 
  df = df.groupby(['userid','itemid'], sort=False)['rating'].max().reset_index()
  
  # Ganti lagi nilai rating menjadi seperti semula
  df['rating'] = df['rating'].replace(map_rating_inv)
  
  # Tampilkan informasi tentang dataset setelah filtering
  print('Besar dataset setelah filter event maksimal :',df.shape)
  print('Jumlah user unique :',len(df['userid'].unique()))
  print('Jumlah item unique :',len(df['itemid'].unique()))
  print('')
  
  return df
  
  
def filter_buy_item(df) :
  '''
  Mengambil item yang pernah dibeli saja pada dataset
  Input :
   - df (Pandas.DataFrame) : Dataset yang digunakan untuk modelling
  '''
  ### Ambil data pembelian
  transaction = df[df['rating']=='transaction']
  
  # Ambil item yang pernah dibeli
  bool_transac = df['itemid'].isin(transaction['itemid'].unique())
  df = df[bool_transac]
  
  # Tampilkan informasi tentang dataset setelah filtering
  print('Besar dataset setelah filter buy item :',df.shape)
  print('Jumlah user unique :',len(df['userid'].unique()))
  print('Jumlah item unique :',len(df['itemid'].unique()))
  print('')
  
  return df
  

def filter_min_view_user(df) :
  '''
  Mengambil user yang pernah melakukan sejumlah view yang ditentukan
  Input :
   - df (Pandas.DataFrame) : Dataset yang digunakan untuk modelling
  '''
  
  # Menentukan minimal view untuk filtering
  min_view = 2
  
  # Hitung berapa jumlah view yang dilakukan tiap user
  view_user = df.groupby('userid').count().reset_index()
  
  # Filter user yang memenuhi minimal jumlah view
  bool_user = view_user['rating'] >= min_view
  filter_user = view_user[bool_user]['userid']
  
  # Filter datasetnya
  df = df[df['userid'].isin(filter_user)]
  
  # Tampilkan informasi tentang dataset setelah filtering
  print('Besar dataset setelah filter min view user :',df.shape)
  print('Jumlah user unique :',len(df['userid'].unique()))
  print('Jumlah item unique :',len(df['itemid'].unique()))
  print('')
  
  return df


#  --- Fungsi Tuning ---
def fit_model_surprise_basic(df,k) #Untested
	import time
  from surprise import Reader, Dataset 
  from surprise import KNNBasic,KNNWithMeans,SVD,SVDpp
  from surprise.model_selection import train_test_split
  from random import randint
  from sklearn.metrics import roc_auc_score
	start_time = time.time()

	reader = Reader(rating_scale=(0, 1))
	data_r = Dataset.load_from_df(data_filter[['userid', 'itemid', 'event']], reader)
	daftar_algo = {"KNNBasicUser":KNNBasic(sim_options = {"user_based":True}),
	               "KNNBasicItem":KNNBasic(sim_options = {"user_based":False}),
	               "KNNWithMeanItem":KNNWithMeans(sim_options = {"user_based":False}),
	               "KNNWithMeanUser":KNNWithMeans(sim_options = {"user_based":True}),
	               "SVD":SVD(),
	               "SVDnoBias":SVD(biased=False),
	               "SVDpp":SVDpp()}
	trainset, testset = train_test_split(data_r, test_size=0.25)
	algo = daftar_algo[k]
	algo.fit(trainset)
	#Buat prediksi
	predictions = algo.test(testset)
	pred = pd.DataFrame(predictions)
	pred.r_ui.replace({1.0:"transaction",0.0:"view"},inplace=True)
	from sklearn.metrics import roc_auc_score
	pred.r_ui.replace({"view":0,"addtocart":0,"transaction":1},inplace=True)
	auc = roc_auc_score(pred.r_ui,pred.est)
	end_time = time.time()
	return auc,end_time-start_time

def tune_KNNBasic(df, base) :
  '''
  Mencari parameter yang optimal untuk algoritma KNNBasic
  Input :
   - df (Pandas.DataFrame) : Dataset setelah mapping surprise
   - base (boolean) : Bool menentukan user based atau tidak
  Output :
   - best_algo (Model KNNBasic) : Model dengan parameter terbaik yang didapat dari grid search
  '''
  
  ### Potensi parameter 
  n_model = 5 # Menentukan jumlah model yang dibuat per similarity
  
  # Import package yang diperlukan
  from surprise import Reader, Dataset, KNNBasic
  from surprise.model_selection import train_test_split
  from random import randint
  from sklearn.metrics import roc_auc_score
  
  # Mengubah dataset ke dalam format Surprise
  reader = Reader(rating_scale=(0, 1))
  data_r = Dataset.load_from_df(df, reader)
  
  # Split dataset untuk training dan evaluasi model
  trainset, testset = train_test_split(data_r, test_size=.25)
  
  # Definisikan list untuk iterasi dan menyimpan value
  sim_choice = ['msd', 'cosine', 'pearson']
  list_sim = []
  list_min_k = []
  list_auc = []
  count = 1
  
  # Mulai iterasi
  for sim in sim_choice :
    for i in range(n_model) :
      
      ### Nilai parameter (rangenya dapat diubah)
      min_k = randint(5,15)

      # Mulai modelling
      algo = KNNBasic(min_k=min_k, sim_options={'name':sim, 'user_based':base}, verbose=False).fit(trainset)

      # Bikin dataframe hasil prediksi
      pred = pd.DataFrame(algo.test(testset))

      # Hitung nilai roc_auc
      auc = roc_auc_score(pred['r_ui'], pred['est'])

      # Menyimpan nilai paremeter
      list_sim.append(sim)
      list_min_k.append(min_k)
      list_auc.append(auc)
      
      # Kasihtau progress
      print('GridSearch',count,'Selesai :',auc)
      count = count + 1
  
  # Ambil parameter dengan nilai metrics terbaik
  params = pd.DataFrame({'min_k':list_min_k, 'sim':list_sim, 'auc':list_auc})
  params = params.sort_values('auc', ascending=False).head(1)
  
  # Buat model dengan parameter terbaik
  best_algo = KNNBasic(min_k=params['min_k']
                      ,sim_options={'name':params['sim'], 'user_based':base})
  
  return best_algo
  
  
  
