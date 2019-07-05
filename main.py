import pandas as pd
import numpy as np
import filter
import mapping
import tuning
import search
import warnings
warnings.filterwarnings('ignore')
# --- Skema utama Recommendation System ---

# Fungsi untuk mengecek input apakah sudah sesuai untuk modelling
def cek_input(df) :
  '''
  Memeriksa apakah data input memiliki 3 kolom, dengan nama kolom [userid,itemid,event] dan tipe dari event harus string
  Input :
    df : pandas DataFrame
  Output :
    boolean yang menyatakan apakah data sudah memenuhi kriteria diatas
  '''
  boole = True
  # Memeriksa dimensi data
  print("Jumlah observasi : {0} \n Jumlah kolom : {1}".format(df.shape[0],df.shape[1]))
  if df.shape[1]==3 :
      print('PASS - Dimensi dataframe sesuai')
  else :
      print('FAILURE - Pastikan data terdiri dari 3 kolom yakni userid,itemid,event')
      boole = False
  print("\n")
  # Memeriksa nama kolom
  if boole:
      print("Nama kolom di dataset : ",end = ' ')
      print(list(df.columns))
      if boole and sum(df.columns==["userid","itemid","event"])==3 :
          print('PASS - Nama kolom sesuai')
      else :
          print('FAILURE - Pastikan nama kolom secara berurut adalah userid,itemid,event')
          boole = False
      print("\n")
      if boole:
          #Memeriksa tipe event
          if df.iloc[:,2].dtype == 'O':
              print('PASS - Data kolom ketiga(event) sudah tepat')
          else :
              print("FAILURE - Data kolom masih numerik, harap koreksi")
              boole = False
              
  return boole

# Fungsi untuk menampilkan deskripsi dataset secara umum
def describe_data(df) :
  '''
  Menanmpilkan karakteristik tentang dataset yang ingin dimodelkan
  Input : df(Pandas Data Frame)
  Output : -
  '''
  temp = df.event.value_counts()
  maximum= temp.max()
  minimum= temp.min()

  print("Jumlah user yang telah melakukan transaksi adalah {0} atau {1}% dari data".format(minimum,np.round(minimum/df.shape[0],3)))
  print("Jumlah user yang tidak melakukan transaksi adalah {0} atau {1}% dari data ".format(maximum,np.round(maximum/df.shape[0],3)))

  print("Jumlah user unik: {0}".format(len(df.userid.unique())))
  print("Jumlah item unik: {0}".format(len(df.itemid.unique())))

# Fungsi untuk melakukan personalized filter untuk Dataset Retail Rocket
def filter_retailrocket(df) :
  '''
  Melakukan filter dan mapping untuk mempersiapkan model
  sehingga cocok untuk modelling
  Input :
   - df (Pandas.DataFrame) : Dataset setelah dicek
  '''
  
  # Filtering
  df = filter.filter_consistent_view_user(df)
  df = filter.filter_event_maksimal(df)
  df = filter.filter_buy_item(df)
  df = filter.filter_min_event_user(df)
  
  # Mapping
  df = mapping.mapping_001(df)
  
  return df

# Fungsi untuk mencari model untuk dituning dari list model yang diinginkan
def search_model_specific(df,algo) :
  '''
  Untuk KNNBasic aja dulu
  Fungsi untuk mencari base model terbaik dari pilihan pilihan algoritma
  Input :
    df(Pandas DataFrame) : Dataframe yang ingin dimodelkan
  '''
  
  # Definisikan list untuk membant permodelan
  list_algo = []
  list_AUC  = []
  list_time = []
  algo_surprise = ["KNNBasicUser","KNNBasicItem","KNNWithMeanItem","KNNWithMeanUser","SVD","SVDnoBias","SVDpp"]
  algo_keras = ["AutoEncoder"]
  
  # Mulai iterasi
  for k in algo:
    list_algo.append(k)
    if k in algo_surprise :
      auc,runtime = search.fit_model_surprise_basic(df,k)
      list_AUC.append(auc)
      list_time.append(runtime)
    #Belum di implementasi
    if k in algo_keras : 
      print("Algoritma belum di implementasi")

  # Membuat output    
  hasil = pd.DataFrame({"Algoritma":list_algo,"Test_AUC":list_AUC,"Runtime":list_time})
  hasil.sort_values("Test_AUC",ascending=False,inplace=True)
  best_algo = hasil.iloc[0,0]
  
  return hasil,best_algo

# Fungsi buat ngetuning
def tuning_model(df, algo) :
  '''
  Melakukan tuning parameter untuk algoritma yang dipilih
  Input :
   - df (Pandas.DataFrame) : Dataset yang digunakan untuk modelling
   - algo (string) : Model yang akan dituning
  Output :
   - best_algo (Model) : Model dengan parameter yang sudah dituning
  '''
  
  # List algoritma tuning sesuai dengan model
  if algo == 'KNNBasicUser' :
      best_algo = tuning.tune_KNNBasic(df, True)
  elif algo == 'KNNBasicItem' :
      best_algo = tuning.tune_KNNBasic(df, False)
  elif algo == 'KNNWithMeansUser' :
      best_algo = tuning.tune_KNNWithMeans(df, True)
  elif algo == 'KNNWithMeansItem' :
      best_algo = tuning.tune_KNNWithMeans(df, False)
  elif algo == 'SVD' :
      best_algo = tuning.tune_SVD(df, True)
  elif algo == 'SVDnoBias' :
      best_algo = tuning.tune_SVD(df, False)
  elif algo == 'SVDpp' :
      best_algo = tuning.tune_SVDpp(df)
  
  return best_algo

# Fungsi fitting model
def fitting_model_surprise(df, algo) :
    
    from surprise import Dataset, Reader
    from surprise.model_selection import train_test_split
    # Mengubah dataset ke dalam format Surprise
    reader = Reader(rating_scale=(0, 1))
    data_r = Dataset.load_from_df(df, reader)
    
    # Buat dataset full
    # Split dataset untuk training dan evaluasi model
    trainset, testset = train_test_split(data_r, test_size=.000000001)
    
    # Fitting best model
    algo.train(trainset)
    
    return algo