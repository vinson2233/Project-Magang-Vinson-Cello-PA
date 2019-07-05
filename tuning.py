#  --- Fungsi Tuning ---

# Fungsi tuning KNNBasic
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
  import pandas as pd
  
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

# Fungsi tuning KNNWithMeans
def tune_KNNWithMeans(df, base) :
  '''
  Mencari parameter yang optimal untuk algoritma KNNWithMeans
  Input :
   - df (Pandas.DataFrame) : Dataset setelah mapping surprise
   - base (boolean) : Bool menentukan user based atau tidak
  Output :
   - best_algo (Model KNNWithMeans) : Model dengan parameter terbaik yang didapat dari grid search
  '''
  
  ### Potensi parameter 
  n_model = 3 # Menentukan jumlah model yang dibuat per similarity
  
  # Import package yang diperlukan
  from surprise import Reader, Dataset, KNNWithMeans
  from surprise.model_selection import train_test_split
  from random import randint
  from sklearn.metrics import roc_auc_score
  import pandas as pd
  
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
      algo = KNNWithMeans(min_k=min_k, sim_options={'name':sim, 'user_based':base}, verbose=False).fit(trainset)

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
  best_algo = KNNWithMeans(min_k=params['min_k']
                      ,sim_options={'name':params['sim'], 'user_based':base})
  
  return best_algo
  
# Fungsi tuning SVD (bias dan tak bias)
def tune_SVD(df, bias) :
  '''
  Mencari parameter yang optimal untuk algoritma SVD
  Input :
   - df (Pandas.DataFrame) : Dataset setelah mapping surprise
   - bias (boolean) : Bool menentukan model menggunakan term bias atau tidak
  Output :
   - best_algo (Model SVD) : Model dengan parameter terbaik yang didapat dari grid search
  '''
  
  ### Potensi parameter 
  n_model = 15 # Menentukan jumlah model yang dibuat
  
  # Import package yang diperlukan
  from surprise import Reader, Dataset, SVD
  from surprise.model_selection import train_test_split
  from random import randint, uniform
  from sklearn.metrics import roc_auc_score
  import pandas as pd
  
  # Mengubah dataset ke dalam format Surprise
  reader = Reader(rating_scale=(0, 1))
  data_r = Dataset.load_from_df(df, reader)
  
  # Split dataset untuk training dan evaluasi model
  trainset, testset = train_test_split(data_r, test_size=.25)
  
  # Definisikan list untuk iterasi dan menyimpan value
  list_n_factors = []
  list_epochs = []
  list_lr = []
  list_reg = []
  list_auc = []
  count = 1
  
  # Mulai iterasi
  for i in range(n_model) :

    ### Nilai parameter (rangenya dapat diubah)
    n_factors = randint(60,120)
    epochs = randint(20,40)
    lr = uniform(0.01,0.05)
    reg = uniform(0.0005,0.001)
    
    # Mulai modelling
    algo = SVD(n_factors=n_factors, n_epochs=epochs, biased=bias
              ,lr_all=lr, reg_all=reg, verbose=False).fit(trainset)

    # Bikin dataframe hasil prediksi
    pred = pd.DataFrame(algo.test(testset))

    # Hitung nilai roc_auc
    auc = roc_auc_score(pred['r_ui'], pred['est'])

    # Menyimpan nilai paremeter
    list_n_factors.append(n_factors)
    list_epochs.append(epochs)
    list_lr.append(lr)
    list_reg.append(reg)
    list_auc.append(auc)

    # Kasihtau progress
    print('GridSearch',count,'Selesai :',auc)
    count = count + 1
  
  # Ambil parameter dengan nilai metrics terbaik
  params = pd.DataFrame({'n_factors':list_n_factors, 'n_epochs':list_epochs, 'lr':list_lr
                        ,'reg':list_reg, 'auc':list_auc})
  params = params.sort_values('auc', ascending=False).head(1)
  
  # Buat model dengan parameter terbaik
  best_algo = SVD(n_factors=params['n_factors'], n_epochs=params['n_epochs']
                 ,biased=bias, lr_all=params['lr'], reg_all=params['reg'], verbose=False)
  
  return best_algo
  

# Fungsi tuning SVDpp
# Cukup lama banget
def tune_SVDpp(df) :
  '''
  Mencari parameter yang optimal untuk algoritma SVDpp
  Input :
   - df (Pandas.DataFrame) : Dataset setelah mapping surprise
  Output :
   - best_algo (Model SVDpp) : Model dengan parameter terbaik yang didapat dari grid search
  '''
  
  ### Potensi parameter 
  n_model = 15 # Menentukan jumlah model yang dibuat
  
  # Import package yang diperlukan
  from surprise import Reader, Dataset, SVDpp
  from surprise.model_selection import train_test_split
  from random import randint, uniform
  from sklearn.metrics import roc_auc_score
  import pandas as pd
  
  # Mengubah dataset ke dalam format Surprise
  reader = Reader(rating_scale=(0, 1))
  data_r = Dataset.load_from_df(df, reader)
  
  # Split dataset untuk training dan evaluasi model
  trainset, testset = train_test_split(data_r, test_size=.25)
  
  # Definisikan list untuk iterasi dan menyimpan value
  list_n_factors = []
  list_epochs = []
  list_lr = []
  list_reg = []
  list_auc = []
  count = 1
  
  # Mulai iterasi
  for i in range(n_model) :

    ### Nilai parameter (rangenya dapat diubah)
    n_factors = randint(20,40)
    epochs = randint(20,40)
    lr = uniform(0.01,0.05)
    reg = uniform(0.0005,0.001)
    
    # Mulai modelling
    algo = SVDpp(n_factors=n_factors, n_epochs=epochs
              ,lr_all=lr, reg_all=reg, verbose=False).fit(trainset)

    # Bikin dataframe hasil prediksi
    pred = pd.DataFrame(algo.test(testset))

    # Hitung nilai roc_auc
    auc = roc_auc_score(pred['r_ui'], pred['est'])

    # Menyimpan nilai paremeter
    list_n_factors.append(n_factors)
    list_epochs.append(epochs)
    list_lr.append(lr)
    list_reg.append(reg)
    list_auc.append(auc)

    # Kasihtau progress
    print('GridSearch',count,'Selesai :',auc)
    count = count + 1
  
  # Ambil parameter dengan nilai metrics terbaik
  params = pd.DataFrame({'n_factors':list_n_factors, 'n_epochs':list_epochs, 'lr':list_lr
                        ,'reg':list_reg, 'auc':list_auc})
  params = params.sort_values('auc', ascending=False).head(1)
  
  # Buat model dengan parameter terbaik
  best_algo = SVDpp(n_factors=params['n_factors'], n_epochs=params['n_epochs']
                 ,lr_all=params['lr'], reg_all=params['reg'], verbose=False)
  
  return best_algo
