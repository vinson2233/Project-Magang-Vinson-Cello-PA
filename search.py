# --- Fungsi search model untuk di tuning ---

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
