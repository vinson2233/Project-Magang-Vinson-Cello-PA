# --- Fungsi search model untuk di tuning ---

def fit_model_surprise_basic(df,k) :
  import time
  from surprise import Reader, Dataset 
  from surprise import KNNBasic,KNNWithMeans,SVD,SVDpp
  from surprise.model_selection import train_test_split
  from sklearn.metrics import roc_auc_score
  import pandas as pd
  start_time = time.time()

  reader = Reader(rating_scale=(0, 1))
  data_r = Dataset.load_from_df(df[['userid', 'itemid', 'event']], reader)
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
  pred.r_ui.replace({"view":0,"addtocart":0,"transaction":1},inplace=True)
  auc = roc_auc_score(pred.r_ui,pred.est)
  end_time = time.time()
  
  return auc,end_time-start_time

def fit_model_keras_basic(df,algo) :
  import time
  import mapping
  from sklearn.metrics import roc_auc_score
  import pandas as pd
  import preprocess_keras as pk
  import fit_autoencoder
  start_time = time.time()
  if algo == "AutoEncoder" :
    #Mapping data terlebih dahulu
    df,dict_user2idx,dict_item2idx = mapping.mapping_reset_index(df)
    A,A_test,mask,mask_test= pk.train_test_split_sparse(df,0.8)
    model = fit_autoencoder.fit_autoencoder(A,A_test)

    a = model.predict(A_test)
    result = pd.DataFrame(a)
    asd = pd.melt(result.reset_index(), id_vars=['index'], value_vars=list(pd.DataFrame(a).columns))
    mask = pd.DataFrame(mask_test.toarray())
    csd = pd.melt(mask.reset_index(), id_vars=['index'], value_vars=list(mask.columns))
    bool1 = csd['value'] == 1
    asd = asd[bool1]
    asd.columns = ['user','item','est']
    asd.index = list(range(len(asd)))
    test = pd.DataFrame(A_test.toarray())
    bsd = pd.melt(test.reset_index(), id_vars=['index'], value_vars=list(pd.DataFrame(A_test.toarray()).columns))
    bsd = bsd[bool1]
    bsd.columns = ['user','item','aktual']
    bsd.index = list(range(len(bsd)))
    score = roc_auc_score(bsd.aktual,asd.est)
  end_time = time.time()
  return score,end_time-start_time



