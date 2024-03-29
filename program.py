# -- File utama menjalankan Recommendation System --
import main
import pandas as pd
import sys

# Memulai program
print("Execute semua kode, harap tunggu")

# Mempersiapkan dataset
print("Membaca dataset...")
data = pd.read_csv("view_berkualitas.csv")
print("Data set berhasil dibaca \n")
data = data.iloc[:,[1,2,3]]
data.columns = ["userid","itemid","event"]

# Pisah antara dataset inputan dan dataset modelling
data_model = data.copy()

# Pengecekan dataset
cek = main.cek_input(data_model)
if not cek :
  print("Harap perbaiki data terlebih dahulu")
  sys.exit()
else :
  print("Data sudah siap diolah :)")

# Informasi singkat mengenai dataset
main.describe_data(data_model)

# Filter dataset general
# pct_user = 1 # Persentase top user dari dataset
# pct_item = 20 # Persentase top item dari dataset
# data_model = main.filter_general(data_model, pct_user, pct_item)

# Filter dataset personalized untuk kasus Retail Rocket
data_model = main.filter_retailrocket(data_model)

# Pemilihan model untuk dituning - Specific
list_model = ['KNNBasicUser','KNNBasicItem','KNNWithMeanItem','KNNWithMeanUser','SVD','SVDnoBias','SVDpp']
#list_model = ["AutoEncoder"]
hasil_search, model_tuning = main.search_model_specific(data_model, list_model)
print("Berikut hasil pencarian model terbaik dengan base configuration")
print(hasil_search)
# Tuning model dari hasil search
# model = main.tuning_model(data_model, model_tuning)
print("Berdasarkan nilai AUC, model yang akan dituning adalah ",model_tuning)

model = main.tuning_model(data_model,"KNNWithMeansUser")

# Fitting model
model = main.fitting_model_surprise(data_model, model)

# Menghitung probabilitas rekomendasi untuk tiap item tiap user
boolean = True
while boolean:
    
    # Menerima inputan
    input_user = int(input("Masukkan user yang ingin diperiksa"))
    pred = main.mockup_recommend(input_user, data, data_model, model)
    
    # Apakah ingin melakukan prediksi kembali ?
    input_decision = input("ingin melanjutkan ? \n [y/n]")
    if input_decision=="n":
        boolean = False
        
# Program selesai
print("Program selesai")
