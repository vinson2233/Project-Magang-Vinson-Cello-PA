# -- File utama menjalankan Recommendation System --
import main
import pandas as pd

# Memulai program
print("Execute semua kode, harap tunggu")

# Mempersiapkan dataset
data = pd.read_csv("data_cello.csv")
data = data.iloc[:,[1,3,2]]
data.columns = ["userid","itemid","event"]

# Pengecekan dataset
main.cek_input(data)

# Informasi singkat mengenai dataset
main.describe_data(data)

# Filter dataset general
# pct_user = 1 # Persentase top user dari dataset
# pct_item = 20 # Persentase top item dari dataset
# data = main.filter_general(data, pct_user, pct_item)

# Filter dataset personalized untuk kasus Retail Rocket
data = main.filter_retailrocket(data)

# Pemilihan model untuk dituning - Specific
list_model = ['KNNBasicUser','KNNBasicItem','KNNWithMeanItem','KNNWithMeanUser','SVD','SVDnoBias','SVDpp']
hasil_search, model_tuning = main.search_model_specific(data, list_model)

# Tuning model dari hasil search
# model = main.tuning_model(data, model_tuning)
model = main.tuning_model(data,"KNNWithMeansUser")

# Fitting model
model = main.fitting_model_surprise(data, model)

# Menghitung probabilitas rekomendasi untuk tiap item tiap user
boolean = True
while boolean:
    
    # Menerima inputan
    input_user = int(input("Masukkan user yang ingin diprediksi : "))
    input_item = int(input("Masukkan item yang ingin diprediksi : "))
    
    # Melakukan prediksi
    hasil = model.predict(input_user,input_item)
    
    # Menampilkan hasil prediksi
    print("Prediksi user-{0} ke item {1} adalah {2}".format(input_user,input_item,list(hasil)[3]))
    
    # Apakah ingin melakukan prediksi kembali ?
    input_decision = input("ingin melanjutkan ? [y/n]")
    if input_decision=="n":
        boolean = False
        
# Program selesai
print("Program selesai")
