# -- File utama menjalankan Recommendation System --
import main
import pandas as pd

# Memulai program
print("Execute semua kode, harap tunggu")

# Mempersiapkan dataset
data = pd.read_csv("data_cello.csv")
data = data.iloc[:,[0,2,1]]
data.columns = ["userid","itemid","event"]

# Pisah antara dataset inputan dan dataset modelling
data_model = data.copy()

# Pengecekan dataset
main.cek_input(data_model)

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
hasil_search, model_tuning = main.search_model_specific(data_model, list_model)

# Tuning model dari hasil search
# model = main.tuning_model(data_model, model_tuning)
model = main.tuning_model(data_model,"KNNWithMeansUser")

# Fitting model
model = main.fitting_model_surprise(data_model, model)

# Menghitung probabilitas rekomendasi untuk tiap item tiap user
boolean = True
while boolean:
    
    # Menerima inputan
    input_user = 1150086
    pred = main.mockup_recommend(input_user, data, data_model, model)
    
    # Apakah ingin melakukan prediksi kembali ?
    input_decision = input("ingin melanjutkan ? [y/n]")
    if input_decision=="n":
        boolean = False
        
# Program selesai
print("Program selesai")
