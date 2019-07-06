# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:01:47 2019

@author: vinso
"""

import main
import pandas as pd
print("Execute semua kode, harap tunggu")
data = pd.read_csv("data_cello.csv")
data = data.iloc[:,[1,3,2]]
data.columns = ["userid","itemid","event"]

# Membuat model
main.cek_input(data)
main.describe_data(data)
data = main.filter_retailrocket(data)
model = main.tuning_model(data,"KNNWithMeansUser")

# Fitting model
model = main.fitting_model_surprise(data, model)

boolean = True
while boolean:
    
    input_user = int(input("Masukkan user yang ingin diprediksi : "))
    input_item = int(input("Masukkan item yang ingin diprediksi : "))
    hasil = model.predict(input_user,input_item)
    print("Prediksi user-{0} ke item {1} adalah {2}".format(input_user,input_item,list(hasil)[3]))
    input_decision = input("ingin melanjutkan ? [y/n]")
    if input_decision=="n":
        boolean = False
print("Program selesai")