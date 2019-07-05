#Apakah akan berubah
def cek_input(df)
'''
Memeriksa apakah data input memiliki 3 kolom, dengan nama kolom [userid,itemid,event] dan tipe dari event harus string
Input :
	df : pandas DataFrame
Output :
	boolean yang menyatakan apakah data sudah memenuhi kriteria diatas
'''
#Memeriksa dimensi data
print("Jumlah observasi : {0} \n Jumlah kolom : {1}".format(df.shape[0],df.shape[1]))
if df.shape[1]==3 :
    print('PASS - Dimensi dataframe sesuai')
else :
    print('FAILURE - Pastikan data terdiri dari 3 kolom yakni userid,itemid,event')
    boole = False
print("\n")

#Memeriksa nama kolom
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
if boole:
	print("Data sudah siap untuk dimodelkan")
else :
	print("Data harap diperbaiki")
return boole

def describe_data(df,verbose)
'''
Menanmpilkan karakteristik tentang dataset yang ingin dimodelkan
Input : df(Pandas Data Frame)
Output : -
'''
	if verbose:
		print("Pastikan data sudah memenuhi kriteria fungsi cek_input")
	print(df.head())
	print("Jumlah user : {0}".format(df.shape[0]))
	print("Jumlah user unik : {0} \n Jumlah item unik : {1}".format(len(df.userid.unique),len(df.itemid.unique)))
	
	print("Dari dataset, terdapat {0}%")

def filter(df)

def search_model_specific(df,algo)
'''
Untuk KNNBasic aja dulu
Fungsi untuk mencari base model terbaik dari pilihan pilihan algoritma
Input :
	df(Pandas DataFrame) : Dataframe yang ingin dimodelkan
'''
list_algo = 0 
list_AUC  = 0
list_time = 0
algo_surprise = ["KNNBasicUser","KNNBasicItem","KNNWithMeanItem","KNNWithMeanUser","SVD","SVDnoBias","SVDpp"]
algo_keras = ["AutoEncoder"]
for k in algo:
	list_algo.append(k)
	if k in algo_surprise :
		auc,runtime = fit_model_surprise_basic(df,k)
		list_AUC.append(auc)
		list_time.append(runtime)
	#Belum di implementasi
	if k in algo_lightfm : 
		print("Algoritma belum di implementasi")

hasil = pd.DataFrame{"Algoritma":list_algo,"Test_AUC":list_AUC,"Runtime":list_time}
hasil.sort_values("Test_AUC",ascending=False,inplace=True)
best_algo = hasil.iloc[0,1]
return hasil,best_algo
