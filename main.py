def cek_input(df)
'''
Memeriksa apakah data input memiliki 3 kolom, dengan nama kolom [userid,itemid,rating] dan tipe dari rating harus numerik
Input :
	df : pandas DataFrame
Output :
	boolean
'''
#Memeriksa dimensi data
print("Jumlah observasi : {0} \n Jumlah kolom : {1}".format(df.shape[0],df.shape[1]))
if df.shape[1]==3 :
    print('PASS - Dimensi dataframe sesuai')
else :
    print('FAILURE - Pastikan data terdiri dari 3 kolom yakni userid,itemid,rating')
    boole = False
print("\n")

#Memeriksa nama kolom
if boole:
    print("Nama kolom di dataset : ",end = ' ')
    print(list(df.columns))
    if boole and sum(df.columns==["userid","itemid","rating"])==3 :
        print('PASS - Nama kolom sesuai')
    else :
        print('FAILURE - Pastikan nama kolom secara berurut adalah userid,itemid,rating')
        boole = False
    print("\n")
    if boole:
        #Memeriksa tipe rating
        if df.iloc[:,2].dtype != 'O':
            print('PASS - Data kolom ketiga(rating) sudah numerik')
        else :
            print("FAILURE - Data kolom ketiga merupakan String atau bukan integer, harap koreksi")
            boole = False
return boole

def filter(df)


def create_model_specific(df,algo)
