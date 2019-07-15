# Berisi fungsi untuk melakukan rekomendasi
import pandas as pd
import numpy as np

# Fungsi untuk mengambil jumlah item dibeli 
def count_transac_item(iid, full_transac) :
    '''
    Fungsi untuk mengambil jumlah transaksi item yang diinginkan
    Input 
     - iid (int) : Item yang terkait
     - full_transac (Pandas.DataFrame) : Dataset berisi transaksi saja
    '''
    
    count = len(full_transac[full_transac['itemid']==iid])
    return count
    
# Fungsi untuk mengambil jumlah item dilihat
def count_view_item(iid, full_view) :
    '''
    Fungsi untuk mengambil jumlah view item yang diinginkan
    Input 
     - iid (int) : Item yang terkait
     - full_view (Pandas.DataFrame) : Dataset berisi view saja
    '''
    
    count = len(full_view[full_view['itemid']==iid])
    return count

# Definisikan fungsi mencari aksi yang akan diambil dari hasil prediksi
# Perlu diganti skema klasifikasinya
def action_recommend(i) :
    '''
    Fungsi untuk menentukan aksi yang akan dilakukan dengan menggunakan batas yang diberikan
    best_threshold : Threshold untuk
    '''
    # Setting threshold
    best_threshold = 0.2
    
    # Tnentukan aksinya
    if i == 0 :
        action = 'Not Recommended'
    elif i > best_threshold :
        action = 'NBO'
    else :
        action = 'Recommended'

    return action

# Fungsi untuk mengeluarkan informasi tentang user
def user_info(id_user, full_transac) :
    '''
    Menampilkan history transaksi yang dilakukan oleh user yang ingin diberikan rekomendasi
    Input :
     - id_user (int) : User yang terkait
     - data_full (Pandas.DataFrame) : Dataset original yang dimasukkan ke dalam sistem
    '''
    N = 5 # Jumlah top item yang ingin ditampilkan
    
    # Filter event transaksi saja untuk user yang diinginkan
    dummy_2 = full_transac[full_transac['userid']==id_user]

    # Cari top N item beserta jumlahnya
    top_item = list(dummy_2.groupby('itemid').count()['event'].sort_values(ascending=False).index[:N]) 
    n_item = list(dummy_2.groupby('itemid').count()['event'].sort_values(ascending=False).values[:N]) 
    
    # Memuat dataframe informasi
    label_top = ['Item', 'Count Item']
    top_N = pd.DataFrame([top_item, n_item,], label_top).T
    top_N.fillna('-', inplace=True)
    
    # Print informasi tentang visitor
    print('Informasi tentang user',id_user)
    print('Banyak transaksi yang dilakukan :',len(dummy_2))
    print('Banyak item yang telah dibeli :',len(dummy_2['itemid'].unique()))
    print('')
    print('History Top item dibelinya')
    print(top_N)
    print('')
    
# Definisikan fungsi untuk memberikan rekomendasi
def hasil_recommendation(id_user, data_model, best_algo, list_item, full_transac, full_view) :
    '''
    Menampilkan hasil rekomendasi
    Input :
     - id_user (int) : User yang ingin diberi rekomendasi
     - data_model (Pandas.DataFrame) : Dataset untuk modelling
     - best_algo (Model) : Model yang sudah dituning dan difitting
     - list_item (Pandas.Series) : List item yang ada pada dataset modelling
     - full_transac (Pandas.DataFrame) : Dataset yang berisikan transaksi saja
     - full_view (Pandas.DataFrame) : Datase yang berisikan view saja
    '''
    
    # Tampilkan informasi tentang user
    user_info(id_user, full_transac)
    
    # Berapa banyak rekomendasi yang ingin diberikan
    n_NBO = 5
    n_recommend = 10
    
    # Ambil barang yang belum pernah dibeli visitor tersebut
    event_transaction = data_model[data_model['event']==1]
    dummy = event_transaction[event_transaction['userid']==id_user]
    bool_item = list_item.isin(dummy['itemid'].unique())
    rec_item = list_item[~bool_item]
    
    # Shuffle list item
    from random import sample
    rec_item = sample(list(rec_item), len(rec_item))
    
    # Lakukan prediksi buat mencari barang Recommend
    count_NBO = 0
    count_recommend = 0
    list_action = []
    pred = []
    
    for i in rec_item :
        pred_i = best_algo.predict(uid=id_user, iid=i)
        action = action_recommend(pred_i[3])
        
        if action == 'NBO' :
            count_NBO = count_NBO + 1
        elif action == 'Recommended' :
            count_recommend = count_recommend + 1
        
        # Simpan action dan prediksi ke dalam list
        pred.append(pred_i)
        list_action.append(action)
        
        # Hentikan looping jika sudah mencapai limit
        if count_NBO > n_NBO and count_recommend > n_recommend :
            break
    
    # Buat jadi dataframe dan sortir berdasarkan nilai 'est'
    pred = pd.DataFrame(pred)
    pred['action'] = list_action
    pred.sort_values('est',ascending=False, inplace=True)
    
    # Merge dengan dataset Full untuk mendapat categoryid dan parentid
    # pred = pred.merge(df_full[['itemid','categoryid','parentid']], left_on='iid', right_on='itemid')
    
    # Print Rekomendasi barang NBO
    NBO = pred[pred['action']=='NBO'].head(n_NBO)
    NBO['n_beli'] = [count_transac_item(item, full_transac) for item in NBO['iid']]
    NBO['n_view'] = [count_view_item(item, full_view) for item in NBO['iid']]
    print('Rekomendasi Barang NBO :')
    print(NBO[['iid', 'n_beli', 'n_view']])
    print('')
    
    # Print Rekomendasi barang Recommended
    rec = pred[pred['action']=='Recommended'].head(n_recommend)
    rec['n_beli'] = [count_transac_item(item, full_transac) for item in rec['iid']]
    rec['n_view'] = [count_view_item(item, full_view) for item in rec['iid']]
    print('Rekomendasi Barang biasa :')
    print(rec[['iid', 'n_beli', 'n_view']])
    
    return pred
