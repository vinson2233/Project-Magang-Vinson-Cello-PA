# Berisi fungsi untuk melakukan rekomendasi

# Fungsi untuk mengambil jumlah item dibeli 
def count_transac_item(iid) :
    '''
    Fungsi untuk mengambil jumlah transaksi item yang diinginkan
    Input 
    '''
    global full_transac
    
    count = len(full_transac[full_transac['itemid']==iid])
    return count
    
# Fungsi untuk mengambil jumlah item dilihat
def count_view_item(iid) :
    '''
    Fungsi untuk mengambil jumlah view item yang diinginkan
    '''
    global full_view
    
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
    best_threshold = 0.4
    
    # Tnentukan aksinya
    if i == 0 :
        action = 'Not Recommended'
    elif i > best_threshold :
        action = 'NBO'
    else :
        action = 'Recommended'

    return action

# Fungsi untuk mengeluarkan informasi tentang user
def user_info(user) :
    '''
    Menampilkan history transaksi yang dilakukan oleh user yang ingin diberikan rekomendasi
    '''
    global data_full
    N = 5
    
    # Filter event transaksi saja untuk user yang diinginkan
    # Ambil dari full_transac aja
    event_transaction_2 = data_full[data_full['event']=='transaction']
    dummy_2 = event_transaction_2[event_transaction_2['userid']==user]

    # Cari top N item beserta jumlahnya
    top_item = list(dummy_2.groupby('itemid').count()['event'].sort_values(ascending=False).index[:N]) 
    n_item = list(dummy_2.groupby('itemid').count()['event'].sort_values(ascending=False).values[:N]) 
    
    # Memuat dataframe informasi
    label_top = ['Item', 'Count Item']
    top_N = pd.DataFrame([top_item, n_item,], label_top).T
    top_N.fillna('-', inplace=True)
    
    # Print informasi tentang visitor
    print('Informasi tentang user',user)
    print('Banyak transaksi yang dilakukan :',len(dummy_2))
    print('Banyak item yang telah dibeli :',len(dummy_2['itemid'].unique()))
    print('')
    print('History Top item dibelinya')
    print(top_N)
    print('')
    
# Definisikan fungsi untuk memberikan rekomendasi
def recommend(visitor, df, algo) :
    global list_item, df_full
    n_NBO = 5
    n_recommend = 10
    
    # Ambil barang yang belum pernah dibeli visitor tersebut
    event_transaction = df[df['event']==5]
    dummy = event_transaction[event_transaction['visitorid']==visitor]
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
        pred_i = algo.predict(uid=visitor, iid=i)
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
    pred = pred.merge(df_full[['itemid','categoryid','parentid']], left_on='iid', right_on='itemid')
    
    # Rapihin dulu dataframenya
    pred.drop(columns='itemid', inplace=True)
    pred.drop_duplicates(subset='iid', inplace=True)
    
    # Print Rekomendasi barang NBO
    NBO = pred[pred['action']=='NBO'].head(n_NBO)
    NBO['n_beli'] = NBO.iid.apply(count_transac_item)
    NBO['n_view'] = NBO.iid.apply(count_view_item)
    print('Rekomendasi Barang NBO :')
    print(NBO[['iid', 'categoryid', 'parentid', 'n_beli', 'n_view']])
    print('')
    
    # Print Rekomendasi barang Recommended
    rec = pred[pred['action']=='Recommended'].head(n_recommend)
    rec['n_beli'] = rec.iid.apply(count_transac_item)
    rec['n_view'] = rec.iid.apply(count_view_item)
    print('Rekomendasi Barang biasa :')
    print(rec[['iid', 'categoryid', 'parentid', 'n_beli', 'n_view']])
    
    return pred


