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
