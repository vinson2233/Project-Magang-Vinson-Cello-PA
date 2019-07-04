''' Kumpulan Fungsi untuk Recommendation System E-Commerce '''

#  --- Fungsi Filtering ---

def filter_consistent_view_user(df) :
  '''
  Mengambil user yang pernah melakukan sejumlah view ke item yang sama
  Input :
   - df (Pandas.DataFrame) : Dataset yang digunakan untuk modelling
  '''
  
  # Menentukan minimal view untuk filtering
  cons_min_view = 2
  
  # Hitung berapa jumlah view yang dilakukan tiap user ke tiap item
  cons_view_user = df.groupby(['userid','itemid']).count().reset_index()
  
  # Filter user yang memenuhi minimal jumlah view
  bool_cons_user = cons_view_user['rating'] >= cons_min_view
  filter_cons_user = list(cons_view_user[bool_cons_user]['userid'].unique())
  
  # Filter datasetnya
  df = df[df['userid'].isin(filter_cons_user)]
  
  # Tampilkan informasi tentang dataset setelah filtering
  print('Besar dataset setelah filter consistent min view user :',df.shape)
  print('Jumlah user unique :',len(df['userid'].unique()))
  print('Jumlah item unique :',len(df['itemid'].unique()))
  print('')
  
  return df


def filter_event_maksimal(df) :
  '''
  Mengambil event terakhir dari dataset untuk tiap user untuk tiap item
  Input :
   - df (Pandas.DataFrame) : Dataset yang digunakan untuk modelling
  '''
  
  ### Buat mapping untuk mengganti value rating buat filtering
  ### Buat mapping untuk mengganti value rating kembali ke semula
  map_rating = {'view':1, 'addtocart':2, 'transaction':3}
  map_rating_inv = {u:i for i,u in map_rating.items()}
  
  # Ganti nilai rating pada dataset sesuai dengan mapping
  df['rating'] = df['rating'].replace(map_rating)
  
  # Simpan rating maksimum dari datasetnya 
  df = df.groupby(['userid','itemid'], sort=False)['rating'].max().reset_index()
  
  # Ganti lagi nilai rating menjadi seperti semula
  df['rating'] = df['rating'].replace(map_rating_inv)
  
  # Tampilkan informasi tentang dataset setelah filtering
  print('Besar dataset setelah filter event maksimal :',df.shape)
  print('Jumlah user unique :',len(df['userid'].unique()))
  print('Jumlah item unique :',len(df['itemid'].unique()))
  print('')
  
  return df
  
  
def filter_buy_item(df) :
  '''
  Mengambil item yang pernah dibeli saja pada dataset
  Input :
   - df (Pandas.DataFrame) : Dataset yang digunakan untuk modelling
  '''
  ### Ambil data pembelian
  transaction = df[df['rating']=='transaction']
  
  # Ambil item yang pernah dibeli
  bool_transac = df['itemid'].isin(transaction['itemid'].unique())
  df = df[bool_transac]
  
  # Tampilkan informasi tentang dataset setelah filtering
  print('Besar dataset setelah filter buy item :',df.shape)
  print('Jumlah user unique :',len(df['userid'].unique()))
  print('Jumlah item unique :',len(df['itemid'].unique()))
  print('')
  
  return df
  

def filter_min_view_user(df) :
  '''
  Mengambil user yang pernah melakukan sejumlah view yang ditentukan
  Input :
   - df (Pandas.DataFrame) : Dataset yang digunakan untuk modelling
  '''
  
  # Menentukan minimal view untuk filtering
  min_view = 2
  
  # Hitung berapa jumlah view yang dilakukan tiap user
  view_user = df.groupby('userid').count().reset_index()
  
  # Filter user yang memenuhi minimal jumlah view
  bool_user = view_user['rating'] >= min_view
  filter_user = view_user[bool_user]['userid']
  
  # Filter datasetnya
  df = df[df['userid'].isin(filter_user)]
  
  # Tampilkan informasi tentang dataset setelah filtering
  print('Besar dataset setelah filter min view user :',df.shape)
  print('Jumlah user unique :',len(df['userid'].unique()))
  print('Jumlah item unique :',len(df['itemid'].unique()))
  print('')
  
  return df
