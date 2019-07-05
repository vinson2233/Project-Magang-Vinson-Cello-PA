#  --- Fungsi Filtering ---


# Buat Data Retail Rocket (E-Commerce)

# Fungsi filter user yang memiliki preferensi yang konsisten
def filter_consistent_view_user(df) :
  '''
  Mengambil user yang pernah melakukan sejumlah view ke item yang sama
  Input :
   - df (Pandas.DataFrame) : Dataset yang digunakan untuk modelling
  '''
  
  # Menentukan minimal view untuk filtering
  cons_min_view = 3
  
  # Hitung berapa jumlah view yang dilakukan tiap user ke tiap item
  cons_view_user = df.groupby(['userid','itemid']).count().reset_index()
  
  # Filter user yang memenuhi minimal jumlah view
  bool_cons_user = cons_view_user['event'] >= cons_min_view
  filter_cons_user = list(cons_view_user[bool_cons_user]['userid'].unique())
  
  # Filter datasetnya
  df = df[df['userid'].isin(filter_cons_user)]
  
  # Tampilkan informasi tentang dataset setelah filtering
  print('Besar dataset setelah filter consistent min view user :',df.shape)
  print('Jumlah user unique :',len(df['userid'].unique()))
  print('Jumlah item unique :',len(df['itemid'].unique()))
  print('')
  
  return df

# Fungsi filter ambil event yang maksimal aja
# Contohnya pada kasus pembelian berarti event view dan addtocartnya dibuang
def filter_event_maksimal(df) :
  '''
  Mengambil event terakhir dari dataset untuk tiap user untuk tiap item
  Input :
   - df (Pandas.DataFrame) : Dataset yang digunakan untuk modelling
  '''
  
  ### Buat mapping untuk mengganti value event buat filtering
  ### Buat mapping untuk mengganti value event kembali ke semula
  map_event = {'view':1, 'addtocart':2, 'transaction':3}
  map_event_inv = {u:i for i,u in map_event.items()}
  
  # Ganti nilai event pada dataset sesuai dengan mapping
  df['event'] = df['event'].replace(map_event)
  
  # Simpan event maksimum dari datasetnya 
  df = df.groupby(['userid','itemid'], sort=False)['event'].max().reset_index()
  
  # Ganti lagi nilai event menjadi seperti semula
  df['event'] = df['event'].replace(map_event_inv)
  
  # Tampilkan informasi tentang dataset setelah filtering
  print('Besar dataset setelah filter event maksimal :',df.shape)
  print('Jumlah user unique :',len(df['userid'].unique()))
  print('Jumlah item unique :',len(df['itemid'].unique()))
  print('')
  
  return df
  
# Fungsi filter item yang pernah dibeli saja
def filter_buy_item(df) :
  '''
  Mengambil item yang pernah dibeli saja pada dataset
  Input :
   - df (Pandas.DataFrame) : Dataset yang digunakan untuk modelling
  '''
  ### Ambil data pembelian
  transaction = df[df['event']=='transaction']
  
  # Ambil item yang pernah dibeli
  bool_transac = df['itemid'].isin(transaction['itemid'].unique())
  df = df[bool_transac]
  
  # Tampilkan informasi tentang dataset setelah filtering
  print('Besar dataset setelah filter buy item :',df.shape)
  print('Jumlah user unique :',len(df['userid'].unique()))
  print('Jumlah item unique :',len(df['itemid'].unique()))
  print('')
  
  return df
  
# Fungsi filter user yang setidaknya melakukan sejumlah event yang ditentukan
def filter_min_event_user(df) :
  '''
  Mengambil user yang pernah melakukan sejumlah event yang ditentukan
  Input :
   - df (Pandas.DataFrame) : Dataset yang digunakan untuk modelling
  '''
  
  # Menentukan minimal event untuk filtering
  min_event = 3
  
  # Hitung berapa jumlah event yang dilakukan tiap user
  event_user = df.groupby('userid').count().reset_index()
  
  # Filter user yang memenuhi minimal jumlah event
  bool_user = event_user['event'] >= min_event
  filter_user = event_user[bool_user]['userid']
  
  # Filter datasetnya
  df = df[df['userid'].isin(filter_user)]
  
  # Tampilkan informasi tentang dataset setelah filtering
  print('Besar dataset setelah filter min event user :',df.shape)
  print('Jumlah user unique :',len(df['userid'].unique()))
  print('Jumlah item unique :',len(df['itemid'].unique()))
  print('')
  
  return df
