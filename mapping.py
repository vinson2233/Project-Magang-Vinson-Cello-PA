# --- Fungsi Mapping ---

# Fungsi untuk melakukan mapping pada 3 event (view-addtocart-transaction)
# Sehingga menjadi kasus binary classification
def mapping_001(df) :
  '''
  Mapping event agar sesuai untuk modelling surprise
  Digunakan pada dataset Retail Rocket
  Input :
   - df (Pandas.DataFrame) : Dataset yang digunakan untuk modelling
  '''
  
  # Membuat mapping 0-0-1
  map_001 = {'view':0, 'addtocart':0, 'transaction':1}
  
  # Lakukan mapping 
  df['event'] = df['event'].replace(map_001)
  
  # Check berhasil atau tidak
  print('Mapping 0-0-1 selesai')
  
  return df

def mapping_reset_index(df) :
  # create a mapping for user ids
  unique_user_ids = set(df.userid.values)
  user2idx = {}
  count = 0
  for user_id in unique_user_ids:
    user2idx[user_id] = count
    count += 1
  df.userid = df["userid"].map(user2idx)

  # create a mapping for item ids
  unique_item_ids = set(df.itemid.values)
  item2idx = {}
  count = 0
  for item_id in unique_item_ids:
    item2idx[item_id] = count
    count += 1
  df.itemid = df["itemid"].map(item2idx)

  return df,user2idx,item2idx
