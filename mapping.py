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
