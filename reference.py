import pandas as pd
import math

df = pd.read_parquet("LD1_items.parquet") 

df['pcslen'] = df['pcslen'].apply(math.ceil)
df['pcshgt'] = df['pcshgt'].apply(math.ceil)
df['pcswid'] = df['pcswid'].apply(math.ceil) 



flight_df = df[(df['dim_vol'] < 0.16) & (df['dim_numpcs'] == 3 )]      
flight_df['volume'] = flight_df['pcslen']*flight_df['pcshgt']*flight_df['pcswid'] 

df_sorted = flight_df.sort_values(by='volume', ascending=False) 
df_sorted = df_sorted[(df_sorted['volume'] < 5000)]


df_sorted.to_parquet("flight_ICN_to_BUD.parquet", index=False)

print("Saved flight data to 'flight_ICN_to_BUD.csv'")
#print(df_sorted['pcslen']) 

