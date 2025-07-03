import pandas as pd
import numpy as np
import json
from packing import is_box_inside_uld
      

df = pd.read_parquet("dimenstondata_KE.parquet")
df['pcslen'] = df['pcslen'] * 0.393701
df['pcswid'] = df['pcswid'] * 0.393701
df['pcshgt'] = df['pcshgt'] * 0.393701

#df.to_parquet("firstfifty.parquet")

# # List of target commodity item names
# target_items = [
#     'GARMENT', 'CLOTHES', 'SHIRT', 'FABRIC', 'TEXTILE', 'CARDIGAN',
#     'CLOTHSHOES', 'FOOTWEAR', 'SHOES', 'BAG', 'HANDBAG',
#     'COSMETICS', 'PERFUME', 'SKINCAREPR', 'BEAUTYPRO',
#     'JEWELRY', 'WATCH', 'GOLDBAR', 'GOLDJWL', 'FANCYJWL',
#     'TOY', 'BOOK', 'MAGAZINE', 'PAPERBOX', 'NOTEBOOK',
#     'COMPUTER', 'LAPTOP', 'SMARTPHONE', 'CAMERA', 'TABLETPC'
# ]

# # Filter the DataFrame
# filtered_df = df[df['cmditmnam'].isin(target_items)]

# # Save to a new CSV file
# filtered_df.to_parquet("LD1_items.parquet", index=False)

# print("File saved as 'LD1_items.parquet")

flight_df = df['dim_vol'] < 0.18

# Save to CSV
flight_df.to_parquet("flight_ICN_to_BUD.parquet", index=False)

print("Saved flight data to 'flight_ICN_to_BUD.csv'")




