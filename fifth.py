# import pandas as pd

# df = pd.read_csv("dimenstondata_KE.csv")



import pandas as pd

# Read the parquet file (same as CSV, but with .parquet)
df = pd.read_parquet("flight_ICN_to_BUD.parquet")

print(df)



