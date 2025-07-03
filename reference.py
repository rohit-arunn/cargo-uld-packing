import pandas as pd

# Load your dataset
df = pd.read_parquet("LD1_items.parquet")  # or use read_csv(...) if it's CSV

# Filter for the desired flight
flight_df = df[(df['dim_vol'] < 0.006)]

# Save to CSV
flight_df.to_parquet("flight_ICN_to_BUD.parquet", index=False)

print("Saved flight data to 'flight_ICN_to_BUD.csv'")
