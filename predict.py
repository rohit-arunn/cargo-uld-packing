import pickle
import numpy as np
import pandas as pd

# ---- Step 1: Load the trained model ----
with open("fitness_predictor.pkl", "rb") as f:  # Change filename if different
    model = pickle.load(f)
    
df = pd.read_parquet("flight_ICN_to_BUD.parquet")

boxes = []
for idx, row in df.iterrows():
    box_id = (
        row['mstdocnum'], row['docowridr'], row['dupnum'],
        row['seqnum'], row['ratlinsernum'], row['dimsernum']
    ) 
    length = float(row['pcslen']) 
    width = float(row['pcswid'])
    height = float(row['pcshgt']) 
    numpcs = int(row['dim_numpcs'])
    weight = float(row['dim_wgt'])

    boxes.append({
        'box_id': box_id,
        'dimensions': (length, width, height),
        'number' : numpcs, 
        'weight': weight                          

    })




# Flatten features: [len, wid, ht, number, weight, len, wid, ht, number, weight, ...]
features = []
for box in boxes:
    length, width, height = box['dimensions']
    features.extend([length, width, height, box['number'], box['weight']])

# Convert to 2D array for scikit-learn
X_input = np.array(features).reshape(1, -1)

# Predict fitness
predicted_fitness = model.predict(X_input)[0]
print("Predicted Fitness:", predicted_fitness)
