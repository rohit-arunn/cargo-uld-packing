import numpy as np
import math
import pandas as pd
import random
from multiprocessing import Pool
import random
import matplotlib.pyplot as plt
from plotly_main import create_ld9, create_box
import plotly.graph_objects as go
from plotly.offline import plot
from itertools import permutations
from ga_main import GeneticAlgorithm

df = pd.read_parquet("flight_ICN_to_BUD.parquet")

             
# boxes = []
# for idx, row in df.iterrows():
#     box_id = (
#         row['mstdocnum'], row['docowridr'], row['dupnum'],
#         row['seqnum'], row['ratlinsernum'], row['dimsernum']
#     ) 
#     length = float(row['pcslen']) 
#     width = float(row['pcswid'])
#     height = float(row['pcshgt']) 
#     numpcs = int(row['dim_numpcs'])
#     weight = float(row['dim_wgt'])

#     boxes.append({
#         'box_id': box_id,
#         'dimensions': (length, width, height),
#         'number' : numpcs, 
#         'weight': weight                          

#     })

boxes = [{'box_id': 36, 'dimensions': (30, 0, 10), 'number': 10}]


if __name__ == '__main__':
    ga = GeneticAlgorithm(
    boxes=boxes,
    uld_checker = "LD9",   
    container_dims=(125, 88, 64), 
    grid_step=1,
    pop_size=8,
    generations=10,
    mutation_rate=0.3,
    processes=8
)
    best_chromosome, generations_plot, fitness_history = ga.run()
    print('Best chromosome rotations:')  
    for g in best_chromosome:
        print(g['box_id'], g['dimensions'])
    
        box_data = [
        (*box['position'], *box['dimensions'], box['colour'])
        for box in best_chromosome
    ]

    a = len(box_data)
    print("No: of boxes plotted -", a)

    container_mesh, container_edges = create_ld9('lightgray')

    traces = [container_mesh, container_edges]

    for x, y, z, dx, dy, dz, color in box_data:
        traces.append(create_box(x, y, z, dx, dy, dz, color=color, opacity=1.0, name="Box"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[0, 140], backgroundcolor="white"),
            yaxis=dict(nticks=10, range=[0, 100], backgroundcolor="white"),
            zaxis=dict(nticks=10, range=[0, 70], backgroundcolor="white"),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Open in browser           
    
    plot(fig, filename='Optimized_packing_ld9.html', auto_open=True)
    
    plt.plot(generations_plot, fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.show()



