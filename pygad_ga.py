import pygad
import numpy as np
import uuid
import pandas as pd
import copy
import matplotlib.pyplot as plt
from plotly_main import create_container, create_box
import plotly.graph_objects as go
from plotly.offline import plot
from packing_ld1 import grid_based_pack, draw_box, draw_uld
from algorith import evaluate_fitness  # assumes this uses box order to evaluate i

# Load your data
df = pd.read_parquet("flight_ICN_to_BUD.parquet")

# Prepare boxes list
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
        'number': numpcs,
        'weight': weight
    })

solution_placements = {}

best_fitness = -np.inf  # Global variables
best_key = None  # Global variable to track best placement key 


       

def fitness_function(ga_instance, solution, solution_idx):
    global best_fitness, best_key
    order = list(map(int, solution))
    print("The order - ", order)
    print("The solution yes -", solution)
    ordered_boxes = [copy.deepcopy(boxes[i]) for i in order]
    print("Box order being evaluated:", [box['box_id'] for box in ordered_boxes])
    fitness_value, placed_boxes = evaluate_fitness(ordered_boxes)

    # Unique key using generation + index 
    generation = ga_instance.generations_completed
    key = f"G{generation}_S{solution_idx}"
    solution_placements[key] = placed_boxes

    # Save the best one
    if fitness_value > best_fitness:
        best_fitness = fitness_value
        best_key = key

    return fitness_value


# # Define fitness function
# def fitness_function(ga_instance, solution, solution_idx): 
#     order = list(map(int, solution))  # convert solution to list of ints   
#     ordered_boxes = [copy.deepcopy(boxes[i]) for i in order]  # preserve box data  
#     fitness_value, placed_boxes = evaluate_fitness(ordered_boxes)
#     unique_key = f"G{ga_instance.generations_completed}_S{solution_idx}_{uuid.uuid4().hex[:4]}"
#     solution_placements[unique_key] = placed_boxes 

#     print("order - ", order)
#     print("ordered boxes -", ordered_boxes)
#     print("unique key - ", unique_key)
#     #solution_placements[solution_idx] = placed_boxes
#     print("IDX for this one - ", solution_idx)
#     return fitness_value            

# GA parameters
num_genes = len(boxes)  # One gene per box (ordering problem)   
crossover_type="uniform"
mutation_type="swap" 
parent_selection_type="tournament"

best_packed_per_generation = []  # This stores actual packed positions
best_fitness_per_generation = []

def on_generation(ga_instance):
    generation = ga_instance.generations_completed
    solution, fitness, solution_idx = ga_instance.best_solution()

    key = f"G{generation}_S{solution_idx}"

    # This will only work if 'solution_placements[key]' was created in fitness function
    if key in solution_placements:
        packed_boxes = solution_placements[key]
        best_packed_per_generation.append(packed_boxes)
        best_fitness_per_generation.append(fitness)

        print(f"Generation {generation} | Fitness: {fitness} | Boxes placed: {len(packed_boxes)}")
    else:
        print(f"[Warning] Placement data not found for {key}")


ga_instance = pygad.GA(
    num_generations=3,
    num_parents_mating=3,
    fitness_func=fitness_function,
    sol_per_pop=3,
    num_genes=num_genes,
    gene_type=int,
    gene_space=list(range(len(boxes))),
    allow_duplicate_genes=False,
    on_generation=on_generation, 
    parent_selection_type=parent_selection_type,
    crossover_type = crossover_type,
    mutation_type = mutation_type,
    mutation_percent_genes=30
)




if __name__ == '__main__':

    # Run the GA 
    ga_instance.run()

    # Get the best solution  
    # solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # print("Best solution (box order):", solution)
    # print("Fitness value of the best solution:", solution_fitness)
    # print("Recorded solution indices:", solution_placements.keys())
    # print("Index:", solution_idx)

    # if ga_instance.best_solution_generation != -1:
    #     print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

    # Get index of best fitness across all generations
   # Find the best generation index
    best_gen_idx = best_fitness_per_generation.index(max(best_fitness_per_generation))

    # Get best packed result with positions
    returned_boxes = best_packed_per_generation[best_gen_idx]
    best_overall_fitness = best_fitness_per_generation[best_gen_idx]


    print("The best fitness across all is -", best_overall_fitness)





    #returned_boxes = solution_placements[best_key]
    for box in returned_boxes:
        print(box) 


    box_data = [
        (*box['position'], *box['dimensions'], box['colour'])
        for box in returned_boxes
    ]

    container_mesh, container_edges = create_container('lightgray')

    traces = [container_mesh, container_edges]

    for x, y, z, dx, dy, dz, color in box_data:
        traces.append(create_box(x, y, z, dx, dy, dz, color=color, opacity=1.0, name="Box"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[0, 100], backgroundcolor="white"),
            yaxis=dict(nticks=10, range=[0, 65], backgroundcolor="white"),
            zaxis=dict(nticks=10, range=[0, 70], backgroundcolor="white"),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Open in browser 
    plot(fig, filename='Optimized_with_PyGAD.html', auto_open=True)



