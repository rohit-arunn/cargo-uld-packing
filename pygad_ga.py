import pygad
import numpy as np
import uuid
import pandas as pd
import copy
import matplotlib.pyplot as plt
from plotly_eg import create_container, create_box
import plotly.graph_objects as go
from plotly.offline import plot
from packing_ld1 import grid_based_pack, draw_box, draw_uld
from algorith import evaluate_fitness  # assumes this uses box order to evaluate

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

best_fitness = -np.inf  # Global variable
best_key = None  # Global variable to track best placement key
       

def fitness_function(ga_instance, solution, solution_idx):
    global best_fitness, best_key
    order = list(map(int, solution))
    print("The order - ", order)
    print("The solution yes -", solution)
    ordered_boxes = [copy.deepcopy(boxes[i]) for i in order]
    print("the ORDERED BOXES -", ordered_boxes)
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
crossover_type="two_points"
mutation_type="swap"
parent_selection_type="tournament"

def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])


ga_instance = pygad.GA(
    num_generations=3,
    num_parents_mating=2,
    fitness_func=fitness_function,
    sol_per_pop=3,
    num_genes=num_genes,
    gene_type=int,
    gene_space=list(range(len(boxes))),
    allow_duplicate_genes=False,
    on_generation=on_gen, 
    parent_selection_type=parent_selection_type,
    crossover_type = crossover_type,
    mutation_type = mutation_type,
    mutation_percent_genes=25
)



if __name__ == '__main__':

    # Run the GA 
    ga_instance.run()

    # Get the best solution  
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Best solution (box order):", solution)
    print("Fitness value of the best solution:", solution_fitness)
    print("Recorded solution indices:", solution_placements.keys())
    print("Index:", solution_idx)


    #print("The solution to everything - ", solution_placements)

    #returned_boxes = solution_placements[solution_idx] #[copy.deepcopy(boxes[i]) for i in solution] 

    returned_boxes = solution_placements[best_key]
    for box in returned_boxes:
        print(box)

#     BELOW IS WITH MATPLOTLIB

#     fig = plt.figure(figsize=(10, 7)) 
#     ax = fig.add_subplot(111, projection='3d') 

#     draw_uld(ax)

#     a = 0

#     for box in returned_boxes:     
#             x, y, z = box['position'] 
#             dx, dy, dz = box['dimensions']
#             color = box['colour']                         
#             draw_box(ax, x, y, z, dx, dy, dz, color)
#             a+=1 

#     print(a)


#     #Axis setup               
#     ax.set_xlabel('X (Width)')
#     ax.set_ylabel('Y (Depth)')
#     ax.set_zlabel('Z (Height)')
#     ax.set_xlim(0, 100)
#     ax.set_ylim(0, 70)
#     ax.set_zlim(0, 70)
#     ax.set_title('Packing inside ULD')
#     ax.view_init(elev=25, azim=35)
#     plt.tight_layout()
#     plt.show() 



#   BELOW IS WITH PLOTLY    


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
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Open in browser 
    plot(fig, filename='Optimized_with_PyGAD.html', auto_open=True)



