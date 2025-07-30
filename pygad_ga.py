import pygad
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from packing_ld1 import grid_based_pack, draw_box, draw_uld
from genetic_alg_2 import evaluate_fitness  # assumes this uses box order to evaluate

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

# Define fitness function
def fitness_function(ga_instance, solution, solution_idx):
    order = list(map(int, solution))  # convert solution to list of ints
    ordered_boxes = [copy.deepcopy(boxes[i]) for i in order]  # preserve box data
    fitness_value, placed_boxes = evaluate_fitness(ordered_boxes)
    solution_placements[solution_idx] = placed_boxes
    return fitness_value

# GA parameters
num_genes = len(boxes)  # One gene per box (ordering problem)
crossover_type="two_points"
mutation_type="random"
parent_selection_type="tournament"


ga_instance = pygad.GA(
    num_generations=1,
    num_parents_mating=2,
    fitness_func=fitness_function,
    sol_per_pop=6,
    num_genes=num_genes,
    gene_type=int,
    gene_space=[i for i in range(num_genes)],
    allow_duplicate_genes=False,  # ensures unique box indices per solution
    parent_selection_type=parent_selection_type,
    crossover_type = crossover_type,
    mutation_type = mutation_type,
    mutation_percent_genes=10
)

# Run the GA
ga_instance.run()

# Get the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution (box order):", solution)
print("Fitness value of the best solution:", solution_fitness)
print("Index:", solution_idx)

# Plot fitness progress
ga_instance.plot_fitness()

returned_boxes = [copy.deepcopy(boxes[i]) for i in solution]

fig = plt.figure(figsize=(10, 7)) 
ax = fig.add_subplot(111, projection='3d') 

draw_uld(ax)

best_chromosome = grid_based_pack(boxes)

a = 0

for box in best_chromosome:
        x, y, z = box['position']
        dx, dy, dz = box['dimensions']
        color = box['colour']                         
        draw_box(ax, x, y, z, dx, dy, dz, color)
        a+=1 

print(a)



#Axis setup               
ax.set_xlabel('X (Width)')
ax.set_ylabel('Y (Depth)')
ax.set_zlabel('Z (Height)')
ax.set_xlim(0, 100)
ax.set_ylim(0, 70)
ax.set_zlim(0, 70)
ax.set_title('Packing inside ULD')
ax.view_init(elev=25, azim=35)
plt.tight_layout()
plt.show()

