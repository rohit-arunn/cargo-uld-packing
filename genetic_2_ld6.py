import numpy as np
import math
import pandas as pd
import random
from multiprocessing import Pool
import random
import matplotlib.pyplot as plt
from plotly_main import create_ld6, create_box
import plotly.graph_objects as go
from plotly.offline import plot
from packing_ld6 import grid_based_pack, get_unique_rotations
from itertools import permutations

GRID_STEP = 1 

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




           
CONTAINER_DIMS = (160, 60.4, 64)    


def all_rotations(dim):
    
    return list(set(permutations(dim)))


def calculate_weight_penalty(placed_boxes, grid_step=1):
    penalty = 0
    for box in placed_boxes:
        x, y, z = box['position']
        dx, dy, dz = box['dimensions']
        weight = box['weight']

        # Check if there's any box directly below 
        for other in placed_boxes:
            if other == box:
                continue
            ox, oy, oz = other['position']
            odx, ody, odz = other['dimensions']
            oweight = other['weight']

            # Check if box is sitting directly on top of 'other'
            same_xy = (
                ox < x + dx and ox + odx > x and
                oy < y + dy and oy + ody > y
            )
            touching_z = abs(z - (oz + oz)) <= grid_step or (abs((oz + odz) - z) <= grid_step)

            if same_xy and touching_z:
                if weight > oweight:
                    penalty += (weight - oweight)  # Or any scaled version 
                break  # Only penalize once per support

    return penalty


def generate_initial_population(box_list, pop_size):
    population = []
    for _ in range(pop_size):
        chromo = []
        for box in box_list:
            chromo.append({'box_id': box['box_id'],
                           'dimensions': box['dimensions'],
                           'number':   box.get('number', 1),
                           'weight': box['weight']})
        population.append(chromo)
    return population

def grid_fill_ratio(grid):
    lz, ly, lx = grid.shape

    for z in reversed(range(lz)):
        if np.any(grid[z]):
            highest_layer = z + 1  # +1 because index starts at 0
            break
    else:
        return 0.0

    used_grid = grid[:highest_layer]

    # Step 3: Count filled cells and total space in used volume  
    filled_spaces = np.count_nonzero(used_grid)
    total_spaces = used_grid.size


    return filled_spaces / total_spaces

def evaluate_fitness(chromosome):
  
    
    boxes_for_fitness  = []
    for src in chromosome:         
        boxes_for_fitness.append({
            'box_id': src['box_id'],
            'dimensions': src['dimensions'],
            'number': src.get('number', 1),
            'weight': weight,
            'colour': src.get('colour', (random.random(), random.random(), random.random()))
        })

    
    placed, grid = grid_based_pack(boxes_for_fitness, CONTAINER_DIMS, grid_step = GRID_STEP)

    
    if not placed:
        return 0.0            



    weight_penalty = calculate_weight_penalty(placed)
    volume_reward = grid_fill_ratio(grid)

    fitness_function = volume_reward - 0.005*weight_penalty          
    return fitness_function, placed




def crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    
    child = [None] * size
    child[a:b] = parent1[a:b]
    
    existing_ids = set(box['box_id'] for box in child if box)
    fill = [gene for gene in parent2 if gene['box_id'] not in existing_ids]
    
    ptr = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill[ptr]
            ptr += 1
            
    return child



def mutate(chromo, mutation_rate=0.3):
    num_swaps = max(1, int(mutation_rate * len(chromo)))
    for _ in range(num_swaps):
        i, j = random.sample(range(len(chromo)), 2)
        chromo[i], chromo[j] = chromo[j], chromo[i]
    return chromo



def run_ga(boxes, generations=9, pop_size=8): 
    pop = generate_initial_population(boxes, pop_size) 

    fitnesses_main = []
    placements_main = []

    for gen in range(generations):
        with Pool(processes=8) as pool:  
            results = pool.map(evaluate_fitness, pop)
        fitnesses, placements = map(list, zip(*results))

        # print("fitnesses - ", fitnesses)         
        # print("placements - ", placements) 

        fitnesses_main += fitnesses
        placements_main += placements



        print(f'GEN {gen:03d}  best = {max(fitnesses):.3f}')

        def select():
            k = 3
            contenders = random.sample(list(zip(pop, fitnesses)), k)
            contenders.sort(key=lambda t: t[1], reverse=True)
            return contenders[0][0]

        next_pop = []
        while len(next_pop) < pop_size:
            p1, p2  = select(), select()
            child   = crossover(p1, p2)
            child = mutate(child, 0.3)
            next_pop.append(child) 

        pop = next_pop

    idx = fitnesses_main.index(max(fitnesses_main))
    #print("Best placement - ", placements_main[idx]) 


    # Return best chromosome
    
    return placements_main[idx]

if __name__ == '__main__':
    best_chromosome = run_ga(boxes)
    print('Best chromosome rotations:')
    for g in best_chromosome:
        print(g['box_id'], g['dimensions'])
    
        box_data = [
        (*box['position'], *box['dimensions'], box['colour'])
        for box in best_chromosome
    ]

    a = len(box_data)
    print("No: of boxes plotted -", a)

    container_mesh, container_edges = create_ld6('lightgray')

    traces = [container_mesh, container_edges]

    for x, y, z, dx, dy, dz, color in box_data:
        traces.append(create_box(x, y, z, dx, dy, dz, color=color, opacity=1.0, name="Box"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[0, 180], backgroundcolor="white"),
            yaxis=dict(nticks=10, range=[0, 65], backgroundcolor="white"),
            zaxis=dict(nticks=10, range=[0, 70], backgroundcolor="white"),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Open in browser
    plot(fig, filename='Optimized_packing_ld6.html', auto_open=True)