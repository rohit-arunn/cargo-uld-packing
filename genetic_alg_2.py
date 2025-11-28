import numpy as np
import pandas as pd
from numba import njit
import random
from multiprocessing import Pool, cpu_count
from packing import draw_uld, draw_box
import matplotlib.pyplot as plt
from plotly_main import create_ld1, create_box
import plotly.graph_objects as go
from plotly.offline import plot
from packing_ld1 import grid_based_pack, is_supported_in_ld1, is_point_inside_ld1, is_box_inside_ld1, get_unique_rotations
from itertools import permutations

GRID_STEP = 1 

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

boxes = [{'box_id': 36, 'dimensions': (5, 20, 5), 'number': 550, 'weight': 20}]


           
CONTAINER_DIMS = (92, 60.4, 64)    


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
            'weight': src.get('weight'),
            'colour': src.get('colour', (random.random(), random.random(), random.random()))
        })

    
    placed, grid = grid_based_pack(boxes_for_fitness, CONTAINER_DIMS, grid_step = GRID_STEP)

    
    if not placed:
        return 0.0            



    weight_penalty = calculate_weight_penalty(placed)
    volume_reward = grid_fill_ratio(grid)

    fitness_function = volume_reward #- 0.005*weight_penalty          
    return fitness_function, placed

# def crossover(parent1, parent2): 
#     size = len(parent1)
#     cut = random.randint(1, size - 1)
#     child = parent1[:cut] + parent2[cut:]    
#     return child


# def mutate(chromo, mutation_rate=0.3):  
#     for gene in chromo:
#         if random.random() < mutation_rate:
#             gene['rotation'] = random.choice(all_rotations(gene['rotation']))    

def crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    
    # Step 1: Copy the slice from parent1
    child = [None] * size
    child[a:b] = parent1[a:b]

    # Step 2: Fill remaining slots with boxes from parent2, skipping ones already presents
    def get_box_id(box): return box['box_id']
    existing_ids = set(get_box_id(box) for box in child if box is not None)

    fill_boxes = [box for box in parent2 if get_box_id(box) not in existing_ids]

    fill_idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill_boxes[fill_idx]
            fill_idx += 1

    return child


# @njit
# def mutate(chromo, mutation_rate=0.1):
#     n = len(chromo)
#     num_swaps = max(1, int(mutation_rate * n))

#     for _ in range(num_swaps):
#         i = np.random.randint(0, n)
#         j = np.random.randint(0, n)
#         while j == i:  # ensure different indices
#             j = np.random.randint(0, n)

#         # swap
#         tmp = chromo[i]
#         chromo[i] = chromo[j]
#         chromo[j] = tmp

#     return chromo


# def mutate(chromo, mutation_rate=0.1):
#     num_swaps = max(1, int(mutation_rate * len(chromo)))
#     for _ in range(num_swaps):
#         i, j = random.sample(range(len(chromo)), 2)   
#         chromo[i], chromo[j] = chromo[j], chromo[i]
#     return chromo


def mutate(chromo, mutation_rate=0.1, P=None):
    chromo = chromo.copy()   # avoid modifying original
    
    num_swaps = max(1, int(mutation_rate * len(chromo)))
    for _ in range(num_swaps):
        i, j = random.sample(range(len(chromo)), 2)

        if P is not None:
            # Use box_id as a key into P
            bi = chromo[i]['box_id'][0]  # e.g. first element of tuple as unique ID
            bj = chromo[j]['box_id'][0]

            pi = P[bi, i] if bi < P.shape[0] else 0
            pj = P[bj, j] if bj < P.shape[0] else 0

            if pj > pi:
                chromo[i], chromo[j] = chromo[j], chromo[i]
            elif random.random() < 0.2:  # exploration
                chromo[i], chromo[j] = chromo[j], chromo[i]
        else:
            # plain random swap
            chromo[i], chromo[j] = chromo[j], chromo[i]
    return chromo




def run_ga(boxes, generations=3, pop_size=8): 
    pop = generate_initial_population(boxes, pop_size)

    fitnesses_main = []
    placements_main = []
    generations_plot = []
    fitness_history = []

    # Initialize probability matrix for positional learning
    num_boxes = len(boxes)
    P = np.ones((num_boxes, num_boxes)) / num_boxes  # uniform start
    learning_rate = 0.05

    for gen in range(generations):
        with Pool(processes=8) as pool:                       
            results = pool.map(evaluate_fitness, pop)
        fitnesses, placements = map(list, zip(*results))

        fitnesses_main += fitnesses
        placements_main += placements

        print(f'GEN {gen:03d}  best = {max(fitnesses):.3f}')
        generations_plot.append(gen)          # x-axis
        fitness_history.append(max(fitnesses))  # y-axis 

        # --- Selection (size k tournament) ---
        def select():
            k = 5
            contenders = random.sample(list(zip(pop, fitnesses)), k)
            contenders.sort(key=lambda t: t[1], reverse=True)
            return contenders[0][0]

        next_pop = []
        while len(next_pop) < pop_size:
            p1, p2  = select(), select()
            child   = crossover(p1, p2)

            # mutation now biased by probability matrix P
            child = mutate(child, 0.3, P)

            next_pop.append(child) 

        # --- Learning step ---
        # We already know which individuals are best this generation
        # -> Update P based on elites (no re-evaluation needed)
        elite_indices = np.argsort(fitnesses)[-3:]   # top 3 as example
        for idx in elite_indices:
            elite = pop[idx]
            for pos, box in enumerate(elite):
                P[box, pos] += learning_rate
        # Normalize
        P /= np.sum(P, axis=0, keepdims=True)

        pop = next_pop

    idx = fitnesses_main.index(max(fitnesses_main))
    print("The generation which is being printed - ", idx)
    #print("Best placement - ", placements_main[idx])


    # Return best chromosome     
    
    return placements_main[idx], generations_plot, fitness_history

if __name__ == '__main__':
    best, generations_plot, fitness_history = run_ga(boxes)
    print('Best chromosome positions:')
    for g in best:
        print(g['box_id'], g['position'])
    

    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d') 

    # draw_uld(ax)                     

    # #best_chromosome = run_ga(boxes) 

    # a = 0

    # for box in best:
    #         x, y, z = box['position']
    #         dx, dy, dz = box['dimensions']
    #         color = box['colour']
    #         draw_box(ax, x, y, z, dx, dy, dz, color)
    #         a+=1

    # print("Boxes plotted - ", a)



    # #Axis setup
    # ax.set_xlabel('X (Width)')
    # ax.set_ylabel('Y (Depth)')
    # ax.set_zlabel('Z (Height)')
    # ax.set_xlim(0, 100)
    # ax.set_ylim(0, 70)
    # ax.set_zlim(0, 70)
    # ax.set_title('Packing inside ULD - 2 ')
    # ax.view_init(elev=25, azim=35)
    # plt.tight_layout() 
    # plt.show() 

    
    box_data = [
        (*box['position'], *box['dimensions'], box['colour'])
        for box in best
    ]

    a = len(box_data)
    print("No: of boxes plotted -", a)

    container_mesh, container_edges = create_ld1('lightgray')

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
    plot(fig, filename='Optimized_packing.html', auto_open=True)

    plt.plot(generations_plot, fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.show()
