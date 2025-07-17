import numpy as np
import math
import pandas as pd
import random
from multiprocessing import Pool, cpu_count
from packing import draw_uld, draw_box
import random
import matplotlib.pyplot as plt
import random
from first import grid_based_pack, is_supported_in_grid, is_point_inside_uld, is_box_inside_uld, get_unique_rotations
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
            rot    = random.choice(all_rotations(box['dimensions']))
            chromo.append({'box_id': box['box_id'],
                           'rotation': rot,
                           'number':   box.get('number', 1),
                           'weight': weight,
                           'colour':   box.get('colour')})
        population.append(chromo)
    return population

def evaluate_fitness(chromosome):
  
    id2rot = {g['box_id']: g['rotation'] for g in chromosome}
    boxes_for_fitness  = []
    for src in boxes:             # BOX_LIST = original data source
        boxes_for_fitness.append({
            'box_id': src['box_id'],
            'dimensions': id2rot[src['box_id']],
            'number': src.get('number', 1),
            'weight': weight,
            'colour': src.get('colour', (random.random(), random.random(), random.random()))
        })

    
    placed = grid_based_pack(boxes_for_fitness, CONTAINER_DIMS, grid_step = GRID_STEP)

    
    if not placed:
        return 0.0            

    
    max_z_real = max(b['position'][2] + b['dimensions'][2] for b in placed)
    max_z_idx  = math.ceil(max_z_real / GRID_STEP)

    
    gx = math.ceil(CONTAINER_DIMS[0] / GRID_STEP)
    gy = math.ceil(CONTAINER_DIMS[1] / GRID_STEP)
    gz = max_z_idx

    grid = np.zeros((gz, gy, gx), dtype=np.uint8)

    
    for b in placed:
        x, y, z  = (int(b['position'][i] / GRID_STEP) for i in (0,1,2))
        dx, dy, dz = (int(b['dimensions'][i] / GRID_STEP) for i in (0,1,2))
        grid[z : z+dz, y : y+dy, x : x+dx] = 1

    filled_cells = np.count_nonzero(grid)
    total_cells  = grid.size   
    weight_penalty = calculate_weight_penalty(placed)

    fitness_function = filled_cells / total_cells - 0.05*weight_penalty          
    return fitness_function, placed

def crossover(parent1, parent2):
    size = len(parent1)
    cut = random.randint(1, size - 1)
    child = parent1[:cut] + parent2[cut:]
    return child


# def mutate(chromo, mutation_rate=0.3):
#     for gene in chromo:
#         if random.random() < mutation_rate:
#             gene['rotation'] = random.choice(all_rotations(gene['rotation']))

# def crossover(parent1, parent2):
#     size = len(parent1)
#     a, b = sorted(random.sample(range(size), 2))
    
#     child = [None] * size
#     child[a:b] = parent1[a:b]
    
#     existing_ids = set(box['box_id'] for box in child if box)
#     fill = [gene for gene in parent2 if gene['box_id'] not in existing_ids]
    
#     ptr = 0
#     for i in range(size):
#         if child[i] is None:
#             child[i] = fill[ptr]
#             ptr += 1
            
#     return child

def smart_mutate(chromo, placement, mutation_rate=0.3):
    if random.random() > mutation_rate:
        return chromo  # No mutation

    # Find max z and boxes contributing to it
    max_z = max(z + dz for (x, y, z, dx, dy, dz) in placement)
    tall_boxes = [b for b in placement if (b[2] + b[5]) == max_z]

    # Get box IDs at top layer
    tall_box_ids = [b[6] for b in tall_boxes]  # Assuming box_id is at index 6

    # Modify rotation of these boxes in the chromosome
    for gene in chromo:
        if gene['box_id'] in tall_box_ids:
            current_rotation = gene['rotation']
            all_possible = get_unique_rotations(current_rotation)
            others = [r for r in all_possible if r != current_rotation]
            if others:
                gene['rotation'] = random.choice(others)

    return chromo


def mutate(chromo, mutation_rate=0.3):
    num_swaps = max(1, int(mutation_rate * len(chromo)))
    for _ in range(num_swaps):
        i, j = random.sample(range(len(chromo)), 2)
        chromo[i], chromo[j] = chromo[j], chromo[i]
    return chromo



def run_ga(boxes, generations=4, pop_size=3):
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

        # Selection (tournament size 3)
        def select():
            k = 3
            contenders = random.sample(list(zip(pop, fitnesses)), k)
            contenders.sort(key=lambda t: t[1], reverse=True)
            return contenders[0][0]

        next_pop = []
        while len(next_pop) < pop_size:
            p1, p2  = select(), select()
            child   = crossover(p1, p2)
            mutate(child, 0.3)
            next_pop.append(child)

        pop = next_pop

    idx = fitnesses_main.index(max(fitnesses_main))
    #print("Best placement - ", placements_main[idx])


    # Return best chromosome
    
    return placements_main[idx]

if __name__ == '__main__':
    best = run_ga(boxes)
    print('Best chromosome rotations:')
    for g in best:
        print(g['box_id'], g['dimensions'])
    

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    draw_uld(ax)

    #best_chromosome = run_ga(boxes)

    a = 0

    for box in best:
            x, y, z = box['position']
            dx, dy, dz = box['dimensions']
            color = box['colour']
            draw_box(ax, x, y, z, dx, dy, dz, color)
            a+=1

    print("Boxes plotted - ", a)



    #Axis setup
    ax.set_xlabel('X (Width)')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z (Height)')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.set_zlim(0, 70)
    ax.set_title('Packing inside ULD - 2 ')
    ax.view_init(elev=25, azim=35)
    plt.tight_layout()
    plt.show()
