import numpy as np
import math
import pandas as pd
import random
from packing import draw_uld, draw_box
import random
import matplotlib.pyplot as plt
import random
from first import grid_based_pack, is_supported_in_grid, is_point_inside_uld, is_box_inside_uld
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

    boxes.append({
        'box_id': box_id,
        'dimensions': (length, width, height),
        'number' : numpcs
    })

           
CONTAINER_DIMS = (92, 60.4, 64)    


def all_rotations(dim):
    """Return list of 6 axis‑aligned rotations (tuples)."""
    return list(set(permutations(dim)))

def generate_initial_population(box_list, pop_size):
    population = []
    for _ in range(pop_size):
        chromo = []
        for box in box_list:
            rot    = random.choice(all_rotations(box['dimensions']))
            chromo.append({'box_id': box['box_id'],
                           'rotation': rot,
                           'number':   box.get('number', 1),
                           'colour':   box.get('colour')})
        population.append(chromo)
    return population

def evaluate_fitness(chromosome):
    """
    ‑ chromosome: list of genes  [{'box_id': .., 'rotation': (l,w,h)}]

    RETURNS
        packing efficiency  (= filled‑cell ratio up to tallest layer)
    """

    # --- 1.  Build a box‑list where each entry carries the chosen rotation
    id2rot = {g['box_id']: g['rotation'] for g in chromosome}
    boxes_for_fitness  = []
    for src in boxes:             # BOX_LIST = original data source
        boxes_for_fitness.append({
            'box_id': src['box_id'],
            'dimensions': id2rot[src['box_id']],
            'number': src.get('number', 1),
            'colour': src.get('colour', (random.random(), random.random(), random.random()))
        })

    # --- 2.  Run the deterministic packer
    placed = grid_based_pack(boxes_for_fitness, CONTAINER_DIMS, grid_step = GRID_STEP)

    # --- 3.  Re‑create a small occupancy grid only up to max‑Z
    if not placed:
        return 0.0            # nothing packed

    # Find highest occupied layer (real units → grid units)
    max_z_real = max(b['position'][2] + b['dimensions'][2] for b in placed)
    max_z_idx  = math.ceil(max_z_real / GRID_STEP)

    # Grid extents in voxels
    gx = math.ceil(CONTAINER_DIMS[0] / GRID_STEP)
    gy = math.ceil(CONTAINER_DIMS[1] / GRID_STEP)
    gz = max_z_idx

    grid = np.zeros((gz, gy, gx), dtype=np.uint8)

    # Mark every placed box into the grid
    for b in placed:
        x, y, z  = (int(b['position'][i] / GRID_STEP) for i in (0,1,2))
        dx, dy, dz = (int(b['dimensions'][i] / GRID_STEP) for i in (0,1,2))
        grid[z : z+dz, y : y+dy, x : x+dx] = 1

    filled_cells = np.count_nonzero(grid)
    total_cells  = grid.size                # only up to highest layer
    return filled_cells / total_cells, placed

def crossover(parent1, parent2):
    size   = len(parent1)
    cut    = random.randint(1, size-1)
    child  = parent1[:cut] + parent2[cut:]
    return child

def mutate(chromo, mutation_rate=0.1):
    for gene in chromo:
        if random.random() < mutation_rate:
            gene['rotation'] = random.choice(all_rotations(gene['rotation']))



def run_ga(boxes, generations=2, pop_size=3):
    pop = generate_initial_population(boxes, pop_size)

    fitnesses_main = []
    placements_main = []

    for gen in range(generations):
        results = [evaluate_fitness(c) for c in pop]   # → [(fit₁, placed₁), (fit₂, placed₂), ...]
        fitnesses, placements = map(list, zip(*results))

        # print("fitnesses - ", fitnesses)
        # print("placements - ", placements)

        fitnesses_main += fitnesses
        placements_main += placements



        print(f'Gen {gen:03d}  best = {max(fitnesses):.3f}')

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
            mutate(child, 0.1)
            next_pop.append(child)

        pop = next_pop

    idx = fitnesses_main.index(max(fitnesses_main))
    print("Best placement - ", placements_main[idx])


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


