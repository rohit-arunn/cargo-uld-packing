import random
import pandas as pd
from functools import lru_cache
from packing import layer_based_pack, draw_uld, draw_box
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import json



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

    boxes.append({
        'box_id': box_id,
        'dimensions': (length, width, height)
        # Add other keys like 'weight' if needed later
    })



# boxes = [
#     {'box_id': 1, 'dimensions': (20, 10, 10)},
#     {'box_id': 2, 'dimensions': (10, 20, 10)},
#     {'box_id': 3, 'dimensions': (10, 10, 20)},
#     {'box_id': 4, 'dimensions': (12, 12, 12)},
#     {'box_id': 5, 'dimensions': (10, 10, 10)},
#     {'box_id': 6, 'dimensions': (15, 15, 15)},
#     {'box_id': 7, 'dimensions': (20, 10, 10)},
#     {'box_id': 8, 'dimensions': (10, 20, 10)},
#     {'box_id': 9, 'dimensions': (10, 10, 20)},
#     {'box_id': 10, 'dimensions': (12, 12, 12)},
#     {'box_id': 11, 'dimensions': (10, 10, 10)},
#     {'box_id': 12, 'dimensions': (15, 15, 15)},
#     {'box_id': 13, 'dimensions': (20, 10, 10)},
#     {'box_id': 14, 'dimensions': (10, 20, 10)},
#     {'box_id': 15, 'dimensions': (15, 15, 15)},
#     {'box_id': 16, 'dimensions': (20, 10, 10)},
#     {'box_id': 17, 'dimensions': (10, 20, 10)},
#     {'box_id': 18, 'dimensions': (15, 15, 15)},
#     {'box_id': 19, 'dimensions': (20, 10, 10)},
#     {'box_id': 20, 'dimensions': (10, 20, 10)},
#     {'box_id': 21, 'dimensions': (10, 10, 20)},
#     {'box_id': 22, 'dimensions': (12, 12, 12)},
#     {'box_id': 23, 'dimensions': (10, 10, 10)},
#     {'box_id': 24, 'dimensions': (15, 15, 15)},
#     {'box_id': 25, 'dimensions': (20, 10, 10)},
#     {'box_id': 26, 'dimensions': (10, 20, 10)},
#     {'box_id': 27, 'dimensions': (10, 10, 20)},
#     {'box_id': 28, 'dimensions': (8, 8, 8)},
#     {'box_id': 29, 'dimensions': (20, 10, 10)},
#     {'box_id': 30, 'dimensions': (10, 20, 10)},
#     {'box_id': 31, 'dimensions': (15, 15, 15)},
#     {'box_id': 32, 'dimensions': (20, 10, 10)},
#     # {'box_id': 33, 'dimensions': (10, 20, 10)},
#     # {'box_id': 34, 'dimensions': (15, 15, 15)},
#     # {'box_id': 35, 'dimensions': (20, 10, 10)},
#     # {'box_id': 36, 'dimensions': (10, 20, 10)}
# ]


#(box_id, length, width, height)
# boxes = [
#     (1, 20, 10, 10), (2, 10, 20, 10), (3, 10, 10, 20), (4, 12, 12, 12),
#     (5, 10, 10, 10), (6, 15, 15, 15), (7, 20, 10, 10), (8, 10, 20, 10),
#     (9, 10, 10, 20), (10, 12, 12, 12), (11, 10, 10, 10), (12, 15, 15, 15),
#     (13, 20, 10, 10), (14, 10, 20, 10), (15, 15, 15, 15), (16, 20, 10, 10),
#     (17, 10, 20, 10), (18, 15, 15, 15), (19, 20, 10, 10), (20, 10, 20, 10),
#     (21, 10, 10, 20), (22, 12, 12, 12), (23, 10, 10, 10), (24, 15, 15, 15),
#     (25, 20, 10, 10), (26, 10, 20, 10), (27, 10, 10, 20), (28, 8, 8, 8),
#     (29, 20, 10, 10), (30, 10, 20, 10), (31, 15, 15, 15), (32, 20, 10, 10),
#     (33, 10, 20, 10), (34, 15, 15, 15), (35, 20, 10, 10), (36, 10, 20, 10)
# ] 

# Container dimensions (ULD LD-1 type)
CONTAINER_DIMS = (92, 60.4, 64)



def evaluate_fitness(chromosome, return_positions=False):
    box_list = [{'box_id': gene['box_id'], 'dimensions': gene['rotation']} for gene in chromosome]
    placed = layer_based_pack(box_list=box_list, container_dims=CONTAINER_DIMS, step=1)
    packed_volume = sum(
    dims[0] * dims[1] * dims[2] for box in placed for dims in [box['dimensions']])

    total_volume = 92 * 60.4 * 64 - 0.5*21.33*30.5*60.4
    fitness = packed_volume / total_volume


    if return_positions:
        return fitness, placed

    return fitness

def evaluate_fitness_wrapper(chromosome):
    return evaluate_fitness(chromosome, return_positions=True)



# Generate initial population
import random

def generate_initial_population(boxes, pop_size):
    population = []
    for _ in range(pop_size):
        shuffled = random.sample(boxes, len(boxes))
        chromosome = []
        for box in shuffled:
            box_id = box['box_id']
            l, w, h = box['dimensions']
            rotation = (l, w, h)
               
            chromosome.append({'box_id': box_id, 'rotation': rotation})
        population.append(chromosome)
    return population



def tournament_selection(pop, fitnesses, k=3):
    selected = random.sample(list(zip(pop, fitnesses)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]

def crossover(p1, p2):
    size = len(p1)
    start, end = sorted(random.sample(range(size), 2))
    mid = p1[start:end]
    used = {gene['box_id'] for gene in mid}
    child = mid + [gene for gene in p2 if gene['box_id'] not in used]
    return child

def mutate(chromosome, mutation_rate=0.1):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(chromosome) - 1)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        if random.random() < mutation_rate:
            l, w, h = chromosome[i]['rotation']
            orientations = [
                (l, w, h), (l, h, w), (w, l, h),
                (w, h, l), (h, l, w), (h, w, l)
            ]
            chromosome[i]['rotation'] = random.choice(orientations)

def run_genetic_algorithm(generations=1, pop_size=3, mutation_rate=0.1):
    population = generate_initial_population(boxes, pop_size)
    placements_main = []
    fitnesses_main = []

    for gen in range(generations):
        
        #results = [evaluate_fitness(chromosome, return_positions=True) for chromosome in population]
        with Pool(processes=8) as pool:  
            results = pool.map(evaluate_fitness_wrapper, population)
        fitnesses, placements = map(list, zip(*results))

        fitnesses_main += fitnesses
        placements_main += placements

        new_population = []
        for _ in range(pop_size):
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            child = crossover(p1, p2)
            mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        print("Done with one gen")

    best_fit = max(fitnesses_main)
    theindex = fitnesses_main.index(best_fit)
    
    return placements_main[theindex]



if __name__ == "__main__":
    best_chromosome = run_genetic_algorithm()

    print("\nBest Packing Order:")
    for box in best_chromosome:
        print(box)
        # box_id = box["box_id"]
        # position = box["position"]
        # rotation = box["dimensions"]
        # print(f"Box ID: {box_id}, Position: {position}, Rotation: {rotation}")

    # Visualize final result
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    draw_uld(ax)

    for box in best_chromosome:
        x, y, z = box['position']
        dx, dy, dz = box['dimensions']
        draw_box(ax, x, y, z, dx, dy, dz)




    #box_dims = [gene['rotation'] for gene in best_chromosome]
    #layer_based_pack(ax=ax, box_dims_list=box_dims, container_dims=CONTAINER_DIMS, step=0.5)

    ax.set_xlabel('X (Width)')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z (Height)')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.set_zlim(0, 70)
    ax.set_title('Diagram')
    ax.view_init(elev=25, azim=35)
    plt.tight_layout()
    plt.show()
