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

def get_predefined_rotations(box_list):
    rotations_dict = {}
    for box in box_list:
        box_id = box['box_id']
        dims = box['dimensions']
        # Store only flat (or smart filtered) permutations
        rotations = [r for r in set(permutations(dims))]   # if r[2] == min(r)
        rotations_dict[box_id] = rotations
    return rotations_dict


def decode_chromosome(chromosome, box_list, rotations_dict):
    decoded = []
    for gene, box in zip(chromosome, box_list):
        rotation = rotations_dict[gene['box_id']][gene['rotation_index']]
        decoded.append({
            'box_id': gene['box_id'],
            'dimensions': rotation,
            'number': box['number'],
            'colour': box.get('colour', (random.random(), random.random(), random.random()))
        })
    return decoded



def generate_rotations_for_box(box_dims):
    return list(set(permutations(box_dims)))  # all unique 3D rotations 

predefined_rotations = {}

for box in boxes:
    box_id = box['box_id']
    dims = box['dimensions']
    predefined_rotations[box_id] = generate_rotations_for_box(dims)


CONTAINER_DIMS = (92, 60.4, 64)

def evaluate_fitness(chromosome, return_positions=False):
    try:
        box_list = [
            {
                'box_id': gene['box_id'],
                'dimensions': predefined_rotations[gene['box_id']][gene['rotation_index']],
                'number': gene.get('number', 1)
            }
            for gene in chromosome
        ]

        placed = grid_based_pack(box_list, container_dims=CONTAINER_DIMS, grid_step=1)

        if not placed:
            return (0.0, []) if return_positions else 0.0

        filled_volume = sum(
            d[0] * d[1] * d[2] for b in placed for d in [b['dimensions']]
        )

        max_height = max((b['position'][2] + b['dimensions'][2]) for b in placed)
        if max_height == 0:
            return (0.0, placed) if return_positions else 0.0

        bounding_volume = CONTAINER_DIMS[0] * CONTAINER_DIMS[1] * max_height
        efficiency = filled_volume / bounding_volume

        if not math.isfinite(efficiency) or efficiency < 0:
            efficiency = 0.0

        return (efficiency, placed) if return_positions else efficiency

    except Exception as e:
        print(f"Error during fitness evaluation: {e}")
        return (0.0, []) if return_positions else 0.0





def generate_initial_population(box_list, rotations_dict, pop_size):
    population = []
    for _ in range(pop_size):
        chromosome = []
        for box in box_list:
            box_id = box['box_id']
            num_rotations = len(rotations_dict[box_id])
            rotation_index = random.randint(0, num_rotations - 1)
            chromosome.append({'box_id': box_id, 'rotation_index': rotation_index})
        population.append(chromosome)
    return population

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-1)
    return parent1[:point] + parent2[point:]

def mutate(chromosome, rotations_dict, mutation_rate=0.1):
    for gene in chromosome:
        if random.random() < mutation_rate:
            num_rot = len(rotations_dict[gene['box_id']])
            gene['rotation_idx'] = random.randint(0, num_rot - 1)


def run_genetic_algorithm(box_list, container_dims = (92, 60.4, 64), generations=2, pop_size=3, mutation_rate=0.1):
    rotations_dict = get_predefined_rotations(box_list)
    population = generate_initial_population(box_list, rotations_dict, pop_size)

    for gen in range(generations):
        fitnesses, placements = zip(*[ evaluate_fitness(ch, return_positions=True) for ch in population])

        fitnesses = list(fitnesses)
        placements = list(placements)
        print("placements is - ", placements)
        print("Fitnesses:", fitnesses)

        next_gen = []

        for _ in range(pop_size):
            
            p1, p2 = random.choices(population, weights=fitnesses, k=2)
            child = crossover(p1, p2)
            mutate(child, rotations_dict, mutation_rate)
            next_gen.append(child)

        population = next_gen
        print(f"Generation {gen}: Best fitness = {max(fitnesses):.4f}")

    best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
    best_chromosome = placements[best_idx]
    best_box_arrangement = decode_chromosome(best_chromosome, box_list, rotations_dict)
    return best_box_arrangement




fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

draw_uld(ax)

best_chromosome = run_genetic_algorithm(boxes)

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
ax.set_title('Packing inside ULD - 2 ')
ax.view_init(elev=25, azim=35)
plt.tight_layout()
plt.show()
