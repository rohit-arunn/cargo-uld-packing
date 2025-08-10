import numpy as np
import pandas as pd
import random
from packing_ld1 import grid_based_pack


CONTAINER_DIMS = (92, 60.4, 64) 
GRID_STEP = 1




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
            'weight': src['weight'],
            'colour': src.get('colour', (random.random(), random.random(), random.random()))
        })

    
    placed, grid = grid_based_pack(boxes_for_fitness, CONTAINER_DIMS, grid_step = GRID_STEP)

    
    if not placed:
        return 0.0            

    weight_penalty = calculate_weight_penalty(placed)
    volume_reward = grid_fill_ratio(grid)

    fitness_function = volume_reward - 0.005*weight_penalty          
    return fitness_function, placed




