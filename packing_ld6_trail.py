import numpy as np
import math
import pandas as pd
from numba import njit
import random
from itertools import permutations
import matplotlib.pyplot as plt
from plotly_main import create_ld6, create_box
import plotly.graph_objects as go
from plotly.offline import plot
from packing import is_box_inside_ld6, is_point_inside_ld6
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 

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



def is_supported_in_ld6(x, y, z, dx, dy, dz, grid, threshold=0.7, give_ratio = False): 
    if z == 0:
        return True  # Base layer is always supported  
    elif -0.82644 * z +17.6280 - 1 < x < -0.82644 * z + 17.6281 + 1:
        return True
    elif 0.826 * z + 142.5 - 1 < (x+dx) < 0.826 * z + 142.5 + 1:
        return True

    support_area = grid[z - 1, y:y + dy, x:x + dx]
    total_cells = support_area.size
    filled_cells = np.count_nonzero(support_area == 1)

    support_ratio = filled_cells / total_cells

    if (give_ratio):
        return support_ratio
    else:
        return support_ratio >= threshold 


def get_unique_rotations(box_dims, grid_step=1):
    
    raw_rotations = set(
        tuple(math.ceil(d / grid_step) for d in p)
        for p in permutations(box_dims)
    )

    new_rotaions = sorted(raw_rotations, key=lambda d: d[2])

    filtered = []
    for rot in new_rotaions:
        a, b, c = rot
        if c < 2*(a+b):
            filtered.append(rot)
    
    return filtered 

def ld6_checker(x, y, z, dx, dy, dz, grid, threshold = 0.9):
    return (
        is_box_inside_ld6(x, y, z, dx, dy, dz)
        and is_supported_in_ld6(x, y, z, dx, dy, dz, grid, threshold)
    )



@njit
def fits_inside(x, y, z, dx, dy, dz, lx, ly, lz):
    return (x + dx <= lx) and (y + dy <= ly) and (z + dz <= lz)

@njit
def no_overlap(grid, x, y, z, dx, dy, dz):
    for k in range(dz):
        for j in range(dy):
            for i in range(dx):
                if grid[z+k, y+j, x+i] != 0:
                    return False
    return True

@njit
def place_box(grid, x, y, z, dx, dy, dz, box_id):
    for k in range(dz):
        for j in range(dy):
            for i in range(dx):
                grid[z+k, y+j, x+i] = box_id

@njit
def grid_based_pack(box_ids, dimensions, numbers, weights,
                    container_dims, grid_step):
    container_length, container_width, container_height = container_dims
    lx = int(container_length / grid_step)
    ly = int(container_width / grid_step)
    lz = int(container_height / grid_step)

    grid = np.zeros((lz, ly, lx), dtype=np.uint8)

    max_boxes = np.sum(numbers)
    placed_positions = np.zeros((max_boxes, 3), dtype=np.int32)
    placed_dims      = np.zeros((max_boxes, 3), dtype=np.int32)
    placed_ids       = np.zeros(max_boxes, dtype=np.int32)
    placed_weights   = np.zeros(max_boxes, dtype=np.float32)

    placed_count = 0

    for b in range(len(box_ids)):
        box_id = box_ids[b]
        dx, dy, dz = dimensions[b]
        numpcs = numbers[b]
        weight = weights[b]

        for _ in range(numpcs):
            placed = False
            for z in range(lz - dz + 1):
                for x in range(lx - dx + 1):
                    for y in range(ly - dy + 1):
                        if fits_inside(x, y, z, dx, dy, dz, lx, ly, lz) and no_overlap(grid, x, y, z, dx, dy, dz):
                            place_box(grid, x, y, z, dx, dy, dz, box_id)

                            # record placement
                            placed_positions[placed_count] = (x*grid_step, y*grid_step, z*grid_step)
                            placed_dims[placed_count] = (dx*grid_step, dy*grid_step, dz*grid_step)
                            placed_ids[placed_count] = box_id
                            placed_weights[placed_count] = weight
                            placed_count += 1

                            placed = True
                            break
                        if placed: break
                    if placed: break
                if placed: break

    return (placed_positions[:placed_count],
            placed_dims[:placed_count],
            placed_ids[:placed_count],
            placed_weights[:placed_count],
            grid)

def to_dicts(box_ids, dimensions, numbers, weights,
                    container_dims, grid_step):
    placed_boxes = []
    positions, dims, ids, weights = grid_based_pack(box_ids, dimensions, numbers, weights,
                    container_dims, grid_step)

    for i in range(len(ids)):
        placed_boxes.append({
            'box_id': int(ids[i]),
            'position': tuple(positions[i]),
            'dimensions': tuple(dims[i]),
            'weight': float(weights[i])
        })
    return placed_boxes
