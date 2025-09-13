import numpy as np
import math
import pandas as pd
import random
from numba import njit
from itertools import permutations
import matplotlib.pyplot as plt
from plotly_main import create_ld8, create_box
import plotly.graph_objects as go
from plotly.offline import plot
from packing_ld1 import ld1_checker as ld1
from packing_ld2 import ld2_checker as ld2
from packing_ld3 import ld3_checker as ld3
from packing_ld6 import ld6_checker as ld6
from packing_ld8 import ld8_checker as ld8
from packing_ld9 import ld9_checker as ld9
# from packing_ld11 import ld11_checker as ld11
# from packing_ld39 import ld11_checker as ld39

from mpl_toolkits.mplot3d.art3d import Poly3DCollection 

# df = pd.read_parquet("flight_ICN_to_BUD.parquet")


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

ULD_CHECKERS = {
    "LD1": ld1,
    "LD2": ld2,
    "LD3": ld3,
    "LD6": ld6,
    "LD8": ld8,
    "LD9": ld9,     
}

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

@njit 
def no_overlap(grid, x, y, z, dx, dy, dz):
    for k in range(dz):
        for j in range(dy):
            for i in range(dx):
                if grid[z+k, y+j, x+i] != 0:
                    return False
    return True

@njit
def place_box(grid, x, y, z, dx, dy, dz):
    for k in range(dz):
        for j in range(dy):
            for i in range(dx):
                grid[z+k, y+j, x+i] = 1

class GridBasedPacking:
    def __init__(self, boxes, uld_checker, grid_step):
        self.boxes = boxes
        self.uld_checker = ULD_CHECKERS[uld_checker]
        self.grid_step = grid_step


    

    def grid_based_pack(self, box_list, container_dims, grid_step):
        container_length, container_width, container_height = container_dims
        
        lx = int(container_length / grid_step)
        ly = int(container_width / grid_step)
        lz = int(container_height / grid_step)

        
        grid = np.zeros((lz, ly, lx), dtype=np.uint8) 
        placed_boxes = []
        next_box_list = []

        for box in box_list:
            box_id = box['box_id']
            original_dims = box['dimensions']
            numpcs = box.get('number', 1)
            weight = box['weight']
            rgb = box.get('colour', (random.random(), random.random(), random.random()))

            
            rotations = get_unique_rotations(original_dims) 

            print("rotations -", rotations)  
            print("rotations number- ", len(rotations) )    

            placed_count = 0 

            for _ in range(numpcs):
                placed = False

                orientations = rotations[0] 

                dx, dy, dz = orientations
                
                for z in range(lz - dz + 1):
                    for x in range(lx - dx + 1):
                        for y in range(ly - dy + 1): 
                            for i in range(0, len(rotations)):

                                dx, dy, dz = rotations[i] 
                            
                                if (no_overlap(grid, x, y, z, dx, dy, dz) and 
                                   # np.all(grid[z:z+dz, y:y+dy, x:x+dx] == 0) and
                                    self.uld_checker(x, y, z, dx, dy, dz, grid, threshold=0.9)):
                                
                                    #grid[z:z+dz, y:y+dy, x:x+dx] = 1
                                    place_box(grid, x, y, z, dx, dy, dz)

                                    px, py, pz = x * grid_step, y * grid_step, z * grid_step
                                    real_dims = (dx * grid_step, dy * grid_step, dz * grid_step)

                                    placed_boxes.append({
                                        'box_id': box_id,
                                        'position': (px, py, pz),
                                        'dimensions': real_dims,
                                        'weight': weight, 
                                        'colour': f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'
                                    })

                                    placed = True
                                    break
                                if placed: break
                            if placed: break            
                        if placed: break
                    if placed: 
                        placed_count += 1
                        break

                    
                remaining = numpcs - placed_count
                if remaining > 0:
                    next_box_list.append({
                        'box_id': box_id,
                        'dimensions': original_dims,
                        'number': remaining,
                        'weight': weight, 
                        'colour': f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'
                    })
                box_list = next_box_list

        return placed_boxes, grid