import numpy as np
import math
import pandas as pd
import random
from itertools import permutations
import matplotlib.pyplot as plt
from properties_ld2 import draw_box, draw_uld_ld2, is_box_inside_ld2, is_point_inside_ld2
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



def is_supported_in_grid(x, y, z, dx, dy, dz, grid, threshold=0.7):
    if z == 0:
        return True  # Base layer is always supported
    elif x+dx > 47 and z < 16:
        return True

    support_area = grid[z - 1, y:y + dy, x:x + dx]
    total_cells = support_area.size
    filled_cells = np.count_nonzero(support_area == 1)

    support_ratio = filled_cells / total_cells

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


def grid_based_pack(box_list, container_dims=(61.5, 60.4, 64), grid_step=1):
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
        color = box.get('colour', (random.random(), random.random(), random.random()))

        
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
                        for i in range(0, len(rotations) - 1):

                            dx, dy, dz = rotations[i] 
                        
                            if (np.all(grid[z:z+dz, y:y+dy, x:x+dx] == 0) and
                                is_box_inside_ld2(x, y, z, dx, dy, dz) and
                                is_supported_in_grid(x, y, z, dx, dy, dz, grid)
                                    ):
                            
                                grid[z:z+dz, y:y+dy, x:x+dx] = 1

                                px, py, pz = x * grid_step, y * grid_step, z * grid_step
                                real_dims = (dx * grid_step, dy * grid_step, dz * grid_step)

                                placed_boxes.append({
                                    'box_id': box_id,
                                    'position': (px, py, pz),
                                    'dimensions': real_dims,
                                    'weight': weight, 
                                    'colour': color
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
                    'colour': color
                })
            box_list = next_box_list

    return placed_boxes



# def grid_based_pack(box_list, rejected, container_dims=(92, 60.4, 64), grid_step=1):
#     container_length, container_width, container_height = container_dims
    
#     lx = int(container_length / grid_step)
#     ly = int(container_width / grid_step)
#     lz = int(container_height / grid_step)

    
#     grid = np.zeros((lz, ly, lx), dtype=np.uint8)
#     placed_boxes = []
#     next_box_list = []

#     for box in box_list:
#         box_id = box['box_id']
#         original_dims = box['dimensions']
#         numpcs = box.get('number', 1)
#         weight = box['weight']
#         color = box.get('colour', (random.random(), random.random(), random.random()))

        
#         rotations = get_unique_rotations(original_dims)  

#         if rejected:
#             excluded_rots = [rot for bid, rot in rejected if bid == box_id]
#             rotations = [r for r in rotations if r not in excluded_rots]
#         else: 
#             continue

#         print("rotations -", rotations)  
#         print("rotations number- ", len(rotations) )    

#         placed_count = 0

#         for _ in range(numpcs):
#             placed = False

#             orientations = rotations[0] 

#             dx, dy, dz = orientations
            
#             for z in range(lz - dz + 1):
#                 for x in range(lx - dx + 1):
#                     for y in range(ly - dy + 1):
#                         for i in range(0, len(rotations) - 1):

#                             dx, dy, dz = rotations[i] 
                        
#                             if (np.all(grid[z:z+dz, y:y+dy, x:x+dx] == 0) and
#                                 is_box_inside_uld(x, y, z, dx, dy, dz) and
#                                 is_supported_in_grid(x, y, z, dx, dy, dz, grid, threshold=0.9)
#                                 ):
                            
#                                 grid[z:z+dz, y:y+dy, x:x+dx] = 1
#                                 px, py, pz = x * grid_step, y * grid_step, z * grid_step
#                                 real_dims = (dx * grid_step, dy * grid_step, dz * grid_step)

#                                 placed_boxes.append({
#                                     'box_id': box_id,
#                                     'position': (px, py, pz),
#                                     'dimensions': real_dims,
#                                     'weight': weight, 
#                                     'colour': color
#                                 })

#                                 placed = True
#                                 break
#                             if placed: break
#                         if placed: break
#                     if placed: break
#                 if placed: 
#                     placed_count += 1
#                     break

                
#             remaining = numpcs - placed_count
#             if remaining > 0:
#                 next_box_list.append({
#                     'box_id': box_id,
#                     'dimensions': original_dims,
#                     'number': remaining,
#                     'weight': weight, 
#                     'colour': color
#                 })
#             box_list = next_box_list

            
#     return placed_boxes, grid


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

draw_uld_ld2(ax)

best_chromosome = grid_based_pack(boxes)
print(best_chromosome)
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