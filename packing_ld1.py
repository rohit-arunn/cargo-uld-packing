import numpy as np
import math
import pandas as pd
from numba import njit
import random
from itertools import permutations
import matplotlib.pyplot as plt
from plotly_main import create_ld1, create_box
import plotly.graph_objects as go
from plotly.offline import plot
from packing import draw_box, draw_uld, is_box_inside_ld1, is_point_inside_ld1
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


boxes = [{'box_id': 36, 'dimensions': (5, 20, 5), 'number': 550, 'weight': 20}]


def is_supported_in_ld1(x, y, z, dx, dy, dz, grid, threshold=0.7, give_ratio = False):
    if z == 0:
        return True  # Base layer is always supported 
    elif 1.4300014300014 * z + 61.4980694 - 1 < (x+dx) < 1.4300014300014 * z + 61.4980694 + 1:
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

def ld1_checker(x, y, z, dx, dy, dz, grid, threshold = 0.9):
    return (
        is_box_inside_ld1(x, y, z, dx, dy, dz)
        and is_supported_in_ld1(x, y, z, dx, dy, dz, grid, threshold)
    )


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

def grid_based_pack(box_list, container_dims=(92, 60.4, 64), grid_step=1):
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
                                ld1_checker(x, y, z, dx, dy, dz, grid, threshold = 0.9)
                                ):
                            
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





#Below is using MATPLOTLIB


# fig = plt.figure(figsize=(10, 7)) 
# ax = fig.add_subplot(111, projection='3d') 

# draw_uld(ax)

# best_chromosome = grid_based_pack(boxes)

# a = 0

# for box in best_chromosome:
#         x, y, z = box['position']
#         dx, dy, dz = box['dimensions']
#         color = box['colour']                         
#         draw_box(ax, x, y, z, dx, dy, dz, color)
#         a+=1 

# print(a)



# #Axis setup               
# ax.set_xlabel('X (Width)')
# ax.set_ylabel('Y (Depth)')
# ax.set_zlabel('Z (Height)')
# ax.set_xlim(0, 100)
# ax.set_ylim(0, 70)
# ax.set_zlim(0, 70)
# ax.set_title('Packing inside ULD')
# ax.view_init(elev=25, azim=35)
# plt.tight_layout()
# plt.show()


#Below is using PLOTLY


# best_chromosome,_ = grid_based_pack(boxes)


# box_data = [
#     (*box['position'], *box['dimensions'], box['colour'])
#     for box in best_chromosome
# ]

# container_mesh, container_edges = create_ld1('lightgray')

# traces = [container_mesh, container_edges]

# for x, y, z, dx, dy, dz, color in box_data: 
#     traces.append(create_box(x, y, z, dx, dy, dz, color=color, opacity=1.0, name="Box"))

# fig = go.Figure(data=traces)       
# fig.update_layout(
#     scene=dict(
#         xaxis=dict(nticks=10, range=[0, 100], backgroundcolor="white"),
#         yaxis=dict(nticks=10, range=[0, 65], backgroundcolor="white"),
#         zaxis=dict(nticks=10, range=[0, 70], backgroundcolor="white"),
#         aspectmode='data'
#     ),
#     margin=dict(l=0, r=0, t=0, b=0),
# )

# # Open in browser
# plot(fig, filename='3d_boxes.html', auto_open=True)

