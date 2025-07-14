import numpy as np
import math
import pandas as pd
import random
from itertools import permutations
import matplotlib.pyplot as plt
from packing import draw_box, is_box_inside_uld, is_point_inside_uld
from packing import draw_uld
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

    boxes.append({
        'box_id': box_id,
        'dimensions': (length, width, height),
        'number' : numpcs
    })


def flat_rotations(dims, grid_step):
    # shortest dimension must be height
    return {
        tuple(math.ceil(d / grid_step) for d in p)
        for p in permutations(dims)
        if p[2] == min(p)
    }


# def is_supported_in_grid(x, y, z, dx, dy, dz, grid, threshold=0.95):
#     if z == 0:
#         return True  # Base layer is always supported

#     support_area = grid[z - 1, y:y + dy, x:x + dx]
#     total_cells = support_area.size
#     filled_cells = np.count_nonzero(support_area == 1)

#     support_ratio = filled_cells / total_cells

#     return support_ratio >= threshold

def is_supported_in_grid(x, y, z, dx, dy, dz, grid, threshold=1):
    if z == 0:
        return True  # Base layer is always supported

    support_area = grid[z - 1, y:y + dy, x:x + dx]
    total_cells = support_area.size
    filled_cells = np.count_nonzero(support_area == 1)

    support_ratio = filled_cells / total_cells

    return support_ratio >= threshold

# def grid_based_pack(box_list, container_dims=(92, 60.4, 64), grid_step=1):
#     container_length, container_width, container_height = container_dims
#     # Convert dimensions to grid units
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
#         color = box.get('colour', (random.random(), random.random(), random.random()))

#         # Convert dimensions to grid steps
#         rotations = sorted(
#         [tuple(int(d / grid_step) for d in p) for p in permutations(original_dims)],
#         key=lambda dim: dim[2]
#         )

#         placed_count = 0

#         for _ in range(numpcs):
#             placed = False

#             for dx, dy, dz in rotations:
#                 # Iterate layer by layer from bottom up
#                 for z in range(0, lz - dz + 1):
#                     for y in range(0, ly - dy + 1):
#                         for x in range(0, lx - dx + 1):
#                             # Check if space is free
#                             if np.all(grid[z:z+dz, y:y+dy, x:x+dx] == 0):
#                                  if (is_box_inside_uld(x, y, z, dx, dy, dz) and is_supported_in_grid(x, y, z, dx, dy, dz, grid, threshold=0.8)
#                                      ):
#                                 # Place box
#                                     grid[z:z+dz, y:y+dy, x:x+dx] = 1

#                                     # Convert back to real coordinates
#                                     px, py, pz = x * grid_step, y * grid_step, z * grid_step
#                                     real_dims = (dx * grid_step, dy * grid_step, dz * grid_step)

#                                     placed_boxes.append({
#                                         'box_id': box_id,
#                                         'position': (px, py, pz),
#                                         'dimensions': real_dims,
#                                         'colour': color
#                                     })

#                                     placed = True
#                                     break
#                         if placed: break
#                     if placed: break
#                 if placed:
#                     placed_count += 1
#                     break

#             # Not all pieces could be placed, requeue
#             remaining = numpcs - placed_count
#             if remaining > 0:
#                 next_box_list.append({
#                     'box_id': box_id,
#                     'dimensions': original_dims,
#                     'number': remaining,
#                     'colour': color
#                 })
#             box_list = next_box_list

#     return placed_boxes


def grid_based_pack(box_list, container_dims=(92, 60.4, 64), grid_step=1):
    container_length, container_width, container_height = container_dims
    # Convert dimensions to grid units
    lx = int(container_length / grid_step)
    ly = int(container_width / grid_step)
    lz = int(container_height / grid_step)

    # Initialize 3D occupancy grid
    grid = np.zeros((lz, ly, lx), dtype=np.uint8)
    placed_boxes = []
    next_box_list = []

    for box in box_list:
        box_id = box['box_id']
        original_dims = box['dimensions']
        numpcs = box.get('number', 1)
        color = box.get('colour', (random.random(), random.random(), random.random()))

        # Convert dimensions to grid steps
        rotations = sorted(
        [tuple(int(d / grid_step) for d in p) for p in permutations(original_dims)],
        key=lambda dim: dim[2]
        )

        placed_count = 0

        for _ in range(numpcs):
            placed = False

            for dx, dy, dz in rotations:
                # Iterate layer by layer from bottom up
                for z in range(lz - dz + 1):
                    for y in range(ly - dy + 1):
                        for x in range(lx - dx + 1):
                            # Check if space is free
                            if np.all(grid[z:z+dz, y:y+dy, x:x+dx] == 0):
                                 if (is_box_inside_uld(x, y, z, dx, dy, dz) and is_supported_in_grid(x, y, z, dx, dy, dz, grid, threshold=0.65)):
                                # Place box
                                    grid[z:z+dz, y:y+dy, x:x+dx] = 1

                                    # Convert back to real coordinates
                                    px, py, pz = x * grid_step, y * grid_step, z * grid_step
                                    real_dims = (dx * grid_step, dy * grid_step, dz * grid_step)

                                    placed_boxes.append({
                                        'box_id': box_id,
                                        'position': (px, py, pz),
                                        'dimensions': real_dims,
                                        'colour': color
                                    })

                                    placed = True
                                    break
                        if placed: break
                    if placed: break
                if placed:
                    placed_count += 1
                    break

            # Not all pieces could be placed, requeue
            remaining = numpcs - placed_count
            if remaining > 0:
                next_box_list.append({
                    'box_id': box_id,
                    'dimensions': original_dims,
                    'number': remaining,
                    'colour': color
                })
            box_list = next_box_list

    return placed_boxes


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
# ax.set_title('Packing inside ULD - 2 ')
# ax.view_init(elev=25, azim=35)
# plt.tight_layout()
# plt.show()


