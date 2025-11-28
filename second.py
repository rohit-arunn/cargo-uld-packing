import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from packing import is_box_inside_uld
from packing import is_point_inside_uld
from packing_ld1 import get_unique_rotations, is_supported_in_grid
from packing import is_supported
from packing import collides_with_existing, draw_box, draw_uld

        # print(f"Generation {gen+1}: Best Fitness = {best_fit_local:.4f}")

        # best_fit_local = max(fitnesses_main)  # assuming it's a list of floats
        # print(f"Generation {gen+1}: Best Fitness = {best_fit_local:.4f}")


boxes = [
    {'box_id': 1, 'dimensions': (20, 10, 10)},
    {'box_id': 2, 'dimensions': (10, 20, 10)},
    {'box_id': 3, 'dimensions': (10, 10, 20)},
    {'box_id': 4, 'dimensions': (12, 12, 12)},
    {'box_id': 5, 'dimensions': (10, 10, 10)},
    {'box_id': 6, 'dimensions': (15, 15, 15)},
    {'box_id': 7, 'dimensions': (20, 10, 10)},
    {'box_id': 8, 'dimensions': (10, 20, 10)},
    {'box_id': 9, 'dimensions': (10, 10, 20)},
    {'box_id': 10, 'dimensions': (12, 12, 12)},
    {'box_id': 11, 'dimensions': (10, 10, 10)},
    {'box_id': 12, 'dimensions': (15, 15, 15)},
    {'box_id': 13, 'dimensions': (20, 10, 10)},
    {'box_id': 14, 'dimensions': (10, 20, 10)},
    {'box_id': 15, 'dimensions': (15, 15, 15)},
    {'box_id': 16, 'dimensions': (20, 10, 10)},
    {'box_id': 17, 'dimensions': (10, 20, 10)}
]




def layer_based_pack(box_list, container_dims=(92, 60.4, 64), layer_height_step=10, step=1):
    container_length, container_width, container_height = container_dims
    placed_boxes = []

    z = 0
    while z < container_height:
        remaining_boxes = []
        for box in box_list:
            box_id = box['box_id']
            original_dims = box['dimensions']

            rotations = [
                (original_dims[0], original_dims[1], original_dims[2]),
                (original_dims[0], original_dims[2], original_dims[1]),
                (original_dims[1], original_dims[0], original_dims[2]),
                (original_dims[1], original_dims[2], original_dims[0]),
                (original_dims[2], original_dims[0], original_dims[1]),
                (original_dims[2], original_dims[1], original_dims[0]),
            ]
            placed = False
            for dims in rotations:
                dx, dy, dz = dims
                if dz + z > container_height:
                    continue
                for y in np.arange(0, container_width - dy + 1, step):
                    for x in np.arange(0, container_length - dx + 1, step):
                        if (is_box_inside_uld(x, y, z, dx, dy, dz)
                                and not collides_with_existing(x, y, z, dx, dy, dz, placed_boxes)
                                and is_supported(x, y, z, dx, dy, dz, placed_boxes)):
                            
                            # if ax:  # Only draw if an axis is provided
                            #     draw_box(ax, x, y, z, dx, dy, dz)

                            placed_boxes.append({
                                'box_id': box_id,
                                'position': (x, y, z),
                                'dimensions': (dx, dy, dz)
                            })
                            placed = True
                            break
                    if placed: break
                if placed: break
            if not placed:
                remaining_boxes.append(box)

        box_list = remaining_boxes
        z += layer_height_step

    return placed_boxes

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
        color = box.get('colour', (random.random(), random.random(), random.random()))
        
        rotations = get_unique_rotations(original_dims)    

        print("rotations - ", rotations)    

        placed_count = 0

        for _ in range(numpcs):
            placed = False

            orientations = rotations[0]

            dx, dy, dz = orientations
            
            for z in range(lz - dz + 1):
                for y in range(ly - dy + 1):
                    for x in range(lx - dx + 1):
                        for i in range(len(rotations)):

                            dx, dy, dz = rotations[i]
                        
                            if (np.all(grid[z:z+dz, y:y+dy, x:x+dx] == 0) and
                                is_box_inside_uld(x, y, z, dx, dy, dz) and
                                is_supported_in_grid(x, y, z, dx, dy, dz, grid, threshold=0.7)
                                    ):
                            
                                grid[z:z+dz, y:y+dy, x:x+dx] = 1

                                # Converting back to real coordinates 
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



packed_container = layer_based_pack(boxes, step=0.5)
print(packed_container)


# #----------- Plotting -----------
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# draw_uld(ax)

# for box in packed_container:
#     x, y, z = box['position']
#     dx, dy, dz = box['dimensions']
#     draw_box(ax, x, y, z, dx, dy, dz)



# #Axis setup
# ax.set_xlabel('X (Width)')
# ax.set_ylabel('Y (Depth)')
# ax.set_zlabel('Z (Height)')
# ax.set_xlim(0, 100)
# ax.set_ylim(0, 70)
# ax.set_zlim(0, 70)
# ax.set_title('BLB Packing inside LD1 ULD (No Overlap)')
# ax.view_init(elev=25, azim=35)
# plt.tight_layout()
# plt.show()             

