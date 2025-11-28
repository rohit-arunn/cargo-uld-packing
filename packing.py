import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from functools import lru_cache
from itertools import permutations
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

   #{'box_id': 36, 'dimensions': (10, 20, 10), 'number': 10}, {'box_id': 26, 'dimensions': (5, 10, 5), 'number': 40}



@lru_cache(maxsize=None)
def is_point_inside_ld1(x, y, z):
    if (0 <= x < 61.5 and 0 <= y <= 60.4 and 0 <= z <= 64):
        return True
    elif x >= 61.5:
        z_limit = 0.6993 * x - 43.0056
        return z_limit < z < 64 and x < 92 and 0 <= y <= 60.4
    else:
        return False
    
@lru_cache(maxsize=None)
def is_box_inside_ld1(x, y, z, dx, dy, dz):
    for dx_i in [0, dx]:
        for dy_i in [0, dy]:
            for dz_i in [0, dz]:
                if not is_point_inside_ld1(x + dx_i, y + dy_i, z + dz_i):
                    return False
    return True


@lru_cache(maxsize=None)
def is_point_inside_ld3(x, y, z):
    if (0 <= x < 61.5 and 0 <= y <= 60.4 and 0 <= z <= 64):
        return True
    elif x >= 61.5:
        z_limit = 1.22 * x - 75
        return z_limit < z < 64 and x < 79 and 0 <= y <= 60.4
    else:
        return False
    
@lru_cache(maxsize=None)
def is_box_inside_ld3(x, y, z, dx, dy, dz):
    for dx_i in [0, dx]:
        for dy_i in [0, dy]:
            for dz_i in [0, dz]:
                if not is_point_inside_ld3(x + dx_i, y + dy_i, z + dz_i):
                    return False
    return True

@lru_cache(maxsize=None)
def is_point_inside_ld6(x, y, z):


    if x <= 17.5:
        z_limit = -1.21 * x + 21.33
        return z_limit < z < 64 and 0 <= y <= 60.4
    if (17.5 <= x < 142.5 and 0 <= y <= 60.4 and 0 <= z <= 64):
        return True
    elif x >= 142.5:
        z_limit = 1.21 * x - 172.425
        return z_limit < z < 64 and x < 160 and 0 <= y <= 60.4
    else:
        return False
    
@lru_cache(maxsize=None)
def is_box_inside_ld6(x, y, z, dx, dy, dz):
    for dx_i in [0, dx]:
        for dy_i in [0, dy]:
            for dz_i in [0, dz]:
                if not is_point_inside_ld6(x + dx_i, y + dy_i, z + dz_i):
                    return False
    return True

@lru_cache(maxsize=None)
def is_point_inside_ld8(x, y, z):
    if x <= 14.5:
        z_limit = -1.44 * x + 21
        return z_limit < z < 64 and 0 <= y <= 60.4
    if (14.5 <= x < 110.5 and 0 <= y <= 60.4 and 0 <= z <= 64):
        return True
    elif x >= 110.5:
        z_limit = 1.45 * x - 160.03
        return z_limit < z < 64 and x < 125 and 0 <= y <= 60.4
    else:
        return False
    
@lru_cache(maxsize=None)
def is_box_inside_ld8(x, y, z, dx, dy, dz):
    for dx_i in [0, dx]:
        for dy_i in [0, dy]:
            for dz_i in [0, dz]:
                if not is_point_inside_ld6(x + dx_i, y + dy_i, z + dz_i):
                    return False
    return True


@lru_cache(maxsize=None)
def is_point_inside_ld9(x, y, z):
    if (0 <= x < 125 and 0 <= y <= 88 and 0 <= z <= 64):
        return True
    else:
        return False
    
@lru_cache(maxsize=None)
def is_box_inside_ld9(x, y, z, dx, dy, dz):
    for dx_i in [0, dx]:
        for dy_i in [0, dy]:
            for dz_i in [0, dz]:
                if not is_point_inside_ld9(x + dx_i, y + dy_i, z + dz_i):
                    return False
    return True

def boxes_overlap(box1, box2):
    x1, y1, z1, dx1, dy1, dz1 = box1
    x2, y2, z2, dx2, dy2, dz2 = box2

    return not (
        x1 + dx1 <= x2 or x2 + dx2 <= x1 or
        y1 + dy1 <= y2 or y2 + dy2 <= y1 or
        z1 + dz1 <= z2 or z2 + dz2 <= z1
    )

def collides_with_existing(x, y, z, dx, dy, dz, placed_boxes):
    new_box = (x, y, z, dx, dy, dz)
    for b in placed_boxes:
        bx, by, bz = b['position']
        bdx, bdy, bdz = b['dimensions']
        existing_box = (bx, by, bz, bdx, bdy, bdz)
        if boxes_overlap(new_box, existing_box):
            return True
    return False

def draw_uld(ax, origin=(0, 0, 0), color='skyblue'):
    x, y, z = origin
    vertices = [
        [x,     y,     z],
        [x+61,  y,     z],
        [x+61,  y+60.4,  z],
        [x,     y+60.4,  z],
        [x,     y,     z+64],
        [x+92,  y,     z+64],
        [x+92,  y+60.4,  z+64],
        [x,     y+60.4,  z+64],
        [x+92,  y+60.4,  z+21.33],
        [x+92,  y,  z+21.33]
    ]
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[9], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6], vertices[8]],
        [vertices[1], vertices[2], vertices[8], vertices[9]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[5], vertices[6], vertices[8], vertices[9]]
    ]
    poly3d = Poly3DCollection(faces, facecolors=color, edgecolors='k', linewidths=1, alpha=0.3)
    ax.add_collection3d(poly3d)

def draw_box(ax, x, y, z, dx, dy, dz, color='orange'):
    if ax is None:
        return  
    corners = np.array([
        [x, y, z],
        [x + dx, y, z],
        [x + dx, y + dy, z],
        [x, y + dy, z],
        [x, y, z + dz],
        [x + dx, y, z + dz],
        [x + dx, y + dy, z + dz],
        [x, y + dy, z + dz]
    ])
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],
        [corners[4], corners[5], corners[6], corners[7]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[2], corners[3], corners[7], corners[6]],
        [corners[1], corners[2], corners[6], corners[5]],
        [corners[0], corners[3], corners[7], corners[4]]
    ]
    box = Poly3DCollection(faces, facecolors=color, edgecolors='k', linewidths=0.5, alpha=0.9)
    ax.add_collection3d(box)


def is_supported(x, y, z, dx, dy, dz, placed_boxes, support_threshold=0.95):
    if z <= 0.01:
        return True

    box_area = dx * dy
    supported_area = 0

    for other in placed_boxes:
        ox, oy, oz = other['position']
        odx, ody, odz = other['dimensions']

        # Only consider boxes directly below
        if abs((oz + odz) - z) > 1e-2:
            continue

        # Find overlap area in x-y plane
        overlap_x = max(0, min(x + dx, ox + odx) - max(x, ox))
        overlap_y = max(0, min(y + dy, oy + ody) - max(y, oy))

        overlap_area = overlap_x * overlap_y
        supported_area += overlap_area

    support_ratio = supported_area / box_area
    return support_ratio >= support_threshold


# def is_supported(x, y, z, dx, dy, dz, placed_boxes, support_threshold=0.5):
#     # Base layer is always supported
#     if z <= 0.01:
#         return True

#     supported_area = 0
#     box_area = dx * dy

#     for px in range(int(x), int(x + dx), 2):
#         for py in range(int(y), int(y + dy), 2):
#             # Check each point on the base of the box
#             point_supported = False
#             for other in placed_boxes:
#                 ox, oy, oz = other['position']
#                 odx, ody, odz = other['dimensions']
#                 top_z = oz + odz

#                 if abs(z - top_z) <= 1e-2:  # Aligned on top surface
#                     if ox <= px < ox + odx and oy <= py < oy + ody:
#                         point_supported = True
#                         break

#             if point_supported:
#                 supported_area += 1

#     support_ratio = supported_area / box_area

#     return support_ratio >= support_threshold

def flat_rotations(dims):
    rotations = []
    for perm in permutations(dims):
        dx, dy, dz = perm
        if dz == min(perm):  # Only allow when height is shortest
            rotations.append((dx, dy, dz))
    return rotations


  
def best_stack_arrangement(box_dims, count):
    l, w, h = box_dims
    min_height = float('inf')
    best_config = None

    for x in range(1, count + 1):
        for y in range(1, count + 1):
            if x * y > count:
                break
            if count % (x * y) != 0:
                continue

            z = count // (x * y)  # Number of vertical layers

            total_h = h * z  # Try to minimize this
            total_l = l * x
            total_w = w * y

            # Optional: prevent overly long rows (e.g., x or y too large)
            if max(total_l, total_w) > 2 * max(l, w):
                continue

            if total_h < min_height:
                min_height = total_h
                best_config = (x, y, z)

    return best_config, min_height




# ----------- BLB Packing (No Overlap) -----------
# def layer_based_pack(ax, box_dims_list, container_dims=(92, 60.4, 64), layer_height_step=2, step=0.5):
#     container_length, container_width, container_height = container_dims
#     placed_boxes = []

#     # Sort boxes by height descending (heavier/larger boxes at the bottom if you associate height with weight)
#     #box_dims_list = sorted(box_dims_list, key=lambda b: max(b), reverse=True)

#     z = 0  
#     while z < container_height:
#         remaining_boxes = []
#         for original_dims in box_dims_list:
            
#             rotations = [
#                 (original_dims[0], original_dims[1], original_dims[2]),
#                 (original_dims[0], original_dims[2], original_dims[1]),
#                 (original_dims[1], original_dims[0], original_dims[2]),
#                 (original_dims[1], original_dims[2], original_dims[0]),
#                 (original_dims[2], original_dims[0], original_dims[1]),
#                 (original_dims[2], original_dims[1], original_dims[0]),
#             ]
#             placed = False
#             for dims in rotations:
#                 dx, dy, dz = dims
#                 if dz + z > container_height:
#                     continue  # Box too tall for current or next layer
#                 for y in np.arange(0, container_width - dy + 1, step):
#                     for x in np.arange(0, container_length - dx + 1, step):
#                         if (is_box_inside_uld(x, y, z, dx, dy, dz) and not collides_with_existing(x, y, z, dx, dy, dz, placed_boxes) 
#                             and is_supported(x, y, z, dx, dy, dz, placed_boxes)):
#                             draw_box(ax, x, y, z, dx, dy, dz)
#                             placed_boxes.append((box_id, x, y, z, dx, dy, dz))
#                             placed = True
#                             break
#                     if placed: break
#                 if placed: break
#             if not placed:
#                 remaining_boxes.append(original_dims)

#         box_dims_list = remaining_boxes  # Remaining boxes for next layer 
#         z += layer_height_step  # Move up to next layer

#     return placed_boxes 

def layer_based_pack(box_list, container_dims=(92, 60.4, 64), layer_height_step=1, step=1):
    container_length, container_width, container_height = container_dims
    placed_boxes = []

    z = 0
    while z < container_height:
        remaining_boxes = []

        for box in box_list:
            box_id = box['box_id']
            original_dims = box['dimensions']
            number_of_pieces = box.get('number', 1)
            color = box.get('colour', (random.random(), random.random(), random.random()))


            rotations = flat_rotations(original_dims)
            # rotations = [
            #     (original_dims[0], original_dims[1], original_dims[2]),
            #     (original_dims[0], original_dims[2], original_dims[1]),
            #     (original_dims[1], original_dims[0], original_dims[2]),
            #     (original_dims[1], original_dims[2], original_dims[0]),
            #     (original_dims[2], original_dims[0], original_dims[1]),
            #     (original_dims[2], original_dims[1], original_dims[0]),
            # ]

            placed_count = 0

            for j in range(number_of_pieces):
                
                placed = False
                for dims in rotations:
                    dx, dy, dz = dims
                    if dz + z > container_height:
                        continue

                    for x in np.arange(0, container_length - dx + 1, step):
                        for y in np.arange(0, container_width - dy + 1, step):
                            if (is_box_inside_ld1(x, y, z, dx, dy, dz)
                                    and not collides_with_existing(x, y, z, dx, dy, dz, placed_boxes)
                                    #and is_supported(x, y, z, dx, dy, dz, placed_boxes) 
                                ):

                                

                                placed_boxes.append({
                                    'box_id': box_id,
                                    'position': (x, y, z),
                                    'dimensions': (dx, dy, dz),
                                    'colour': color
                                })

                                placed = True
                                break
                        if placed:
                            break
                    if placed:
                        placed_count += 1
                        break  
                if not placed:
                    break  

            if placed_count < number_of_pieces:
                remaining_boxes.append({
                    'box_id': box_id,
                    'dimensions': original_dims,
                    'number': number_of_pieces - placed_count,
                    'colour': color
                })

        box_list = remaining_boxes
        z += layer_height_step

    return placed_boxes



#This is not the best layer for fitness, they are computer science students they are good ennu paranj, man those 2 cracks were the best, 

# def layer_based_pack(box_list, ax=None, container_dims=(92, 60.4, 64), layer_height_step = 0.5, step=0.2):
#     container_length, container_width, container_height = container_dims
#     placed_boxes = []

#     z = 0
#     while z < container_height:
#         remaining_boxes = []
#         max_layer_height = 0

#         for box in box_list:
#             box_id = box['box_id']
#             original_dims = box['dimensions']
#             numpcs = box.get('number', 1)
#             color = (random.random(), random.random(), random.random())

#             # Get best x, y, z stacking
#             x_count, y_count, z_count = best_stack_arrangement(original_dims, numpcs)[0]
#             lx, ly, lz = original_dims
#             total_dx = lx * x_count
#             total_dy = ly * y_count
#             total_dz = lz * z_count

#             placed = False
#             for y in np.arange(0, container_width - total_dy + 1, step):
#                 for x in np.arange(0, container_length - total_dx + 1, step):
#                     if (z + total_dz <= container_height and
#                         is_box_inside_uld(x, y, z, total_dx, total_dy, total_dz) and
#                         not collides_with_existing(x, y, z, total_dx, total_dy, total_dz, placed_boxes) and
#                         is_supported(x, y, z, total_dx, total_dy, total_dz, placed_boxes)):

#                         # Place each small box in its stacked position
#                         for i in range(x_count):
#                             for j in range(y_count):
#                                 for k in range(z_count):
#                                     bx = x + i * lx
#                                     by = y + j * ly
#                                     bz = z + k * lz 
#                                     placed_boxes.append({
#                                         'box_id': box_id,
#                                         'position': (bx, by, bz),
#                                         'dimensions': original_dims, 
#                                         'colour': color

#                                     })
#                                     max_layer_height = max(max_layer_height, total_dz)

#                         placed = True
#                         break
#                 if placed: break

#             if not placed:
#                 remaining_boxes.append(box)

#         box_list = remaining_boxes
#         if max_layer_height == 0:
#             # Prevent infinite loop
#             z += layer_height_step
#         else:
#             z += max_layer_height 

#     return placed_boxes

