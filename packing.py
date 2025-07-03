import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



@lru_cache(maxsize=None)
def is_point_inside_uld(x, y, z):
    if (0 <= x < 61.5 and 0 <= y <= 60.4 and 0 <= z <= 64):
        return True
    elif x >= 61.5:
        z_limit = 0.6993 * x - 43.0056
        return z_limit < z < 64 and x < 92 and 0 <= y <= 60.4
    else:
        return False
    
@lru_cache(maxsize=None)
def is_box_inside_uld(x, y, z, dx, dy, dz):
    for dx_i in [0, dx]:
        for dy_i in [0, dy]:
            for dz_i in [0, dz]:
                if not is_point_inside_uld(x + dx_i, y + dy_i, z + dz_i):
                    return False
    return True

def boxes_overlap(box1, box2):
    x1, y1, z1, dx1, dy1, dz1 = box1
    x2, y2, z2, dx2, dy2, dz2 = box2

    return not (
        (x1 + dx1 <= x2 or x2 + dx2 <= x1) and
        (y1 + dy1 <= y2 or y2 + dy2 <= y1) and
        (z1 + dz1 <= z2 or z2 + dz2 <= z1)
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
        return  # Don't draw if no plotting axis
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

def is_supported(x, y, z, dx, dy, dz, placed_boxes, support_threshold=0.8):
    # Base layer is always supported
    if z <= 0.01:
        return True

    supported_area = 0
    box_area = dx * dy

    for px in range(int(x), int(x + dx), 2):
        for py in range(int(y), int(y + dy), 2):
            # Check each point on the base of the box
            point_supported = False
            for other in placed_boxes:
                ox, oy, oz = other['position']
                odx, ody, odz = other['dimensions']
                top_z = oz + odz

                if abs(z - top_z) <= 1e-2:  # Aligned on top surface
                    if ox <= px < ox + odx and oy <= py < oy + ody:
                        point_supported = True
                        break

            if point_supported:
                supported_area += 1

    support_ratio = supported_area / box_area

    return support_ratio >= support_threshold
  


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

def layer_based_pack(box_list, container_dims=(92, 60.4, 64), layer_height_step=2, step=1):
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



# ----------- Plotting -----------
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# draw_uld(ax)



# layer_based_pack(ax, boxes, step=0.5)

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

