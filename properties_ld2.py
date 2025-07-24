import numpy as np
import pandas as pd
from functools import lru_cache
import matplotlib.pyplot as plt
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

   #{'box_id': 36, 'dimensions': (10, 20, 10), 'number': 30} 



@lru_cache(maxsize=None)
def is_point_inside_ld2(x, y, z):
    if (0 <= x < 47 and 0 <= y <= 60.4 and 0 <= z <= 64):
        return True
    elif x >= 47:
        z_limit = 1.10345 * x - 51.86215
        return z_limit < z < 64 and x < 61.5 and 0 <= y <= 60.4
    else:
        return False
    
@lru_cache(maxsize=None)
def is_box_inside_ld2(x, y, z, dx, dy, dz):
    for dx_i in [0, dx]:
        for dy_i in [0, dy]:
            for dz_i in [0, dz]:
                if not is_point_inside_ld2(x + dx_i, y + dy_i, z + dz_i):
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

def draw_uld_ld2(ax, origin=(0, 0, 0), color='skyblue'):
    x, y, z = origin
    vertices = [
        [x,     y,     z],
        [x+47,  y,     z],
        [x+47,  y+60.4,  z],
        [x,     y+60.4,  z],
        [x,     y,     z+64],
        [x+61.5,  y,     z+64],
        [x+61.5,  y+60.4,  z+64],
        [x,     y+60.4,  z+64],
        [x+61.5,  y+60.4,  z+16],
        [x+61.5,  y,  z+16]
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




# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# draw_uld(ax)





#Axis setup
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