import json
import matplotlib.pyplot as plt
from packing import draw_uld, layer_based_pack

# Load the saved chromosome (box order + rotations)
with open("packed_result.json", "r") as f:
    best_chromosome = json.load(f)

box_dims = [tuple(gene) for gene in best_chromosome]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

draw_uld(ax)
layer_based_pack(ax=ax, box_dims_list=box_dims, step=0.5)

# Plot settings
ax.set_xlabel('X (Width)')
ax.set_ylabel('Y (Depth)')
ax.set_zlabel('Z (Height)')
ax.set_xlim(0, 100)
ax.set_ylim(0, 70)
ax.set_zlim(0, 70)
ax.set_title('Final Packed Result from Genetic Algorithm')
ax.view_init(elev=25, azim=35)
plt.tight_layout()
plt.show()
