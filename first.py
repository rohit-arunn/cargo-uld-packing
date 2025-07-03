import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_cuboid(ax, origin, color='skyblue'):
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

    # The seven faces of the ULD are mentioned below with the order of vertices of each face going in a circle. 
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  
        [vertices[4], vertices[5], vertices[6], vertices[7]],  
        [vertices[0], vertices[1], vertices[9], vertices[5], vertices[4]],  
        [vertices[2], vertices[3], vertices[7], vertices[6], vertices[8]],  
        [vertices[1], vertices[2], vertices[8], vertices[9]],  
        [vertices[0], vertices[3], vertices[7], vertices[4]], 
        [vertices[5], vertices[6], vertices[8], vertices[9]]   
    ]     

    # Draw the cuboid using Poly3DCollection
    poly3d = Poly3DCollection(faces, facecolors=color, edgecolors='k', linewidths=1, alpha=0.8)
    ax.add_collection3d(poly3d)


    
# === Plot setup ===
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Call the function to draw a cuboid
draw_cuboid(ax, origin=(0, 0, 0))  


# Configure axes
# Coordinates of the point
spot_x, spot_y, spot_z = 70, 10, 20

# Add a darkened spot using scatter
ax.scatter(spot_x, spot_y, spot_z, color='black', s=20, label='Point of Interest')  # s is the size


ax.set_xlabel('X-axis (Width)')
ax.set_ylabel('Y-axis (Depth)')
ax.set_zlabel('Z-axis (Height)')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)
ax.set_title('LD1 ULD')
ax.view_init(elev=25, azim=35)
plt.tight_layout()
plt.show()
