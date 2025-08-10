import plotly.graph_objects as go
from plotly.offline import plot

def create_box(x, y, z, dx, dy, dz, color, opacity=1.0, name=""):
    # Returns a Mesh3d representing a box
    vertices = [
        [x,     y,     z],
        [x+dx,  y,     z],
        [x+dx,  y+dy,  z],
        [x,     y+dy,  z],
        [x,     y,     z+dz],
        [x+dx,  y,     z+dz],
        [x+dx,  y+dy,  z+dz],
        [x,     y+dy,  z+dz],
    ]
    x_vals, y_vals, z_vals = zip(*vertices)
    
    I = [0, 1, 0, 1, 2, 2, 1, 1, 5, 4, 3, 0]
    J = [1, 2, 1, 4, 3, 6, 2, 5, 6, 5, 4, 3]
    K = [3, 3, 4, 5, 7, 7, 6, 6, 7, 7, 7, 4]

    mesh = go.Mesh3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        i=I, j=J, k=K,
        color=color,
        opacity=opacity,
        name=name,
        showscale=False
    )
    return mesh

def create_ld1_edges(vertices, line_color='black', line_width=2):
    # Define edge pairs (12 for a cube, more for complex shapes)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (3, 7), (2, 8), (1, 9),  # vertical edges
        (5, 9), (6, 8), (8, 9)    # ULD slant face
    ]
    
    lines = []
    for i, j in edges:
        x = [vertices[i][0], vertices[j][0], None]
        y = [vertices[i][1], vertices[j][1], None]
        z = [vertices[i][2], vertices[j][2], None]
        lines.extend([x, y, z])

    # Flatten line coordinates
    x_vals = [pt for edge in edges for pt in [vertices[edge[0]][0], vertices[edge[1]][0], None]]
    y_vals = [pt for edge in edges for pt in [vertices[edge[0]][1], vertices[edge[1]][1], None]]
    z_vals = [pt for edge in edges for pt in [vertices[edge[0]][2], vertices[edge[1]][2], None]]

    return go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='lines',
        line=dict(color=line_color, width=line_width),
        showlegend=False
    )


def create_ld6_edges(vertices, line_color='black', line_width=2):
    # Define edge pairs (12 for a cube, more for complex shapes)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (3, 7), (1, 5), (2, 6),  # vertical edges
        (4, 8), (5, 9), (6, 10), (7, 11), (8, 9), (9, 10), (10, 11), (8, 11)  
    ]
    
    lines = []
    for i, j in edges:
        x = [vertices[i][0], vertices[j][0], None]
        y = [vertices[i][1], vertices[j][1], None]
        z = [vertices[i][2], vertices[j][2], None]
        lines.extend([x, y, z])

    # Flatten line coordinates
    x_vals = [pt for edge in edges for pt in [vertices[edge[0]][0], vertices[edge[1]][0], None]]
    y_vals = [pt for edge in edges for pt in [vertices[edge[0]][1], vertices[edge[1]][1], None]]
    z_vals = [pt for edge in edges for pt in [vertices[edge[0]][2], vertices[edge[1]][2], None]]

    return go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='lines',
        line=dict(color=line_color, width=line_width),
        showlegend=False
    )

def create_container(color = "blue", origin=(0, 0, 0), opacity=0.15, name=""):
    # Returns a Mesh3d representing a box
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
    x_vals, y_vals, z_vals = zip(*vertices)
    

    I = [0, 0, 8, 8, 8, 0, 0, 0, 4, 4, 3, 0, 5, 5, 1, 1]
    J = [1, 2, 2, 6, 3, 1, 5, 4, 6, 5, 4, 3, 6, 8, 2, 8]
    K = [2, 3, 3, 7, 7, 9, 9, 5, 7, 6, 7, 4, 8, 9, 8, 9]

    mesh = go.Mesh3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        i=I, j=J, k=K,
        color=color,
        opacity=opacity,
        name=name,
        showscale=False
    )
    
    edges = create_ld1_edges(vertices)
    return mesh, edges


def create_ld6(color = "blue", origin=(0, 0, 0), opacity=0.3, name=""):
    # Returns a Mesh3d representing a box
    x, y, z = origin
    vertices = [
        [x+17.5,     y,     z],
        [x+142.5,  y,     z],
        [x+142.5,  y+60.4,  z],
        [x+17.5,   y+60.4,  z],
        [x,     y,     z+21.33],
        [x+160,  y,     z+21.33],
        [x+160,  y+60.4,  z+21.33],
        [x,     y+60.4,  z+21.33],
        [x,  y,  z+64],
        [x+160,  y,  z+64],
        [x+160,  y+60.4,  z+64],
        [x,  y+60.4,  z+64]
    ]
    x_vals, y_vals, z_vals = zip(*vertices)
    

    I = [0, 0, 0, 1, 2, 3, 1, 2, 0, 3, 4, 5, 6, 7, 5, 6, 4, 7, 8, 9]
    J = [1, 2, 1, 4, 3, 6, 2, 5, 3, 4, 5, 8, 7, 10, 6, 9, 7, 8, 9, 10]
    K = [2, 3, 4, 5, 6, 7, 5, 6, 4, 7, 8, 9, 10, 11, 9, 10, 8, 11, 11, 11]

    mesh = go.Mesh3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        i=I, j=J, k=K,
        color=color,
        opacity=opacity,
        name=name,
        showscale=False
    )
    
    edges = create_ld6_edges(vertices)
    return mesh, edges
# Dimensions of the big box
#container = create_box(0, 0, 0, 10, 10, 10, color='lightgray', opacity=0.1, name="Container")

# Smaller boxes inside
box1 = create_box(1, 1, 1, 3, 3, 3, color='red', opacity=1.0, name="Box 1")
box2 = create_box(5, 1, 1, 2, 4, 2, color='blue', opacity=1.0, name="Box 2")
box3 = create_box(2, 5, 2, 4, 2, 3, color='green', opacity=1.0, name="Box 3")

container_mesh, container_edges = create_ld6('lightgray')

fig = go.Figure(data=[container_mesh, container_edges, box1, box2, box3])
fig.update_layout(
    scene=dict(
        xaxis=dict(nticks=10, range=[0, 180], backgroundcolor="white"),
        yaxis=dict(nticks=10, range=[0, 65], backgroundcolor="white"),
        zaxis=dict(nticks=10, range=[0, 70], backgroundcolor="white"),
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, t=0, b=0),
)

# Open in browser
#plot(fig, filename='3d_boxes.html', auto_open=True)
