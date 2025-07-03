





# Function to check whether a point is inside the AKC

def is_point_inside_uld(x, y, z):

    if (0 <= x < 61.5 and 0 <= y <= 60.4 and 0 <= z <= 64):
        return True
    elif x >= 61.5:
        z_limit = 0.6993 * x - 43.0056
        if (z_limit < z < 64 and x < 92 and 0 <= y <= 60.4):
            return True
        else:
            return False
    else:
        return False
    
# Function to check whether the box is inside the ULD by considering the above function for each corner

def is_box_inside_uld(x, y, z, dx, dy, dz):
    for dx_i in [0, dx]:
        for dy_i in [0, dy]:
            for dz_i in [0, dz]:
                if not is_point_inside_uld(x + dx_i, y + dy_i, z + dz_i):
                    return False
    return True