from Assets import wall_hit, check_pixel_color, Colors

class Graph():
    def __init__(self, x_start, y_start, cave_mat):
        self.cave_mat = cave_mat 
        # Initialize a list to hold positions (nodes)
        self.pos = []
        # Set the starting point of the graph
        self.pos.append((x_start,y_start))
    
    # Add the next node to the positions list
    def add_node(self, pos):
        self.pos.append(pos)
    
    # Check if the last added node is valid (WHITE) and if the 
    # connection with the second to last node crosses any cave walls
    def is_valid(self, surface, curr_pos, candidate_pos, step=False):
        if step:
            # Check if the candidate position does not hit a wall
            # and does not cross any obstacles
            if (not wall_hit(self.cave_mat, candidate_pos)
                and not self.cross_obs(*curr_pos, *candidate_pos)):
                    return True
            return False
        else:
            # Check if the candidate position is WHITE (valid),
            # does not hit a wall, and does not cross any obstacles
            if (check_pixel_color(surface, candidate_pos, Colors.WHITE.value)
                and not wall_hit(self.cave_mat, candidate_pos)
                and not self.cross_obs(*curr_pos, *candidate_pos)):
                    return True
            return False
    
    # Check if the connection between the last two nodes crosses the cave walls
    # using Bresenham's line algorithm to find the pixels between the two points
    def cross_obs(self, x1, y1, x2, y2):
        # Initialize deltas and directions for Bresenham's algorithm
        dx = abs(x2 - x1) # Change in x
        sx = 1 if x1<x2 else -1 # Step in the x direction
        dy = -abs(y2 - y1) # Change in y 
        sy = 1 if y1<y2 else -1 # Step in the y direction
        
        # Define the initial error value
        error = dx + dy

        while True:
            # Check for collision at the current pixel
            if wall_hit(self.cave_mat, (x1, y1)):
                return True # Collision detected, return True

            # Check if the end point has been reached (no collisions)
            if x1==x2 and y1==y2: 
                return False # Reached the end point, no collisions

            # Move to the next pixel using Bresenham's algorithm
            err2 = 2*error # Double the error for comparison
            if err2 >= dy:
                if x1==x2: 
                    return False  # Reached the end point, no collisions
                error += dy
                x1 += sx
            if err2 <= dx:
                if y1==y2: 
                    return False # Reached the end point, no collisions
                error += dx
                y1 += sy