from PIL import Image

class PathPlanner:
    def __init__(self):
        # load map
        return

    def plan(self, current_loc, destination_loc):
        """
        current_loc: coordination in grid map 
        destination_loc: coordination in grid map
        for exmaple:
        current_loc: (0, 2)
        destination_loc: (2, 3)

        output: the path list
        for example: [(0, 2), (1, 2), (2, 2), (2, 3)]
        """
        path = []
        current_loc = list(current_loc)

        cur = current_loc
        path.append(cur.copy())
        while cur[0] != destination_loc[0]:
            if cur[0] < destination_loc[0]:
                cur[0] += 1
            else:
                cur[0] -= 1
            path.append(cur.copy())
        while cur[1] != destination_loc[1]:
            if cur[1] < destination_loc[1]:
                cur[1] += 1
            else:
                cur[1] -= 1
            path.append(cur.copy())
        print(path)
        return path
    
