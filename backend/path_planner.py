import numpy as np
import heapq
from functools import lru_cache
from PIL import Image, ImageDraw
from utils import *

class Node:
    def __init__(self, x, y, high, cost=0, parent=None):
        self.x = x
        self.y = high - y - 1  # Coordinate transformation to make the origin at the bottom left
        self.cost = cost
        self.parent = parent
        self.high = high

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __lt__(self, other):
        return self.cost < other.cost

    def __add__(self, other):
        return Node(self.x + other[0], self.high - (self.y + other[1]) - 1, self.high, self.cost, self)

    def __sub__(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)

class NodeQueue:
    def __init__(self):
        self.elements = []
        self.elements_set = set()

    def empty(self):
        return len(self.elements) == 0

    def put(self, node):
        heapq.heappush(self.elements, node)
        self.elements_set.add(node)

    def get(self):
        node = heapq.heappop(self.elements)
        self.elements_set.remove(node)
        return node

    def __contains__(self, node):
        return node in self.elements_set

class AStar:
    def __init__(self, start_pos, end_pos, map_array, high):
        self.map_array = map_array
        self.width = map_array.shape[1]
        self.height = map_array.shape[0]
        self.start = Node(*start_pos, high)
        self.end = Node(*end_pos, high)
        self.open_queue = NodeQueue()
        self.came_from = {}
        self.g_score = {self.start: 0}
        self.f_score = {self.start: self.start - self.end}
        self.close_set = set()

    def heuristic(self, a, b):
        return abs(a.x - b.x) + abs(a.y - b.y)

    def _in_map(self, node):
        return 0 <= node.x < self.width and 0 <= node.y < self.height

    def _is_collided(self, node):
        return self.map_array[node.y, node.x] == 1

    def _move(self):
        return [
            (0, 1),  # up
            (1, 0),  # right
            (0, -1), # down
            (-1, 0)  # left
        ]

    def __call__(self):
        self.open_queue.put((self.f_score[self.start], self.start))
        
        while not self.open_queue.empty():
            current = self.open_queue.get()[1]
            
            if current == self.end:
                return self.reconstruct_path(current)
            
            self.close_set.add(current)
            for move in self._move():
                neighbor = current + move
                if neighbor in self.close_set or not self._in_map(neighbor) or self._is_collided(neighbor) or neighbor in self.close_set:
                    continue
                
                tentative_g_score = self.g_score[current] + 1
                if neighbor not in self.g_score or tentative_g_score < self.g_score[neighbor]:
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    self.f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, self.end)
                    if neighbor not in self.open_queue:
                        self.open_queue.put((self.f_score[neighbor], neighbor))
        
        return []

    def reconstruct_path(self, current):
        total_path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            total_path.append(current)
        return total_path[::-1]


    

class PathPlanner:
    def __init__(self, grid):
        self.grid = grid

    def plan(self, current_loc, destination_loc):
        height = self.grid.shape[0]
        planner = AStar(start_pos=current_loc, end_pos=destination_loc, map_array=self.grid, high=height)
        path = planner()
        path = list(map(lambda x: (x.x, height - x.y - 1), path))  # 转换回常规坐标系以进行绘图等操作

        # 初始化方向列表和当前朝向
        directions = []
        if len(path) > 1:
            # 初始朝向设置为从第一步到第二步的方向
            current_direction = (path[1][0] - path[0][0], path[1][1] - path[0][1])

            # 对后续的每一步计算转向
            for i in range(1, len(path)):
                if i == 1:
                    directions.append('forward')  # 第一步默认为前进
                    continue
                next_direction = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])

                if next_direction == current_direction:
                    directions.append('forward')
                elif next_direction == (-current_direction[0], -current_direction[1]):
                    directions.append('backward')
                elif next_direction == (-current_direction[1], current_direction[0]):
                    directions.append('left')
                elif next_direction == (current_direction[1], -current_direction[0]):
                    directions.append('right')
                else:
                    directions.append('none')  # 在异常情况下，添加“none”

                # 更新当前朝向
                current_direction = next_direction

            return path, directions
        else:
            return path, ['none']  # 如果路径中没有移动或只有一个位置



def main():
    grid = np.loadtxt('data/grid.txt', dtype=int)
    path_planner = PathPlanner(grid)
    
    start_pos = (20, 20)
    end_pos = (180, 220)

    # 检查起点和终点是否在障碍物上
    if check_obstacle(grid, start_pos[0], start_pos[1], grid.shape[0]):
        print("起点在障碍物上！")
    else:
        print("起点ok")

    if check_obstacle(grid, end_pos[0], end_pos[1], grid.shape[0]):
        print("终点在障碍物上！")
    else:
        print("终点ok")

    # 如果起点和终点都不在障碍物上，则进行路径规划
    if not check_obstacle(grid, start_pos[0], start_pos[1], grid.shape[0]) and not check_obstacle(grid, end_pos[0], end_pos[1], grid.shape[0]):
        
        ######

        #重点在这里readme！！！
        #方向有:"forward", "backward", "left", "right"
        #返回两个列表path, direction，path里面是每一步的坐标，direction里面是每一步的方向
        
        ######

        path, direction = path_planner.plan(start_pos, end_pos)
        if path:
            # 绘制路径到图像
            image_path = 'data/Map.jpg'  # 确保使用正确的图像文件名
            output_path = 'data/Map_plan.jpg'
            draw_path_on_image(image_path, path, output_path)
            print("Path coordinates:")
            for node in path:
                #print(1)
                print(f"({node[0]}, {node[1]})")  # 输出转换后的坐标
        else:
            print("No path found.")
        print(path)
        print(direction)

if __name__ == "__main__":
    main()
