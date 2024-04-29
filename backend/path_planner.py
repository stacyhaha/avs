import numpy as np
import heapq
from functools import lru_cache
from PIL import Image, ImageDraw
from utils import *

class Node:
    def __init__(self, x, y, cost=0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent
    
    def __eq__(self, other):
        # 比较节点是否相等，需要根据坐标来判断
        return (self.x == other.x) and (self.y == other.y)
    
    def __hash__(self):
        # 使得Node对象可哈希，以便在set中使用
        return hash((self.x, self.y))
    
    def __lt__(self, other):
        # 比较两个节点的成本，用于优先队列中的排序
        return self.cost < other.cost
    
    def __add__(self, other):
        # 实现节点的加法，便于计算移动后的新节点
        return Node(self.x + other[0], self.y + other[1], self.cost, self)
    
    def __sub__(self, other):
        # 用于计算两个节点之间的启发式距离
        return abs(self.x - other.x) + abs(self.y - other.y)

class NodeQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, node):
        heapq.heappush(self.elements, node)
    
    def get(self):
        return heapq.heappop(self.elements)


class AStar:
    """A*算法"""
    def __init__(self, start_pos, end_pos, map_array, move_step=3, move_direction=8):
        self.map_array = map_array  # H * W
        self.width = self.map_array.shape[1]
        self.high = self.map_array.shape[0]
        self.start = Node(*start_pos)  # 初始位置
        self.end = Node(*end_pos)      # 结束位置

        # Check if the start and end nodes are within the map boundaries
        if not self._in_map(self.start) or not self._in_map(self.end):
            raise ValueError(f"X coordinate range should be 0 to {self.width-1}, Y coordinate range should be 0 to {self.high-1}")
        # Check if the start node is colliding with an obstacle
        if self._is_collided(self.start):
            raise ValueError("The starting node's x or y coordinate is on an obstacle")
        # Check if the end node is colliding with an obstacle
        if self._is_collided(self.end):
            raise ValueError("The ending node's x or y coordinate is on an obstacle")

        
        self.reset(move_step, move_direction)

    def reset(self, move_step=3, move_direction=8):
        self.move_step = move_step
        self.move_direction = move_direction
        self.close_set = set()
        self.open_queue = NodeQueue()
        self.path_list = []

    def _in_map(self, node):
        return 0 <= node.x < self.width and 0 <= node.y < self.high
    
    def _is_collided(self, node):
        return self.map_array[node.y, node.x] == 1  # 障碍物是1
    
    def _move(self):
        @lru_cache(maxsize=3)
        def _move_cached(move_step, move_direction):
            moves = [
                (0, move_step),        # 上
                (0, -move_step),       # 下
                (-move_step, 0),       # 左
                (move_step, 0),        # 右
                (move_step, move_step), # 右上
                (move_step, -move_step), # 右下
                (-move_step, move_step), # 左上
                (-move_step, -move_step) # 左下
            ]
            return moves[:move_direction]
        return _move_cached(self.move_step, self.move_direction)

    def _update_open_list(self, curr):
        for add in self._move():
            next_ = curr + add
            if not self._in_map(next_) or self._is_collided(next_) or next_ in self.close_set:
                continue
            H = next_ - self.end
            next_.cost += H
            self.open_queue.put(next_)
            if H < 20:
                self.move_step = 1

    def __call__(self):
        self.open_queue.put(self.start)
        while not self.open_queue.empty():
            curr = self.open_queue.get()
            curr.cost -= (curr - self.end)
            self._update_open_list(curr)
            self.close_set.add(curr)
            if curr == self.end:
                break
        while curr.parent is not None:
            self.path_list.append(curr)
            curr = curr.parent
        self.path_list.reverse()
        return self.path_list
    

class PathPlanner:
    def __init__(self, map_path):
        self.grid = np.loadtxt(map_path, dtype=int)

    def plan(self, current_loc, destination_loc):
        planner = AStar(start_pos=current_loc, end_pos=destination_loc, map_array=self.grid)
        path = planner()
        path = list(map(lambda x:(x.x, x.y), path))
        path = [current_loc] + path
        return path



if __name__ == "__main__":
    path_planner = PathPlanner("data/grid.txt")
    path = path_planner.plan((40, 40), (270, 232))
    print(path)


    
