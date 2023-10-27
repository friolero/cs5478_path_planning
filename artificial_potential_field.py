import numpy as np
from map import ImageMap2D
from utils import distance, vis_path
from primitives import Node

# 你好，这里是珊禾牌ChatGPT，很高兴为您服务（ε=(´ο｀*)))唉）
class ArtificialPotentialField:
    def __init__(self, map, start, goal, k_att=0.5, k_rep=100.0, max_iterations=1000, step_size=1.0):
        # 初始化
        self.map = map                        # 加载地图
        self.start = start                    # 起点
        self.goal = goal                      # 终点
        self.k_att = k_att                    # 引力常数
        self.k_rep = k_rep                    # 斥力常数
        self.max_iterations = max_iterations  # 最大迭代次数
        self.step_size = step_size            # 步长

    def attractive_potential(self, current_node):
        return 0.5 * self.k_att * distance(current_node, self.goal) ** 2

    def repulsive_potential(self, current_node):
        repulsive_potential = 0.0
        for obstacle_node in self.map.free_conf:
            dist = distance(current_node, obstacle_node)
            if dist > 0:
                repulsive_potential += 0.5 * self.k_rep / dist
        return repulsive_potential

    def plan(self):
        # 开整
        current_node = self.start
        path = [current_node]

        for _ in range(self.max_iterations):
            if distance(current_node, self.goal) < self.step_size:
                return path

            attractive_force = (self.goal - current_node) * self.k_att
            repulsive_force = Node(0, 0, None)

            for obstacle_node in self.map.free_conf:
                dist = distance(current_node, obstacle_node)
                if dist < self.step_size and dist > 0:
                    repulsive_force += (current_node - obstacle_node) * (self.k_rep / dist ** 3)

            total_force = attractive_force + repulsive_force
            next_node = current_node + total_force
            path.append(next_node)
            current_node = next_node

        return None

if __name__ == "__main__":
    map = ImageMap2D("data/2d_maze_2.png")
    start = Node(10, 20, None)
    goal = Node(200, 100, None)

    planner = ArtificialPotentialField(map, start, goal)
    path = planner.plan()

    if path is None:
        print("ε=(´ο｀*)))")
    else:
        print("(ಡωಡ)hiahiahia")
        # 可视化路径
        # 头好痒，可能要长脑子了
        l = len(path)
        i = 0
        re_path = []
        while i < l:
            a = int(path[i].x)
            b = int(path[i].y)
            re_path.append(Node(a, b, None))
            print(re_path[i].x, re_path[i].y)
            i = i + 1
        
        vis_path(map, re_path)
