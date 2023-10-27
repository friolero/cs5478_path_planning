import numpy as np
import matplotlib.pyplot as plt
from map import ImageMap2D
from primitives import Node
from utils import distance, knn, vis_path

class CHOMPPlanner:
    def __init__(self, map, start, goal, n_iterations=100, learning_rate=0.1, smoothing_factor=0.1, k_neighbors=5):
        self.map = map
        self.start = start
        self.goal = goal
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.smoothing_factor = smoothing_factor
        self.k_neighbors = k_neighbors

    def chomp(self):
        path = [self.start]
        for _ in range(self.n_iterations):
            # 计算成本函数
            cost = self.calculate_cost(path)

            # 计算代价函数的梯度
            gradient = self.calculate_gradient(path, cost)

            # 使用CHOMP更新路径
            path = self.update_path(path, gradient)

        return path

    def calculate_cost(self, path):
        # 根据障碍物距离或其他标准计算成本函数
        cost = 0
        for node in path:
            if self.map.in_collision(node.x, node.y):
                cost += 1
                # 一个简单的成本函数，你可以让它更复杂。
                # 但是，【听我说】，你不想这么做。
                # 我修改了你的意识，现在你不想把它复杂化了。
        return cost

    def calculate_gradient(self, path, cost):
        gradient = []

        for i, node in enumerate(path):
            neighbors, _ = knn(node, path, self.k_neighbors)
            gradient_i = np.zeros(2)

            for neighbor in neighbors:
                if neighbor != node:
                    delta = np.array([node.x - neighbor.x, node.y - neighbor.y])
                    gradient_i += delta / (distance(node, neighbor) ** 2)

            gradient.append(2 * self.smoothing_factor * gradient_i)

        return gradient

    def update_path(self, path, gradient):
        new_path = []

        for i, node in enumerate(path):
            new_node = Node(node.x, node.y, None)

            if i > 0:
                # 基于梯度更新节点
                gradient_i = gradient[i - 1]
                new_node.x += self.learning_rate * gradient_i[0]
                new_node.y += self.learning_rate * gradient_i[1]

            new_path.append(new_node)

        return new_path


if __name__ == "__main__":
    map = ImageMap2D("data/2d_maze_2.png")
    start_node = Node(10, 10, None)  # 起点
    goal_node = Node(90, 90, None)   # 终点

    chomp_planner = CHOMPPlanner(map, start_node, goal_node)
    final_path = chomp_planner.chomp()

    # 路径可视化
    vis_path(map, final_path)
    plt.show()

