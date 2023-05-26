import heapq
import random

class MazeEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.maze = [[0] * width for _ in range(height)]

    def generate_maze(self, obstacle_density):
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < obstacle_density:
                    self.maze[y][x] = 1

    def is_valid_position(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and not self.maze[y][x]

    def get_neighboring_cells(self, x, y):
        neighbors = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid_position(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def calculate_heuristic(self, x, y, goal_x, goal_y):
        return abs(goal_x - x) + abs(goal_y - y)

    def find_path(self, start_x, start_y, goal_x, goal_y):
        open_set = []
        heapq.heappush(open_set, (0, start_x, start_y))  # (f_score, x, y)
        came_from = {}
        g_scores = {cell: float('inf') for row in self.maze for cell in row}
        g_scores[(start_x, start_y)] = 0
        f_scores = {cell: float('inf') for row in self.maze for cell in row}
        f_scores[(start_x, start_y)] = self.calculate_heuristic(start_x, start_y, goal_x, goal_y)

        while open_set:
            _, current_x, current_y = heapq.heappop(open_set)

            if current_x == goal_x and current_y == goal_y:
                path = []
                while (current_x, current_y) in came_from:
                    path.append((current_x, current_y))
                    current_x, current_y = came_from[(current_x, current_y)]
                path.append((start_x, start_y))
                path.reverse()
                return path

            neighbors = self.get_neighboring_cells(current_x, current_y)
            for neighbor_x, neighbor_y in neighbors:
                tentative_g_score = g_scores[(current_x, current_y)] + 1
                if tentative_g_score < g_scores[(neighbor_x, neighbor_y)]:
                    came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)
                    g_scores[(neighbor_x, neighbor_y)] = tentative_g_score
                    f_scores[(neighbor_x, neighbor_y)] = tentative_g_score + self.calculate_heuristic(
                        neighbor_x, neighbor_y, goal_x, goal_y)
                    heapq.heappush(open_set, (f_scores[(neighbor_x, neighbor_y)], neighbor_x, neighbor_y))

        return None

# Usage example
env = MazeEnvironment(10, 10)
env.generate_maze(0.3)  # Obstacle density of 0.3

start_x, start_y = 0, 0
goal_x, goal_y = 9, 9

path = env.find_path(start_x, start_y, goal_x, goal_y)
if path:
    print("Path found:", path)
else:
    print("No path found.")
