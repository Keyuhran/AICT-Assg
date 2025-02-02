import heapq
from collections import deque
import time

# Graph representation
class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, node1, node2, weight):
        if node1 not in self.graph:
            self.graph[node1] = []
        if node2 not in self.graph:
            self.graph[node2] = []
        self.graph[node1].append((node2, weight))
        self.graph[node2].append((node1, weight))  # Assuming undirected graph

# BFS implementation
def bfs(graph, start, goal):
    visited = set()
    queue = deque([(start, [start])])

    while queue:
        current, path = queue.popleft()

        if current == goal:
            return path

        if current not in visited:
            visited.add(current)
            # Sort neighbors to maintain expected order
            for neighbor, _ in sorted(graph.graph[current]):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    return None


# DFS implementation
def dfs(graph, start, goal):
    visited = set()
    stack = [(start, [start])]

    while stack:
        current, path = stack.pop()

        if current == goal:
            return path

        if current not in visited:
            visited.add(current)
            for neighbor, _ in graph.graph[current]:
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

    return None

# GBFS implementation
def gbfs(graph, start, goal, heuristic):
    visited = set()
    priority_queue = [(heuristic[start], start, [start])]

    while priority_queue:
        _, current, path = heapq.heappop(priority_queue)

        if current == goal:
            return path

        if current not in visited:
            visited.add(current)
            for neighbor, _ in graph.graph[current]:
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (heuristic[neighbor], neighbor, path + [neighbor]))

    return None

# A* Search implementation
def a_star(graph, start, goal, heuristic):
    visited = set()
    priority_queue = [(0, start, [start], 0)]  # (priority, node, path, cost)

    while priority_queue:
        _, current, path, cost = heapq.heappop(priority_queue)

        if current == goal:
            return path

        if current not in visited:
            visited.add(current)
            for neighbor, weight in graph.graph[current]:
                if neighbor not in visited:
                    new_cost = cost + weight
                    priority = new_cost + heuristic[neighbor]
                    heapq.heappush(priority_queue, (priority, neighbor, path + [neighbor], new_cost))

    return None

# Evaluate performance of algorithms
def evaluate_algorithms(graph, start, goal, heuristic):
    algorithms = {
        "BFS": bfs,
        "DFS": dfs,
        "GBFS": gbfs,
        "A*": a_star
    }

    results = {}
    for name, algorithm in algorithms.items():
        start_time = time.time()
        if name in ["GBFS", "A*"]:
            path = algorithm(graph, start, goal, heuristic)
        else:
            path = algorithm(graph, start, goal)
        runtime = time.time() - start_time
        results[name] = (path, runtime)

    return results


def main():
    city_graph = Graph()
    city_graph.add_edge("A", "B", 1)
    city_graph.add_edge("A", "C", 4)
    city_graph.add_edge("B", "C", 2)
    city_graph.add_edge("B", "D", 5)
    city_graph.add_edge("C", "D", 1)
    city_graph.add_edge("D", "E", 3)

    heuristic = {
        "A": 7,
        "B": 6,
        "C": 2,
        "D": 1,
        "E": 0
    }

    start, goal = "A", "E"
    results = evaluate_algorithms(city_graph, start, goal, heuristic)

    for algo, (path, runtime) in results.items():
        print(f"{algo}: Path: {path}, Runtime: {runtime:.6f} seconds")

if __name__ == "__main__":
    main()