import random
import math
from collections import deque

# Graph representation
class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, node1, node2, weight, capacity):
        if node1 not in self.graph:
            self.graph[node1] = []
        if node2 not in self.graph:
            self.graph[node2] = []
        self.graph[node1].append((node2, weight, capacity))
        self.graph[node2].append((node1, weight, capacity))  # Assuming undirected graph

    def get_neighbors(self, node):
        return self.graph.get(node, [])

# BFS implementation to find the shortest path
def bfs(graph, start, goal):
    visited = set()
    queue = deque([(start, [start])])

    while queue:
        current, path = queue.popleft()

        if current == goal:
            return path

        if current not in visited:
            visited.add(current)
            for neighbor, _, _ in graph.get_neighbors(current):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    return None

# Calculate total traffic cost based on paths and road capacities
def calculate_cost(graph, vehicles, paths):
    road_usage = {road: 0 for road in graph.graph.keys()}
    for vehicle in vehicles:
        for road in paths[vehicle]:
            road_usage[road] += 1

    total_cost = 0
    for road, usage in road_usage.items():
        _, _, capacity = graph.graph[road][0]  # Road capacity
        if usage > capacity:  # If usage exceeds capacity
            total_cost += (usage - capacity) ** 2  # Penalize heavily for exceeding capacity
    return total_cost

# Check if all vehicles have reached their destination
def all_vehicles_reached_destination(vehicles, paths):
    return all(paths[vehicle][-1] == vehicle.goal for vehicle in vehicles)

# Simulated Annealing for optimizing traffic flow (fixed)
def simulated_annealing(graph, vehicles, temperature=100, cooling_rate=0.995, min_temp=0.01):
    # Randomly initialize the paths for each vehicle
    paths = {vehicle: random.choice(vehicle.paths) for vehicle in vehicles}
    
    current_cost = calculate_cost(graph, vehicles, paths)
    
    while temperature > min_temp:
        if all_vehicles_reached_destination(vehicles, paths):
            print("All vehicles reached their destination.")
            break
        
        # Choose a random vehicle and randomly change its path
        vehicle = random.choice(vehicles)
        old_path = paths[vehicle]
        new_path = random.choice(vehicle.paths)
        
        if old_path == new_path:
            continue  # If no change, skip
        
        paths[vehicle] = new_path  # Apply the new path
        new_cost = calculate_cost(graph, vehicles, paths)

        # Decide whether to accept the new solution
        if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temperature):
            current_cost = new_cost  # Accept the new solution
        else:
            paths[vehicle] = old_path  # Revert the change

        # Cooling down the temperature
        temperature *= cooling_rate

    return paths, current_cost  # Return the final paths and their cost

# Hill Climbing for optimizing traffic flow
def hill_climbing(graph, vehicles, max_iterations=1000, temperature=1.0, cooling_rate=0.99, max_restarts=10):
    best_paths_overall = None
    best_cost_overall = float('inf')
    
    for restart in range(max_restarts):
        print(f"Starting restart {restart + 1}/{max_restarts}")
        
        # Randomly initialize paths for all vehicles
        paths = {vehicle: random.choice(vehicle.paths) for vehicle in vehicles}
        current_best_paths = paths.copy()
        current_best_cost = calculate_cost(graph, vehicles, paths)
        
        temperature_local = temperature  # Reset the temperature for each restart
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Generate neighbors (small modifications to one vehicle's path)
            neighbors = []
            for vehicle in vehicles:
                current_path = paths[vehicle]
                # Generate alternative paths for this vehicle
                new_paths = [path for path in vehicle.paths if path != current_path]
                for new_path in new_paths:
                    # Generate a new solution by modifying one vehicle's path
                    new_paths_dict = paths.copy()
                    new_paths_dict[vehicle] = new_path
                    new_cost = calculate_cost(graph, vehicles, new_paths_dict)
                    neighbors.append((new_paths_dict, new_cost))
            
            if not neighbors:
                break
            
            # Randomly select a neighbor to explore
            neighbor = random.choice(neighbors)
            new_paths, new_cost = neighbor
            
            # Calculate the acceptance probability for a worse solution
            if new_cost < current_best_cost or random.uniform(0, 1) < acceptance_probability(current_best_cost, new_cost, temperature_local):
                # Accept the new solution (even if worse, with some probability)
                paths = new_paths
                current_best_cost = new_cost
                current_best_paths = paths
            
            # Cool down the temperature
            temperature_local *= cooling_rate
            
            # Ensure all vehicles have valid paths to their destinations
            if all_vehicles_reached_destination(vehicles, paths):
                break  # Exit the loop once all vehicles reach their destinations
        
        # Update the best solution found across restarts
        if current_best_cost < best_cost_overall:
            best_paths_overall = current_best_paths
            best_cost_overall = current_best_cost
    
    # Return the best paths and their cost after all restarts
    return best_paths_overall, best_cost_overall

def acceptance_probability(old_cost, new_cost, temperature):
    """Calculate the probability of accepting a worse solution."""
    if new_cost < old_cost:
        return 1.0  # Always accept better solutions
    else:
        return math.exp((old_cost - new_cost) / temperature)

# Local Search for optimizing traffic flow
def local_search(graph, vehicles):
    paths = {vehicle: random.choice(vehicle.paths) for vehicle in vehicles}

    def calculate_neighbors(paths):
        neighbors = []
        for vehicle in vehicles:
            for path in vehicle.paths:
                new_paths = paths.copy()
                new_paths[vehicle] = path
                neighbors.append(new_paths)
        return neighbors

    current_cost = calculate_cost(graph, vehicles, paths)
    while True:
        neighbors = calculate_neighbors(paths)
        best_neighbor = min(neighbors, key=lambda x: calculate_cost(graph, vehicles, x))

        best_cost = calculate_cost(graph, vehicles, best_neighbor)
        if best_cost < current_cost:
            paths = best_neighbor
            current_cost = best_cost
        else:
            break

    return paths, current_cost

# Vehicle class
class Vehicle:
    def __init__(self, start, goal, paths):
        self.start = start
        self.goal = goal
        self.paths = paths

def main():
    # Define the city graph
    city_graph = Graph()
    city_graph.add_edge("A", "B", 1, 2)  # Road from A to B with capacity 2
    city_graph.add_edge("A", "C", 4, 3)  # Road from A to C with capacity 3
    city_graph.add_edge("B", "C", 2, 2)
    city_graph.add_edge("B", "D", 5, 1)
    city_graph.add_edge("C", "D", 1, 2)
    city_graph.add_edge("D", "E", 3, 1)

    # Define vehicles with their possible paths
    vehicles = [
        Vehicle("A", "E", [bfs(city_graph, "A", "E"), bfs(city_graph, "A", "D")]),
        Vehicle("B", "E", [bfs(city_graph, "B", "E"), bfs(city_graph, "B", "D")]),
        Vehicle("C", "E", [bfs(city_graph, "C", "E"), bfs(city_graph, "C", "D")]),
    ]

    # Choose optimization method (simulated_annealing, hill_climbing, local_search)
    optimization_method = simulated_annealing
    optimized_paths, final_cost = optimization_method(city_graph, vehicles)

    # Print the optimized paths and final traffic cost
    print(f"Optimization using {optimization_method.__name__}:")
    for vehicle in vehicles:
        print(f"Vehicle from {vehicle.start} to {vehicle.goal}: {optimized_paths[vehicle]}")
    print(f"Final Traffic Cost: {final_cost}")

if __name__ == "__main__":
    main()
