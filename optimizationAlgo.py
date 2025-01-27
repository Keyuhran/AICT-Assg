import random
import math

def calculateTotalTravelTime(routes, traffic):
    """Calculate the total travel time for all routes given traffic conditions."""
    totalTime = 0  # Initialize total travel time
    for route in routes:  # Loop through each route
        if len(route) < 2:  # Skip empty or single-node routes
            continue
        for i in range(len(route) - 1):  # Loop through consecutive stops in the route
            start, end = route[i], route[i + 1]  # Get the current and next location
            if start in traffic and end in traffic[start]:  # Check if edge exists in traffic
                totalTime += traffic[start][end]  # Add the travel time to total
            else:
                print(f"Missing traffic data for edge {start} -> {end}.")
                return None  # Return None or handle the error
    return totalTime

def simulatedAnnealing(routes, traffic, initialTemp, coolingRate, maxIter):
    """Optimize vehicle routes using Simulated Annealing."""
    currentRoutes = routes
    currentCost = calculateTotalTravelTime(routes, traffic)
    bestRoutes = currentRoutes[:]
    bestCost = currentCost

    temperature = initialTemp

    for _ in range(maxIter):
        # Generate a neighboring solution
        newRoutes = generateNeighboringSolution(currentRoutes)
        newCost = calculateTotalTravelTime(newRoutes, traffic)

        # Decide whether to accept the new solution
        if newCost < currentCost or random.random() < math.exp((currentCost - newCost) / temperature):
            currentRoutes = newRoutes
            currentCost = newCost
            if currentCost < bestCost:
                bestRoutes = currentRoutes[:]
                bestCost = currentCost

        # Cool down the temperature
        temperature *= coolingRate

        # Stop if temperature is low enough
        if temperature < 1e-3:
            break

    return bestRoutes, bestCost

def generateNeighboringSolution(routes):
    """Generates a neighboring solution by swapping two locations in a randomly chosen route."""
    # Make a deep copy of the current routes to avoid modifying the original
    newRoutes = [route.copy() for route in routes]
    
    # Select a random route from the list of routes
    randomRouteIndex = random.randint(0, len(newRoutes) - 1)
    randomRoute = newRoutes[randomRouteIndex]
    
    # If the route has less than 2 locations, return the original solution (no swaps possible)
    if len(randomRoute) < 2:
        return newRoutes
    
    # Select two random, distinct locations in the route to swap
    i, j = random.sample(range(len(randomRoute)), 2)
    randomRoute[i], randomRoute[j] = randomRoute[j], randomRoute[i]
    
    return newRoutes

traffic = {
    0: {1: 10, 2: 15, 3: 20},
    1: {0: 10, 2: 35, 3: 25},
    2: {0: 15, 1: 35, 3: 30},
    3: {0: 20, 1: 25, 2: 30}
}

routes = [[0, 1, 3], [2, 3, 1]]
optimizedRoutes, optimizedCost = simulatedAnnealing(routes, traffic, 100, 0.95, 1000)

print("Optimized Routes:", optimizedRoutes)
print("Optimized Total Travel Time:", optimizedCost)
