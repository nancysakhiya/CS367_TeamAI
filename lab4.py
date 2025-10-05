import numpy as np
import random
from math import sin, cos, sqrt, atan2, radians

# Function to calculate the Haversine distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    a = sin(dLat / 2) * sin(dLat / 2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon / 2) * sin(dLon / 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Function to calculate the objective function (total cost of the tour)
def objective_function(tour, distances):
    total_cost = 0
    for i in range(len(tour) - 1):
        total_cost += distances[tour[i]][tour[i + 1]]
    total_cost += distances[tour[-1]][tour[0]]  # Add the cost of returning to the starting city
    return total_cost

# Function to generate a random solution for the TSP
def generate_random_solution(num_cities):
    tour = list(range(num_cities))
    random.shuffle(tour)
    return tour

# Function to generate a new solution by swapping two cities in the tour
def generate_new_solution(old_solution):
    new_solution = old_solution.copy()
    i, j = random.sample(range(len(new_solution)), 2)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

# Function to implement the Simulated Annealing algorithm
def simulated_annealing(distances, num_cities, num_iterations, initial_temperature, cooling_rate):
    current_solution = generate_random_solution(num_cities)
    current_cost = objective_function(current_solution, distances)
    best_solution = current_solution
    best_cost = current_cost
    temperature = initial_temperature

    for iteration in range(num_iterations):
        new_solution = generate_new_solution(current_solution)
        new_cost = objective_function(new_solution, distances)

        if new_cost < current_cost:
            current_solution = new_solution
            current_cost = new_cost

            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost

        else:
            if random.random() < np.exp(-(new_cost - current_cost) / temperature):
                current_solution = new_solution
                current_cost = new_cost
        temperature *= cooling_rate

    return best_solution, best_cost


# Load the coordinates of the 20 important tourist locations in Rajasthan
coordinates = np.loadtxt('coordinate.txt', delimiter=',')

# Calculate the pairwise distances between these locations using the Haversine formula
distances = np.zeros((20, 20))

for i in range(20):
    for j in range(i + 1, 20):
        distances[i][j] = haversine(coordinates[i][0], coordinates[i][1], coordinates[j][0], coordinates[j][1])

# Apply Simulated Annealing
initial_temperature = 1000
num_iterations = 10000
cooling_rate = 0.95
best_solution, best_cost = simulated_annealing(distances, 20, num_iterations, initial_temperature, cooling_rate)

# Print the optimal tour
print("Optimal Tour:")

for city in best_solution:
    print(city)

# Print the total cost of the optimal tour
print("Total Cost: ", best_cost)