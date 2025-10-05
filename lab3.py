import random
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass(order=True)
class PrioritizedEntry:
    priority: int
    item: Any = field(compare=False)

# Function to create a random k-SAT problem instance
def create_k_sat_instance(num_vars: int, clause_size: int, num_clauses: int) -> str:
    variable_list = [chr(i + 65) for i in range(num_vars)]
    sat_problem = "(("
    clause = []

    for i in range(clause_size * num_clauses):
        selected_var = random.choice(variable_list)
        variable_list.remove(selected_var)
        clause.append(selected_var)

        if i % clause_size == clause_size - 1:
            while clause:
                variable_list.append(clause.pop(0))

        if random.random() < 0.5:
            sat_problem += "~"

        sat_problem += selected_var

        if i % clause_size == clause_size - 1 and i != (clause_size * num_clauses - 1):
            sat_problem += ") and ("
        elif i != (clause_size * num_clauses - 1):
            sat_problem += " or "

    sat_problem += "))"
    return sat_problem

# Function to generate a random variable assignment
def random_assignment(num_variables: int) -> List[int]:
    return [random.randint(0, 1) for _ in range(num_variables)]

# Function to assess the quality of a variable assignment
def assess_assignment(assignment: Dict[str, int], clause_size: int, variables: List[str], signs: List[str]) -> int:
    fitness_score = 0
    clause_evaluation = 0

    for idx in range(len(variables)):
        if signs[idx] == 'P':
            clause_evaluation = clause_evaluation or assignment[variables[idx]]
        else:
            clause_evaluation = clause_evaluation or (1 - assignment[variables[idx]])

        if idx % clause_size == clause_size - 1 and clause_evaluation == 1:
            fitness_score += 1
            clause_evaluation = 0
            
    return fitness_score

# Hill Climbing algorithm for solving k-SAT
def hill_climb(assignment: Dict[str, int], max_depth: int, clause_size: int, variables: List[str], signs: List[str]) -> Dict[str, int]:
    depth = 0  
    while depth < max_depth:
        current_fitness = assess_assignment(assignment, clause_size, variables, signs)

        if current_fitness == len(variables):
            return assignment

        best_change = None

        for var in assignment.keys():
            neighbor_assignment = assignment.copy()
            neighbor_assignment[var] = 1 - neighbor_assignment[var]  # Flip the variable
            
            neighbor_fitness = assess_assignment(neighbor_assignment, clause_size, variables, signs)
            if neighbor_fitness > current_fitness:
                current_fitness = neighbor_fitness
                best_change = var

        depth += 1
        if best_change:
            assignment[best_change] = 1 - assignment[best_change]

    return assignment

# Beam Search algorithm for solving k-SAT
def beam_search(assignment: Dict[str, int], clause_size: int, variables: List[str], signs: List[str], beam_width: int, max_steps: int) -> Dict[str, int]:
    beam = PriorityQueue()
    current_state = assignment
    initial_fitness = assess_assignment(current_state, clause_size, variables, signs)
    beam.put(PrioritizedEntry(-initial_fitness, assignment))
    step_count = 0

    while not beam.empty() and step_count < max_steps:
        state = beam.get()
        current_state = state.item
        current_fitness = -state.priority

        if current_fitness == len(variables):
            return current_state

        for var in current_state.keys():
            neighbor_assignment = current_state.copy()
            neighbor_assignment[var] = 1 - neighbor_assignment[var]

            neighbor_fitness = assess_assignment(neighbor_assignment, clause_size, variables, signs)

            if beam.qsize() < beam_width:
                beam.put(PrioritizedEntry(-neighbor_fitness, neighbor_assignment))
            else:
                lowest = beam.get()
                if neighbor_fitness > -lowest.priority:
                    beam.put(PrioritizedEntry(-neighbor_fitness, neighbor_assignment))
                else:
                    beam.put(lowest)
                    
            step_count += 1

    return current_state

# Neighbor functions to generate new states
def flip_random_variable(assignment: Dict[str, int]) -> Dict[str, int]:
    random_var = random.choice(list(assignment))
    assignment[random_var] = 1 - assignment[random_var]
    return assignment

def swap_two_variables(assignment: Dict[str, int]) -> Dict[str, int]:
    var_a = random.choice(list(assignment))
    var_b = random.choice(list(assignment))

    while var_b == var_a:
        var_b = random.choice(list(assignment))

    assignment[var_a], assignment[var_b] = assignment[var_b], assignment[var_a]
    return assignment

def flip_first_variable(assignment: Dict[str, int]) -> Dict[str, int]:
    first_var = list(assignment.keys())[0]
    assignment[first_var] = 1 - assignment[first_var]
    return assignment

def variable_neighborhood_search(assignment: Dict[str, int], clause_size: int, variables: List[str], signs: List[str], max_steps: int) -> Dict[str, int]:
    step_count = 0
    current_state = assignment
    
    while step_count < max_steps:
        current_state = assignment
        current_fitness = assess_assignment(assignment, clause_size, variables, signs)

        if current_fitness == len(variables):
            return current_state

        neighbor1 = flip_random_variable(current_state.copy())
        neighbor2 = swap_two_variables(current_state.copy())
        neighbor3 = flip_first_variable(current_state.copy())

        fitness1 = assess_assignment(neighbor1, clause_size, variables, signs)
        fitness2 = assess_assignment(neighbor2, clause_size, variables, signs)
        fitness3 = assess_assignment(neighbor3, clause_size, variables, signs)

        best_fitness = max(fitness1, fitness2, fitness3)
        if best_fitness > current_fitness:
            current_fitness = best_fitness
            if current_fitness == fitness1:
                current_state = neighbor1
            elif current_fitness == fitness2:
                current_state = neighbor2
            else:
                current_state = neighbor3
        
        step_count += 1
    
    return current_state

# Main execution flow to set up and solve the problem
num_vars = 25
clause_size = 3
num_clauses = 1000
k_sat_problem = create_k_sat_instance(num_vars, clause_size, num_clauses)
variable_set = set()
variables = []
signs = []

previous_negative = False

for char in k_sat_problem:
    if char == '~':
        previous_negative = True
    elif 'A' <= char <= 'Z':
        signs.append('N' if previous_negative else 'P')
        variables.append(char)
        variable_set.add(char)
        previous_negative = False

initial_values = random_assignment(len(variable_set))
initial_state = {var: initial_values[i] for i, var in enumerate(variable_set)}

print(initial_state)
print("Initial State Fitness: ", assess_assignment(initial_state, clause_size, variables, signs))

hill_climb_solution = hill_climb(initial_state.copy(), 100, clause_size, variables, signs)
print("Hill Climbing Solution Fitness: ", assess_assignment(hill_climb_solution, clause_size, variables, signs))

beam_search_solution_3 = beam_search(initial_state.copy(), clause_size, variables, signs, 3, 1000)
print("Beam Search Solution Fitness (Beam-Width = 3): ", assess_assignment(beam_search_solution_3, clause_size, variables, signs))

beam_search_solution_4 = beam_search(initial_state.copy(), clause_size, variables, signs, 4, 1000)
print("Beam Search Solution Fitness (Beam-Width = 4): ", assess_assignment(beam_search_solution_4, clause_size, variables, signs))

print("Random Variable Flip Neighbor: ", flip_random_variable(initial_state.copy()))
print("Variable Swap Neighbor: ", swap_two_variables(initial_state.copy()))
print("First Variable Flip Neighbor: ", flip_first_variable(initial_state.copy()))

variable_neighborhood_solution = variable_neighborhood_search(initial_state.copy(), clause_size, variables, signs, 1000)
print("Variable Neighborhood Search Fitness: ", assess_assignment(variable_neighborhood_solution, clause_size, variables, signs))