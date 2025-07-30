import pygad
import numpy as np

# Define the fitness function
def fitness_function(ga_instance,solution, solution_idx):
    # Objective: Maximize the sum of squares of the solution elements
    return np.sum(np.square(solution))

# Problem parameters
num_generations = 50  # Number of generations
num_parents_mating = 4  # Number of parents for mating
sol_per_pop = 10  # Population size
num_genes = 5  # Number of genes (variables in the solution)

# Define the range of values for each gene 
gene_space = {'low': -10, 'high': 10}

# Initialize the GA
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    gene_space=gene_space
)

# Run the GA
ga_instance.run()

# Get the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution:", solution)
print("Fitness value of the best solution:", solution_fitness)
print("idx is - ", solution_idx)
           
           
# Plot the fitness progress
ga_instance.plot_fitness()
