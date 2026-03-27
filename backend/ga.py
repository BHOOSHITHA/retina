import numpy as np
from abc import ABC, abstractmethod

class SwarmOptimizer(ABC):
    @abstractmethod
    def optimize(self):
        pass

class GeneticAlgorithm(SwarmOptimizer):
    """
    Implements a standard continuous Genetic Algorithm to tune hyperparameters.
    """
    def __init__(self, fitness_function, bounds, pop_size=10, max_generations=5, mutation_rate=0.15):
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.dim = len(bounds)
        self.population = np.zeros((self.pop_size, self.dim))
        
        for i in range(self.dim):
            low, high = bounds[i]
            self.population[:, i] = np.random.uniform(low, high, self.pop_size)

    def optimize(self):
        print(f"Starting Genetic Algorithm Optimization over {self.max_generations} generations...")
        best_overall = None
        best_score_overall = float('-inf')

        for generation in range(self.max_generations):
            print(f"--- Generation {generation + 1}/{self.max_generations} ---")
            scores = []
            
            for i in range(self.pop_size):
                score = self.fitness_function(self.population[i])
                scores.append(score)
                if score > best_score_overall:
                    best_score_overall = score
                    best_overall = np.copy(self.population[i])

            scores = np.array(scores)
            
            # Tournament Selection & Creation of new population
            new_population = np.zeros_like(self.population)
            for i in range(self.pop_size):
                p1, p2 = np.random.choice(self.pop_size, 2, replace=False)
                winner_idx = p1 if scores[p1] > scores[p2] else p2
                parent1 = self.population[winner_idx]
                
                # Arithmetic Crossover
                parent2_idx = np.random.choice(self.pop_size)
                alpha = np.random.random()
                child = alpha * parent1 + (1 - alpha) * self.population[parent2_idx]
                
                # Gaussian Mutation
                for d in range(self.dim):
                    if np.random.random() < self.mutation_rate:
                        low, high = self.bounds[d]
                        scale = (high - low) * 0.1
                        child[d] += np.random.normal(0, scale)
                        # Enforce bounds
                        child[d] = np.clip(child[d], low, high)
                
                new_population[i] = child
                
            self.population = new_population

        return best_overall, best_score_overall
