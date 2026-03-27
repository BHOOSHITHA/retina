import torch
import numpy as np
import copy
from typing import Callable, List, Tuple
from abc import ABC, abstractmethod

class SwarmOptimizer(ABC):
    """Abstract base class for nature-inspired optimizers."""
    
    @abstractmethod
    def optimize(self):
        pass

class ParticleSwarmOptimizer(SwarmOptimizer):
    """
    Implements Particle Swarm Optimization (PSO) to tune PyTorch neural networks.
    Specifically targets hyperparameter search spaces (e.g., learning rate, weight decay).
    """
    def __init__(self, 
                 fitness_function: Callable,
                 bounds: List[Tuple[float, float]],
                 num_particles: int = 10,
                 max_iterations: int = 5,
                 w: float = 0.5,
                 c1: float = 1.5,
                 c2: float = 1.5):
        """
        :param fitness_function: Function that takes a vector of hyperparameters,
                                 trains a model, and returns a fitness score.
        :param bounds: List of tuples representing the MIN and MAX bounds for each hyperparameter.
        :param num_particles: Total particles in the swarm.
        :param max_iterations: Total iterations/generations for the swarm.
        """
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.dim = len(bounds)

        # Initialize particles within bounds
        self.positions = np.zeros((self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        
        for i in range(self.dim):
            low, high = bounds[i]
            self.positions[:, i] = np.random.uniform(low, high, self.num_particles)
            self.velocities[:, i] = np.random.uniform(-1, 1, self.num_particles) * (high - low) * 0.1

        # Track personal bests and global best
        self.personal_bests = np.copy(self.positions)
        self.personal_best_scores = np.ones(self.num_particles) * float('-inf')  # Assuming we MAXIMIZE fitness
        
        self.global_best = None
        self.global_best_score = float('-inf')

    def optimize(self):
        print(f"Starting PSO Optimization over {self.max_iterations} iterations...")
        
        for iteration in range(self.max_iterations):
            print(f"--- Iteration {iteration + 1} ---")
            
            for i in range(self.num_particles):
                # Evaluate the current particle configuration
                hyperparams = self.positions[i]
                print(f"Evaluating Particle {i+1} args: {hyperparams}")
                
                score = self.fitness_function(hyperparams)
                
                # Update Personal Best
                if score > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_bests[i] = np.copy(hyperparams)
                    
                # Update Global Best
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best = np.copy(hyperparams)

            # Update Velocities and Positions
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                
                # PSO Velocity update formula
                cognitive_velocity = self.c1 * r1 * (self.personal_bests[i] - self.positions[i])
                social_velocity = self.c2 * r2 * (self.global_best - self.positions[i])
                
                self.velocities[i] = (self.w * self.velocities[i]) + cognitive_velocity + social_velocity
                self.positions[i] = self.positions[i] + self.velocities[i]
                
                # Enforce bounds
                for dim in range(self.dim):
                    low, high = self.bounds[dim]
                    self.positions[i, dim] = np.clip(self.positions[i, dim], low, high)

            print(f"Best Score so far: {self.global_best_score} with params {self.global_best}")

        return self.global_best, self.global_best_score
