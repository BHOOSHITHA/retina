import numpy as np
try:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
except ImportError:
    pass

class HybridMultiObjectiveMHO:
    """
    The Ultimate Hybrid: GA + PSO + NSGA-II.
    1. Evaluates multi-objective fitness (Dice Acc and Inference Speed).
    2. Ranks the swarm using NSGA-II Non-Dominated Pareto Sorting.
    3. Searches space using PSO Velocity Convergence towards the Pareto Front.
    4. Escapes local optima using GA Crossover & Mutation on the weakest particles.
    """
    def __init__(self, fitness_function, bounds, pop_size=10, max_iterations=5):
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.dim = len(bounds)
        
        # PSO parameters
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        
        # GA parameters
        self.mutation_rate = 0.2
        
        self.positions = np.zeros((self.pop_size, self.dim))
        self.velocities = np.zeros((self.pop_size, self.dim))
        
        for i in range(self.dim):
            low, high = bounds[i]
            self.positions[:, i] = np.random.uniform(low, high, self.pop_size)
            self.velocities[:, i] = np.random.uniform(-1, 1, self.pop_size) * (high - low) * 0.1

        self.personal_bests = np.copy(self.positions)
        self.global_best = None

    def optimize(self, log_callback=None):
        def log(msg):
            print(msg)
            if log_callback: log_callback(msg)

        log(f"Executing Advanced Multi-Objective Hybrid (GA+PSO+NSGA2) over {self.max_iterations} Iterations")
        
        best_overall_position = None
        best_dice_score = float('-inf')
        
        nds = NonDominatedSorting()

        for iteration in range(self.max_iterations):
            F = [] 
            log(f"[Generation {iteration+1}] Evaluating swarm particles...")
            
            # 1. EVALUATION (Multi-Objective)
            for i in range(self.pop_size):
                obj1, obj2 = self.fitness_function(self.positions[i])
                # PyMoo NDS minimizes by default. Our objectives are currently NEGATIVE losses.
                # So we simply multiply by -1 to frame it as a minimization problem for PyMoo.
                F.append([-obj1, -obj2])
                
                # Keep track of absolute best single metric (Dice) just for UI reporting
                if obj1 > best_dice_score:
                    best_dice_score = obj1
                    best_overall_position = np.copy(self.positions[i])

            F = np.array(F)
            
            # 2. NSGA-II NON-DOMINATED SORTING
            fronts = nds.do(F)
            pareto_front_indices = fronts[0]
            
            # PSO Global Best is chosen randomly from the NSGA-II Pareto Front
            gbest_idx = np.random.choice(pareto_front_indices)
            self.global_best = np.copy(self.positions[gbest_idx])
            
            # 3. PSO VELOCITY & POSITION UPDATES
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cog = self.c1 * r1 * (self.personal_bests[i] - self.positions[i])
                soc = self.c2 * r2 * (self.global_best - self.positions[i])
                self.velocities[i] = (self.w * self.velocities[i]) + cog + soc
                self.positions[i] += self.velocities[i]
                
            # 4. GA CROSSOVER & MUTATION using NSGA-II Rank Density
            flattened_fronts = [idx for front in fronts for idx in front]
            best_half = flattened_fronts[:self.pop_size//2]
            worst_half = flattened_fronts[self.pop_size//2:]
            
            for i in worst_half:
                # Need to allow replacement so small swarm sizes (like pop=3) don't crash when needing 2 parents!
                p1, p2 = np.random.choice(best_half, 2, replace=True)
                alpha = np.random.random()
                
                child_pos = alpha * self.positions[p1] + (1 - alpha) * self.positions[p2]
                child_vel = alpha * self.velocities[p1] + (1 - alpha) * self.velocities[p2]
                
                for d in range(self.dim):
                    if np.random.random() < self.mutation_rate:
                        low, high = self.bounds[d]
                        child_pos[d] += np.random.normal(0, (high-low)*0.1)
                
                self.positions[i] = child_pos
                self.velocities[i] = child_vel
                
            # Clamping boundaries
            for i in range(self.pop_size):
                for d in range(self.dim):
                    low, high = self.bounds[d]
                    self.positions[i, d] = np.clip(self.positions[i, d], low, high)
            
            log(f"[Generation {iteration+1}] Complete - Best Dice Score: {best_dice_score:.3f}")

        return best_overall_position, best_dice_score
