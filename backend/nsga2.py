import numpy as np
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
except ImportError:
    print("Please install pymoo: pip install pymoo")

class MultiObjectiveHyperparameterProblem(Problem):
    def __init__(self, fitness_function, bounds):
        """
        fitness_function: must return (objective1, objective2).
        For PyMoo, we minimize objectives. So if you want to MAXIMIZE dice, return -dice.
        """
        self.fitness_function = fitness_function
        n_var = len(bounds)
        xl = np.array([b[0] for b in bounds])
        xu = np.array([b[1] for b in bounds])
        
        super().__init__(n_var=n_var, 
                         n_obj=2,   # Opt 1: Loss/Dice, Opt 2: Inference Time/Params
                         n_ieq_constr=0, 
                         xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for x in X:
            # Evaluate fitness function
            f1, f2 = self.fitness_function(x)
            F.append([f1, f2])
            
        out["F"] = np.array(F)

class NSGA2Optimizer:
    def __init__(self, fitness_function, bounds, pop_size=10, max_generations=5):
        self.problem = MultiObjectiveHyperparameterProblem(fitness_function, bounds)
        self.algorithm = NSGA2(
            pop_size=pop_size,
            n_offsprings=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        self.max_generations = max_generations

    def optimize(self):
        print(f"Starting NSGA-II Multi-Objective Optimization over {self.max_generations} generations...")
        
        res = minimize(
            self.problem,
            self.algorithm,
            ('n_gen', self.max_generations),
            seed=42,
            verbose=True
        )
        
        # Returns the entire Pareto Front of solutions
        best_params_front = res.X
        best_scores_front = res.F
        
        return best_params_front, best_scores_front
