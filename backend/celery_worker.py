import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from celery import Celery

redis_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_app = Celery("mho_tasks", broker=redis_url, backend=redis_url)

@celery_app.task(name="run_optimization", bind=True)
def run_optimization(self):
    logs = []
    def log_callback(msg):
        logs.append(msg)
        self.update_state(state='PROGRESS', meta={'logs': logs})
        
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    from train import evaluate_fitness
    from hybrid_mho import HybridMultiObjectiveMHO
    
    bounds = [(1e-5, 1e-2), (1e-6, 1e-3), (2, 8), (0, 1)]
    fitness = lambda hp: evaluate_fitness(hp, dataset_name="ALL_HYBRID")
    
    opt = HybridMultiObjectiveMHO(fitness, bounds, pop_size=3, max_iterations=2)
    best_params, best_score = opt.optimize(log_callback=log_callback)
    
    return {
        "status": "completed",
        "result": {
            "learning_rate": float(best_params[0]),
            "weight_decay": float(best_params[1]),
            "batch_size": int(best_params[2]),
            "architecture": "ResUNet" if int(np.round(best_params[3])) == 1 else "UNet",
            "fitness_score": float(best_score)
        }
    }
