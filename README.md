# System Automated Retinal AI - Metaheuristic Optimization (MHO)

This project is an advanced Medical Image Analysis platform focused on **Retinal Vessel Segmentation**. Using deep learning architectures optimized by state-of-the-art **Metaheuristic Algorithms**, the system autonomously tunes clinical network hyperparameters and searches for the best network design without manual intervention.

## 🧠 Principle of Operation

Deep learning models (like U-Net) achieve state-of-the-art results for retinal vessel segmentation (used to detect diseases like Diabetic Retinopathy or Glaucoma). However, these networks are extremely sensitive to their hyperparameter configurations (e.g., learning rate, weight decay). 

Instead of manual "guess-and-check" tuning or computationally expensive Grid Searches, this system utilizes **Bio-Inspired Metaheuristic Optimization (MHO)**. These algorithms simulate natural phenomena—like biological evolution or flocking birds—to smartly navigate the vast hyperparameter search space. 
By framing tuning as a **Multi-Objective Problem**, the system automatically finds the "Pareto Front" of configurations that maximize the vessel segmentation accuracy (Dice score) while minimizing the model's inference time.

---

## 🧬 Metaheuristic Algorithms Used

The backend features multiple discrete MHO solvers, culminating in a custom Hybrid optimizer:

1. **Genetic Algorithm (GA)**: Mimics biological evolution. Maintains a "population" of hyperparameter sets, evaluating fitness, selecting the strongest candidates via tournaments, and blending them through *Arithmetic Crossover* and *Gaussian Mutation* to produce better generations.
2. **Particle Swarm Optimization (PSO)**: Simulates the social behavior of bird flocks. Each potential hyperparameter set is a "particle" that "flies" through the search space, updating its velocity based on its own best historical performance (cognitive) and the swarm's overall best performance (social).
3. **NSGA-II (Non-dominated Sorting Genetic Algorithm II)**: A dedicated multi-objective optimizer that ranks solutions into "Pareto Fronts." It prevents a single objective (like just accuracy) from dominating, ensuring a diverse set of solutions that balance accuracy and computational speed.
4. **Hybrid MHO (The Ultimate Optimizer)**: A custom algorithm integrating the best of the above:
   - Evaluates multi-objective fitness features.
   - Ranks the swarm using NSGA-II Non-Dominated Pareto Sorting.
   - Searches the continuous space using PSO velocity convergence towards the Pareto front.
   - Escapes local minima by applying GA Crossover & Mutation on the weakest particles in the swarm.

---

## 🏗️ Model Architecture & Training Process

### 1. Neural Architectures
The system implements Neural Architecture Search (NAS) capabilities to choose between two robust PyTorch networks:
- **U-Net**: The classical Encoder-Decoder architecture with skip connections.
- **Improved ResUNet**: Enhances the standard U-Net by replacing standard convolutional blocks with **Residual Blocks**. This stabilizes deeper network gradients and improves the detection of complex bifurcations and thin vessel structures.

### 2. The Training Loop (`train.py`)
- **Dataset Merging**: The script dynamically searches for clinical datasets (like DRIVE, CHASE_DB1, FIVES) inside the `backend/datasets/` folder and merges them using PyTorch's `ConcatDataset` to construct a massive, generalized training pipeline.
- **Fast Fitness Evaluation**: To make MHO feasible, the system samples a random mini-batch (subset) of the massive dataset to quickly evaluate a candidate's "fitness score" instead of running full lengthy epochs.
- **Dual Objectives**: The fitness function returns two metrics to the NSGA-II/Hybrid optimizer: `(Segmentation Loss, Inference Time)`.
- **ONNX Export**: Once the absolute best hyperparameter and architecture combination is found, the optimal model is constructed, traced, and exported directly to `best_unet.onnx` for lightning-fast inference by the FastAPI backend.

---

## 🎛️ Hyperparameters Tuned

During execution, the MHO algorithms actively tune a 4-dimensional continuous and discrete space:

1. **Learning Rate (`1e-5` to `1e-2`)**: Controls the step size during the Adam optimizer's gradient descent.
2. **Weight Decay (`1e-6` to `1e-3`)**: L2 regularization penalty to prevent the model from overfitting to the retinal dataset.
3. **Batch Size (`2` to `8`)**: The number of vessel images processed concurrently. (Treated as a continuous variable by the swarm but rounded safely at evaluation).
4. **Architecture Choice (`0` to `1`)**: A discrete NAS parameter. If rounded to 0, it dynamically instantiates the standard **U-Net**. If rounded to 1, it instantiates the deeper **ResUNet**.

---

## 🚀 Setup & Installation

### 1. Run the Backend (FastAPI + AI Engine)
Requires Python 3.8+
```bash
cd backend/

# Install required numerical, vision, and web libraries
pip install fastapi uvicorn python-multipart opencv-python numpy celery mlflow torch torchvision onnx pymoo

# Start the API server
uvicorn app:app --reload
```
*The server will be available at `http://127.0.0.1:8000`. Models are monitored utilizing MLflow.*

### 2. Run the Auto-Tuner (Optional)
To initiate the metaheuristic training process and generate a new `.onnx` weights file:
```bash
cd backend/
python train.py
```

### 3. Run the Frontend (React + Vite)
Requires Node.js 16+
```bash
cd backend/frontend/

# Install React dependencies
npm install

# Start the Vite development server
npm run dev
```
*The application UI will be available in your browser (usually `http://localhost:5173`).*
