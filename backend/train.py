import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import mlflow
import onnx
from model import UNet, ResUNet
from dataset import RetinalDataset

def evaluate_fitness(hyperparams, dataset_name="SYSTEM_AUTO"):
    lr = float(hyperparams[0])
    weight_decay = float(hyperparams[1])
    batch_size = max(1, int(hyperparams[2]))
    arch_choice = int(np.round(hyperparams[3])) if len(hyperparams) > 3 else 0
    
    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "architecture": "ResUNet" if arch_choice == 1 else "UNet",
            "dataset_config": "HYBRID_MERGED"
        })

        root_dir = os.path.join(os.path.dirname(__file__), "datasets")
        
        # System Automatically grabs all working datasets!
        full_datasets = []
        for name in ["DRIVE", "CHASE_DB1", "FIVES"]:
            try:
                ds = RetinalDataset(root_dir=root_dir, dataset_name=name, split="train")
                if len(ds) > 0: full_datasets.append(ds)
            except Exception as e:
                pass
        
        if not full_datasets:
            return float('-inf')
            
        massive_combined_dataset = ConcatDataset(full_datasets)
        
        subset_size = min(8, len(massive_combined_dataset))
        if subset_size == 0: return float('-inf')
            
        subset_indices = np.random.choice(len(massive_combined_dataset), subset_size, replace=False)
        fitness_dataset = Subset(massive_combined_dataset, subset_indices)
        loader = DataLoader(fitness_dataset, batch_size=batch_size, shuffle=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if arch_choice == 1:
            model = ResUNet(in_channels=3, out_channels=1).to(device)
        else:
            model = UNet(in_channels=3, out_channels=1).to(device)
            
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()
        
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
                
        inference_time = (time.time() - start_time) / subset_size * 1000 
        fitness_score = -total_loss
        
        mlflow.log_metrics({"fitness_score": fitness_score, "inference_time_ms": inference_time})
        
        # MUST Return two objectives for NSGA-II: (maximize dice representation, maximize speed representation)
        # Where inference_time is faster when it's smaller. We negative it so higher is better for our max-optimizer framework.
        return fitness_score, -inference_time

def export_to_onnx(model, dummy_input, filename="best_model.onnx"):
    model.eval()
    temp_path = os.path.join(os.path.dirname(__file__), filename)
    torch.onnx.export(model, dummy_input, temp_path, opset_version=11,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"System Model successfully updated into {temp_path}")

if __name__ == "__main__":
    from hybrid_mho import HybridMultiObjectiveMHO
    
    bounds = [(1e-5, 1e-2), (1e-6, 1e-3), (2, 8), (0, 1)]
    mlflow.set_experiment("MHO_System_AutoTuner")
    
    with mlflow.start_run(run_name="Hybrid_AI_Tuning"):
        opt = HybridMultiObjectiveMHO(
            fitness_function=lambda hp: evaluate_fitness(hp, dataset_name="ALL"),
            bounds=bounds,
            pop_size=3,
            max_iterations=2
        )
        best_params, best_score = opt.optimize()
        
        mlflow.log_metrics({"best_overall_score": best_score})
        mlflow.log_params({
            "best_lr": best_params[0],
            "best_wd": best_params[1],
            "best_batch": best_params[2],
            "best_arch": best_params[3]
        })
    
    arch_choice = int(np.round(best_params[3]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if arch_choice == 1: best_model = ResUNet(in_channels=3, out_channels=1).to(device)
    else: best_model = UNet(in_channels=3, out_channels=1).to(device)
        
    print("=========================================")
    print("Training Best Architected Model using Swarm Hyperparameters...")
    optimizer = optim.Adam(best_model.parameters(), lr=best_params[0], weight_decay=best_params[1])
    criterion = nn.BCELoss()
    
    root_dir = os.path.join(os.path.dirname(__file__), "datasets")
    full_datasets = []
    for name in ["DRIVE", "CHASE_DB1", "FIVES"]:
        try:
            ds = RetinalDataset(root_dir=root_dir, dataset_name=name, split="train")
            if len(ds) > 0: full_datasets.append(ds)
        except:
            pass
            
    if full_datasets:
        massive_combined_dataset = ConcatDataset(full_datasets)
        subset_size = min(32, len(massive_combined_dataset)) # Mini subset to prevent freezing your terminal for 6 hours
        subset_indices = np.random.choice(len(massive_combined_dataset), subset_size, replace=False)
        training_dataset = Subset(massive_combined_dataset, subset_indices)
        loader = DataLoader(training_dataset, batch_size=max(1, int(best_params[2])), shuffle=True)
        
        best_model.train()
        epochs = 15
        for epoch in range(epochs):
            epoch_loss = 0
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = best_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Final Model Train Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.4f}")
            
    print("Training complete! Exporting solid AI model to ONNX...")
    
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    export_to_onnx(best_model, dummy_input, "best_unet.onnx")
