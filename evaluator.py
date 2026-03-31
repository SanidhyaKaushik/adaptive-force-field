import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Importing our custom modules
from conformal_pred import AdaptiveConformalForceField
from models import ForceFieldPredictor, ErrorPredictor

class ForceFieldEvaluator:
    def __init__(self, charges, positions, 
                 pred_embed_dim=128, 
                 pred_hidden_dim=128,
                 pred_n_blocks=4, 
                 err_hidden_dim=128, 
                 device=torch.device("cpu"), 
                 alpha=0.95, 
                 max_error=0.5):
        """
        Orchestrates the training, calibration, and fallback logic.
        Upgraded for Residual Architectures and stable physical gradients.
        """
        self.charges = charges
        self.positions = positions
        self.device = device
        self.alpha = alpha
        self.max_error = max_error
        
        # Initialize Residual Architectures
        self.pred_model = ForceFieldPredictor(
            embed_dim=pred_embed_dim, 
            hidden_dim=pred_hidden_dim,
            n_blocks=pred_n_blocks
        ).to(device)
        
        self.err_model = ErrorPredictor(
            hidden_dim=err_hidden_dim
        ).to(device)
        
        self.conformal_pred = None
        
        # History tracking for Plotting Convergence
        self.history = {
            "pred": {"train": [], "val": []},
            "err": {"train": [], "val": []}
        }

    def _prepare_loader(self, X, Y, batch_size, shuffle=True):
        tensor_x = torch.as_tensor(X, dtype=torch.float32)
        tensor_y = torch.as_tensor(Y, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_y)
        # pin_memory speeds up data transfer to GPU
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                          pin_memory=True if self.device.type == 'cuda' else False)

    def train_pred_model(self, X_train, Y_train, X_val, Y_val, num_epochs=150, batch_size=256, lr=1e-3):
        """Trains the Force Predictor using Huber Loss and OneCycleLR for smooth convergence."""
        train_loader = self._prepare_loader(X_train, Y_train, batch_size)
        val_loader = self._prepare_loader(X_val, Y_val, batch_size, shuffle=False)
        
        # AdamW provides better regularization for residual networks
        optimizer = torch.optim.AdamW(self.pred_model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Huber Loss: Prevents gradient explosions near 1/r^2 singularities
        criterion = nn.HuberLoss(delta=1.0) 
        
        # OneCycleLR: Fast, stable convergence (Super-Convergence)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=num_epochs
        )

        print(f"--- Training Predictor (ResNet, Batch: {batch_size}, Epochs: {num_epochs}) ---")
        for epoch in range(num_epochs):
            self.pred_model.train()
            train_loss = 0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                
                preds = self.pred_model(X)
                loss = criterion(preds, y)
                loss.backward()
                
                # Gradient Clipping: Essential for smoothing jittery curves
                torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                train_loss += loss.item() * X.size(0)
            
            # Validation Phase
            self.pred_model.eval()
            val_loss = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    v_preds = self.pred_model(X)
                    val_loss += criterion(v_preds, y).item() * X.size(0)
            
            self.history["pred"]["train"].append(train_loss / len(X_train))
            self.history["pred"]["val"].append(val_loss / len(X_val))
            
            if (epoch+1) % 25 == 0:
                print(f"Epoch {epoch+1}: Val Loss {self.history['pred']['val'][-1]:.6f}")

    def train_err_model(self, X_train, Y_train, X_val, Y_val, num_epochs=100, batch_size=256, lr=1e-3):
        """Trains the Error Estimator using a Cosine annealing schedule."""
        self.pred_model.eval()
        def get_residuals(X, Y):
            X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
            Y_t = torch.as_tensor(Y, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                preds = self.pred_model(X_t)
                return torch.norm(preds - Y_t, dim=1, keepdim=True).cpu().numpy()

        err_train_targets = get_residuals(X_train, Y_train)
        err_val_targets = get_residuals(X_val, Y_val)

        train_loader = self._prepare_loader(X_train, err_train_targets, batch_size)
        val_loader = self._prepare_loader(X_val, err_val_targets, batch_size, shuffle=False)
        
        optimizer = torch.optim.AdamW(self.err_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.HuberLoss() 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        print(f"\n--- Training Error Estimator (Batch: {batch_size}) ---")
        for epoch in range(num_epochs):
            self.err_model.train()
            train_loss = 0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.err_model(X), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X.size(0)

            self.err_model.eval()
            val_loss = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    val_loss += criterion(self.err_model(X), y).item() * X.size(0)

            self.history["err"]["train"].append(train_loss / len(X_train))
            self.history["err"]["val"].append(val_loss / len(X_val))
            scheduler.step()

    def initialize_conformal_pred(self):
        """Sets up the Conformal Predictor wrapper."""
        self.conformal_pred = AdaptiveConformalForceField(self.pred_model, self.err_model, device=self.device)
  
    def calibrate_conformal_pred(self, X_calib, Y_calib): 
        """Calculates non-conformity scores on the hold-out calibration set."""
        self.conformal_pred.calibrate(X_calib, Y_calib)

    def get_force_field(self, X_test):
        """
        Calculates force field with optimized shape handling for both single points and batches.
        Implements vectorized Brute Force fallback for points with low confidence.
        """
        # 1. Input Standardization
        if isinstance(X_test, list):
            X_test = np.array(X_test)
            
        if X_test.ndim == 1 or X_test.shape[0] == 1:
            X_test_input = X_test.reshape(1, 3)
            is_single_point = True
        else:
            X_test_input = X_test
            is_single_point = False

        # 2. Physical Parameter Sanitization (The Fix for the Broadcast Error)
        # This ensures positions are (N, 3) and charges are (N,)
        pos_fixed = self.positions.reshape(-1, 3)
        charges_fixed = self.charges.flatten()

        # 3. Neural Network Path
        self.pred_model.eval()
        X_tensor = torch.as_tensor(X_test_input, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            nn_preds = self.pred_model(X_tensor).cpu().numpy()
        
        # 4. Conformal Path
        # get_confidence_scores is already vectorized
        confidences = self.conformal_pred.get_confidence_scores(X_test_input, self.max_error)
        
        # 5. Hybrid Logic
        fallback_mask = confidences < self.alpha
        
        final_fields = nn_preds.copy()
        fallback_flags = np.where(fallback_mask, -1, 0)

        # 6. Vectorized Brute Force (Only for uncertain points)
        if np.any(fallback_mask):
            fallback_mask = np.asarray(fallback_mask).reshape(-1)

            X_fb = X_test_input[fallback_mask] # (M_uncertain, 3)
            
            # (M_uncertain, 1, 3) - (1, N_charges, 3) -> (M_uncertain, N_charges, 3)
            diff = X_fb[:, np.newaxis, :] - pos_fixed[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2, keepdims=True)
            
            # (1, N_charges, 1)
            q = charges_fixed[np.newaxis, :, np.newaxis]
            
            # (M_uncertain, N_charges, 3)
            # Using 1e-2 epsilon for stability near singularities
            force_comps = q * diff / (dist**3 + 1e-2)
            
            # Sum over axis 1 (the charges) -> (M_uncertain, 3)
            brute_results = np.sum(force_comps, axis=1)
            
            # Update the specific indices
            final_fields[fallback_mask] = brute_results

        # 7. Output Formatting
        # if is_single_point:
        #     return final_fields[0], confidences[0], fallback_flags[0]
        return final_fields, confidences, fallback_flags
    
    def plot_history(self):
        """Generates comprehensive training history plots (Plot #1)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        for ax, model_key, title in zip([ax1, ax2], ["pred", "err"], ["Force Predictor", "Error Estimator"]):
            ax.plot(self.history[model_key]["train"], label="Train Loss", color='tab:blue', lw=1.5)
            ax.plot(self.history[model_key]["val"], label="Val Loss", color='tab:orange', linestyle='--', lw=1.5)
            ax.set_title(f"{title} Training History")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Huber Loss")
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, which="both", ls="-", alpha=0.2)

        plt.tight_layout()
        plt.show()