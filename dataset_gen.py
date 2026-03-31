import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader , TensorDataset
import numpy as np

class ForceFieldDatasetGenerator:
    def __init__(self):
        self.k = 1.0  # Coulomb constant
        self.epsilon = 1e-3  # Softening factor to avoid singularities

    def _sample_spherical_shell(self, n, r_min, r_max):
        """Generates n points uniformly within a spherical shell [r_min, r_max]."""
        # Generate random directions (unit vectors)
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(-1, 1, n)
        sin_theta = np.sqrt(1 - cos_theta**2)

        x = sin_theta * np.cos(phi)
        y = sin_theta * np.sin(phi)
        z = cos_theta
        unit_vectors = np.stack([x, y, z], axis=1)

        # Sample radii. To ensure uniform volume density, we sample r^3
        # However, for simplicity in field evaluation, a linear radial sample is often used.
        # We will use the cubic root of uniform to ensure uniform spatial density.
        r = (np.random.uniform(r_min**3, r_max**3, n))**(1/3)
        
        return unit_vectors * r[:, np.newaxis]

    def calculate_field(self, probe_pos, charge_pos, charge_vals):
        """Vectorized Coulomb calculation for a batch of probe points."""
        # probe_pos: (M, 3), charge_pos: (N, 3), charge_vals: (N,)
        diff = probe_pos[:, np.newaxis, :] - charge_pos[np.newaxis, :, :] # (M, N, 3)
        dist = np.linalg.norm(diff, axis=2, keepdims=True) # (M, N, 1)
        
        # E = sum ( k * q * r_vec / r^3 )
        field_components = self.k * charge_vals[np.newaxis, :, np.newaxis] * diff / (dist**3 + self.epsilon)
        return np.sum(field_components, axis=1) # (M, 3)

    def generate(self):
        print("--- Adaptive Force Field Dataset Generator ---")
        
        # 1. Collect Inputs
        n_charges = int(input("Number of charges (n): "))
        n_samples = int(input("Number of probe data points (m): "))
        
        q_min = float(input("Min charge value: "))
        q_max = float(input("Max charge value: "))
        
        rq_min = float(input("Min charge distance from origin: "))
        rq_max = float(input("Max charge distance from origin: "))
        
        rf_min = float(input("Min field (probe) distance from origin: "))
        rf_max = float(input("Max field (probe) distance from origin: "))

        # 2. Generate Charges and Positions
        print(f"\nGenerating {n_charges} charges...")
        charge_positions = self._sample_spherical_shell(n_charges, rq_min, rq_max)
        charge_values = np.random.uniform(q_min, q_max, n_charges)

        # 3. Generate Random Probe Points
        print(f"Generating {n_samples} random probe points...")
        X_random = self._sample_spherical_shell(n_samples, rf_min, rf_max)
        Y_random = self.calculate_field(X_random, charge_positions, charge_values)

        # 4. Interactive User Data Entry
        X_manual = []
        Y_manual = []
        
        add_more = input("\nWould you like to add custom data points manually? (y/n): ").lower()
        if add_more == 'y':
            while True:
                try:
                    line = input("Enter x, y, z (or 'done' to finish): ")
                    if line.lower() == 'done':
                        break
                    coords = [float(c) for c in line.split(',')]
                    if len(coords) != 3:
                        print("Please enter exactly 3 coordinates separated by commas.")
                        continue
                    
                    pos = np.array([coords])
                    # We calculate the ground truth for the manual point to ensure it matches the system
                    field = self.calculate_field(pos, charge_positions, charge_values)
                    
                    X_manual.append(pos[0])
                    Y_manual.append(field[0])
                    print(f"Added. Calculated Field: {field[0]}")
                except ValueError:
                    print("Invalid input format.")

        # 5. Combine Datasets
        if len(X_manual) > 0:
            X_total = np.vstack([X_random, np.array(X_manual)])
            Y_total = np.vstack([Y_random, np.array(Y_manual)])
        else:
            X_total = X_random
            Y_total = Y_random

        print(f"\nFinal dataset ready. Total points: {len(X_total)}")
        
        # Return everything needed for the Evaluator
        return {
            "X": X_total.astype(np.float32),
            "Y": Y_total.astype(np.float32),
            "charges": charge_values.astype(np.float32),
            "positions": charge_positions.astype(np.float32)
        }