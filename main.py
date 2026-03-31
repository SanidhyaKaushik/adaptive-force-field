import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm 
import sys
print(sys.executable)
# Import functional components from your modules
from dataset_gen import ForceFieldDatasetGenerator
from evaluator import ForceFieldEvaluator
from plotting import plot_confidence_heatmap
from adversarial import find_adversarial_points, plot_adversarial_vulnerability

class AdaptiveForceFieldRunner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = ForceFieldDatasetGenerator()
        self.evaluator = None
        self.data = None
        self.is_trained = False

    def prepare_dataset(self):
        """Interactive dataset generation and preprocessing."""
        print("\n--- Dataset Preparation ---")
        self.data = self.generator.generate()
        
        # Split: 60% Train, 20% Val, 20% Calibration
        X, Y = self.data['X'], self.data['Y']
        X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
        self.X_val, self.X_cal, self.Y_val, self.Y_cal = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
        self.X_train, self.Y_train = X_train, Y_train

        # Initialize/Reset the Evaluator with new physical parameters
        self.evaluator = ForceFieldEvaluator(
            charges=self.data['charges'],
            positions=self.data['positions'],
            device=self.device,
            alpha=0.95,
            max_error=0.5
        )
        self.is_trained = False

    def execute_training_pipeline(self, epochs=50):
        """Full training and conformal calibration workflow."""
        # 150 Epochs allows the Residual Network to fully converge
        # Batch size 256 stabilizes the high-gradient regions
        epochs = 150 
        batch = 256
        lr = 1e-3

        print("\n[1/3] Training Force Predictor...")
        self.evaluator.train_pred_model(self.X_train, self.Y_train, self.X_val, self.Y_val, 
                                       num_epochs=epochs, batch_size=batch, lr=lr)
        
        print("\n[2/3] Training Error Estimator...")
        self.evaluator.train_err_model(self.X_train, self.Y_train, self.X_val, self.Y_val, 
                                      num_epochs=epochs, batch_size=batch, lr=lr)
        
        print("\n[3/3] Performing Conformal Calibration...")
        self.evaluator.initialize_conformal_pred()
        self.evaluator.calibrate_conformal_pred(self.X_cal, self.Y_cal)
        
        self.is_trained = True
        print("\n--- Pipeline Complete: Model is Ready ---")

    
    def predict_single_point(self):
        """Allows user to query the force field at a specific coordinate."""
        print("\n--- Single Point Inference ---")
        try:
            coords = input("Enter coordinates x, y, z (comma separated): ")
            x_test = np.array([float(c.strip()) for c in coords.split(',')])
            
            start_time = time.time()
            field, conf, flag = self.evaluator.get_force_field(x_test)
            end_time = time.time()

            source = "Brute Force Physics (Low Confidence)" if flag == -1 else "Neural Network (Trusted)"
            
            print(f"\nResults for Point {x_test}:")
            print(f"  > Force Vector: {np.round(field, 5)}")
            print(f"  > Confidence Score: {conf:.4f}")
            print(f"  > Compute Source: {source}")
            print(f"  > Query Time: {(end_time - start_time)*1000:.2f} ms")
        except Exception as e:
            print(f">> Error in input: {e}")

    def evaluate_test_set_performance(self):
        """Runs the hybrid system on a large test set and reports metrics."""
        print("\n--- Full System Performance Evaluation ---")
        # Use the validation set as a proxy for the test set
        X_test = self.X_val
        Y_gt = self.Y_val # Ground Truth (or observed)
        
        start_time = time.time()
        fields, confs, flags = self.evaluator.get_force_field(X_test)
        total_time = time.time() - start_time

        fallback_ratio = np.mean(flags == -1)
        mse = np.mean((fields - Y_gt)**2)
        
        print(f"\nTest Summary ({len(X_test)} points):")
        print(f"  > Mean Squared Error: {mse:.6e}")
        print(f"  > Physics Fallback Ratio: {fallback_ratio*100:.2f}%")
        print(f"  > AI Trust Ratio: {(1 - fallback_ratio)*100:.2f}%")
        print(f"  > Avg Time Per Point: {(total_time/len(X_test))*1000:.4f} ms")

    # --- Analysis & Visualization Methods ---

    def visualize_learning_convergence(self):
        """Plots the training and validation loss history."""
        self.evaluator.plot_history()

    def analyze_data_scaling_efficiency(self):
        """
        Analyzes how increasing training data reduces the physics fallback ratio.
        Prompts user for max size and step size (in thousands), starting from 10K.
        """
        # Calculate available training data in thousands
        max_available_k = len(self.X_train) // 1000
        
        if max_available_k < 10:
            print(f">> Error: Current training set only has {max_available_k}K points. "
                  "Scaling analysis requires at least 11K.")
            return

        # 1. Get User Input
        try:
            prompt = f"Enter max size and step size in thousands (Available: {max_available_k}K, e.g., {max_available_k},5): "
            user_input = input(prompt)
            max_k, step_k = map(int, user_input.split(','))
        except (ValueError, AttributeError):
            print(">> Invalid format. Please enter numbers as 'max,step' (e.g., 50,5).")
            return

        # 2. Validation and Capping
        if max_k <= 10:
            print(">> Error: Max dataset size must be greater than 10 (thousands).")
            return

        if max_k > max_available_k:
            print(f">> Note: Requested {max_k}K exceeds training data. Capping at {max_available_k}K.")
            max_k = max_available_k

        # 3. Generate size range starting from 10,000
        # k * 1000 ensures we use the actual integer counts
        sizes = [k * 1000 for k in range(10, max_k + 1, step_k)]
        ratios = []

        print(f"Benchmarking model efficiency across {len(sizes)} increments...")

        # 4. Iterative Training and Evaluation
        for s in tqdm(sizes, desc="Scaling Analysis"):
            # Initialize fresh models for each data subset
            temp_eval = ForceFieldEvaluator(
                charges=self.data['charges'],
                positions=self.data['positions'],
                device=self.device
            )
            
            # Train on the subset [:s]
            temp_eval.train_pred_model(self.X_train[:s], self.Y_train[:s], self.X_val, self.Y_val, num_epochs=150)
            temp_eval.train_err_model(self.X_train[:s], self.Y_train[:s], self.X_val, self.Y_val, num_epochs=150)
            
            # Calibrate with the fixed calibration set
            temp_eval.initialize_conformal_pred()
            temp_eval.calibrate_conformal_pred(self.X_cal, self.Y_cal)
            
            # Evaluate fallback ratio on a fixed validation slice (500 points)
            _, _, flags = temp_eval.get_force_field(self.X_val[:500])
            ratios.append(np.mean(flags == -1))
        
        # 5. Plotting Results
        plt.figure(figsize=(10, 6))
        # X-axis shows values in K (thousands) for better readability
        plt.plot([s/1000 for s in sizes], ratios, 'o--', color='tab:red', label='Fallback Ratio')
        plt.title(f"Efficiency Gain: 10K to {max_k}K Samples")
        plt.xlabel("Training Set Size (Thousands)")
        plt.ylabel("Ratio of Brute Force Fallbacks")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend()
        plt.show()

    def generate_spatial_confidence_map(self):
        """Visualizes the 2D spatial safety zones where the NN is trusted."""
        plot_confidence_heatmap(self.evaluator, resolution=100, range_lim=4.0)

    def evaluate_parameter_sensitivity(self):
        """Sweeps error tolerance and confidence levels to see impact on computation cost."""
        # Sensitivity to Max Error
        errors = np.linspace(0.1, 1.5, 15)
        r_err = []
        orig_e = self.evaluator.max_error
        for e in errors:
            self.evaluator.max_error = e
            _, _, f = self.evaluator.get_force_field(self.X_val)
            r_err.append(np.mean(f == -1))
        self.evaluator.max_error = orig_e

        # Sensitivity to Confidence Level (Alpha)
        alphas = np.linspace(0.8, 0.99, 15)
        r_alpha = []
        orig_a = self.evaluator.alpha
        for a in alphas:
            self.evaluator.alpha = a
            _, _, f = self.evaluator.get_force_field(self.X_val)
            r_alpha.append(np.mean(f == -1))
        self.evaluator.alpha = orig_a

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(errors, r_err, 's-', color='tab:blue', label='Cost')
        ax1.set_title("Impact of Error Tolerance on Computation")
        ax1.set_xlabel("Max Error Allowed")
        ax1.set_ylabel("Fallback Ratio")

        ax2.plot(alphas, r_alpha, 's-', color='tab:green', label='Cost')
        ax2.set_title("Impact of Confidence Level on Computation")
        ax2.set_xlabel("Alpha (Target Reliability)")
        plt.tight_layout(); plt.show()

    def conduct_adversarial_stress_test(self):
        """Locates and analyzes points where the model is mathematically most vulnerable."""
        results = find_adversarial_points(self.evaluator, n_points=5)
        plot_adversarial_vulnerability(self.evaluator, results)

    def run_interface(self):
        """Main terminal-based user menu."""
        while True:
            print("\n" + "="*55)
            print("   ADAPTIVE FORCE FIELD EVALUATION (AFFE) TOOL")
            print("="*55)
            print(" [DATA & TRAINING]")
            print("  1. Prepare Dataset (Generator)")
            print("  2. Train & Calibrate Models")
            print("\n [INFERENCE & UTILITY]")
            print("  3. Predict Force at a Custom Point (x,y,z)")
            print("  4. Evaluate System Performance (Full Test Set)")
            print("\n [RESEARCH EXPERIMENTS]")
            print("  5. Visualize Learning History (Loss Curves)")
            print("  6. Analyze Data Scaling Laws")
            print("  7. Generate Spatial Confidence Heatmap")
            print("  8. Evaluate Parameter Sensitivity")
            print("  9. Conduct Adversarial Stress Test")
            print("\n  0. Exit")
            
            choice = input("\nSelection: ")
            
            if choice == '0': break
            if choice == '1': 
                self.prepare_dataset()
                continue
            
            if self.evaluator is None:
                print(">> Please generate a dataset first (Option 1).")
                continue

            if not self.is_trained and choice in ['2','3','4','5','6','7','8','9']:
                self.execute_training_pipeline()

            if choice == '3': self.predict_single_point()
            elif choice == '4': self.evaluate_test_set_performance()
            elif choice == '5': self.visualize_learning_convergence()
            elif choice == '6': self.analyze_data_scaling_efficiency()
            elif choice == '7': self.generate_spatial_confidence_map()
            elif choice == '8': self.evaluate_parameter_sensitivity()
            elif choice == '9': self.conduct_adversarial_stress_test()

if __name__ == "__main__":
    #from tqdm import tqdm # Added for the scaling loop progress bar
    runner = AdaptiveForceFieldRunner()
    runner.run_interface()