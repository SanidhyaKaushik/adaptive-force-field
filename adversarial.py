import torch
import numpy as np
import matplotlib.pyplot as plt

def find_adversarial_points(evaluator, n_points=5, iterations=100, lr=0.1, bounds=[-5, 5]):
    """
    Performs Gradient Ascent on the spatial coordinates to maximize predicted uncertainty,
    effectively finding the 'blind spots' of the Neural Network.
    """
    device = evaluator.device
    
    # 1. Initialize random points in R3 as trainable parameters
    # Starting points are sampled within the specified physical bounds
    x_adv = torch.randn(n_points, 3, device=device).uniform_(bounds[0], bounds[1])
    x_adv = torch.nn.Parameter(x_adv, requires_grad=True)
    
    # We only need the Error Model to find the highest uncertainty
    evaluator.err_model.eval()
    optimizer = torch.optim.Adam([x_adv], lr=lr)
    
    print(f"Searching for {n_points} adversarial points (maximizing uncertainty)...")

    for i in range(iterations):
        optimizer.zero_grad()
        
        # Predict uncertainty (sigma) at current coordinates
        # We use .view(-1) to handle batches of any size safely
        sigma_pred = evaluator.err_model(x_adv).view(-1)
        
        # Loss = -Sum(sigma). Minimizing negative sigma is Maximizing sigma.
        # This drives the confidence score toward zero.
        loss = -torch.sum(sigma_pred)
        
        loss.backward()
        optimizer.step()
        
        # Physical Constraint: Keep points within the simulation volume
        with torch.no_grad():
            x_adv.clamp_(bounds[0], bounds[1])

    # 2. Evaluate discovered points
    x_final = x_adv.detach().cpu().numpy()
    results = []
    
    print("\n--- Adversarial Search Results ---")
    for i in range(n_points):
        # We slice [i:i+1] to pass a (1, 3) array, ensuring compatibility
        # with the batch-processing logic in Force_Field_Evaluator.py
        current_pt = x_final[i:i+1, :]
        
        field, confidence, fallback = evaluator.get_force_field(current_pt)
        
        results.append({
            "position": x_final[i],
            "confidence": confidence,
            "fallback_triggered": (fallback == -1)
        })
        
        status = "FALLBACK TRIGGERED" if fallback == -1 else "ACCEPTED (MODEL VULNERABLE)"
        print(f"Point {i+1}: Pos {np.round(x_final[i], 2)} | Conf: {confidence.item():.4f} | {status}")

    return results

def plot_adversarial_vulnerability(evaluator, adversarial_results):
    """
    Visualizes the robustness of the system by comparing adversarial scores 
    against the calibration distribution.
    """
    # 1. Extract Predicted Sigmas for the discovered points
    adv_positions = np.array([res['position'] for res in adversarial_results])
    evaluator.err_model.eval()
    
    with torch.no_grad():
        x_tensor = torch.as_tensor(adv_positions, dtype=torch.float32, device=evaluator.device)
        adv_sigmas = evaluator.err_model(x_tensor).view(-1).cpu().numpy()
    
    # 2. Calculate the Non-Conformity Scores (S = max_error / sigma)
    # These are the scores the system uses to search the calibration set.
    # Higher sigma results in a lower score (left side of the histogram).
    s_adv = evaluator.max_error / adv_sigmas
    
    # 3. Retrieve the calibration scores from the Conformal Predictor
    if evaluator.conformal_pred is None or evaluator.conformal_pred.calibrated_scores is None:
        print("Error: Conformal Predictor must be calibrated before plotting vulnerability.")
        return
        
    cal_scores = evaluator.conformal_pred.calibrated_scores
    
    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot the background distribution of "normal" errors
    plt.hist(cal_scores, bins=10, alpha=0.4, color='tab:blue', 
             label='Calibration Scores (Normal Behavior)', density=True)
    
    # Plot vertical lines for each adversarial point discovered
    for i, score in enumerate(s_adv):
        plt.axvline(score, color='red', linestyle='--', alpha=0.7, 
                    label='Adversarial Score' if i == 0 else "")
    
    plt.title("Adversarial Robustness Analysis\n(Scores close to 0 = High Uncertainty/Fallback)", fontsize=14)
    plt.xlabel("Non-Conformity Score ($max\_error / \sigma$)", fontsize=12)
    plt.ylabel("Relative Frequency", fontsize=12)
    
    # Highlight the "Safe Zone" (everything to the right of the red lines)
    plt.grid(True, which="both", ls="-", alpha=0.15)
    plt.legend()
    plt.show()