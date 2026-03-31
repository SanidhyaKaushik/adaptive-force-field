# AFFE: Adaptive Force Field Evaluation

A high-performance, reliability-guaranteed framework for evaluating force fields in $\mathbb{R}^3$. This project combines **Deep Residual Learning** with **Conformal Prediction** to create a hybrid AI-Physics surrogate model.

## 🌟 The Core Innovation
Evaluating $O(N)$ Coulomb forces is computationally expensive. Neural Networks (NN) offer $O(1)$ speed but are unreliable near physical singularities (charges). 

**AFFE** solves this by:
1. **Surrogate Modeling:** A Deep ResNet with **Huber Loss** maps the complex $1/r^2$ field.
2. **Uncertainty Quantification:** Using **Conformal Prediction** to generate locally adaptive confidence intervals.
3. **Selective Fallback:** Automatically triggering Brute Force physics only when the AI's confidence drops below a user-defined safety threshold ($\alpha$).

## 📊 Key Research Experiments
This framework includes an automated runner for 6 critical experiments:
* **Convergence Analysis:** Training vs. Validation loss for the ResNet and Error models.
* **Scaling Laws:** How training data volume reduces the need for physics fallback.
* **Spatial Confidence Heatmaps:** Visualizing the "Safe Zones" in 3D space.
* **Parameter Sensitivity:** Impact of error tolerance on computational overhead.
* **Adversarial Stress Test:** Using **Gradient Ascent** to find and flag AI "blind spots."

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/SanidhyaKaushik/adaptive-force-field.git
cd adaptive-force-field

# Install dependencies
pip install -r requirements.txt