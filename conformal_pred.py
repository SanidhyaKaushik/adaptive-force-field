import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader , TensorDataset
import numpy as np



class AdaptiveConformalForceField:
    def __init__(self, model, error_model, device=torch.device("cpu")):
        self.model = model
        self.error_model = error_model
        self.calibrated_scores = None
        self.device = device

    def calibrate(self, x_cal, y_cal):
        self.error_model.eval()
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.as_tensor(x_cal, device=self.device).float()
            y_tensor = torch.as_tensor(y_cal, device=self.device).float()
            
            y_pred = self.model(x_tensor)
            sigmas = self.error_model(x_tensor).squeeze()
            actual_errors = torch.norm(y_tensor - y_pred, dim=1)
            
            self.calibrated_scores = (actual_errors / sigmas).cpu().numpy()
            self.calibrated_scores.sort()

    def get_confidence_scores(self, x_test, max_error):
        """Vectorized confidence calculation for a batch of points"""
        self.error_model.eval()
        with torch.no_grad():
            x_tensor = torch.as_tensor(x_test, device=self.device).float()
            sigmas_test = self.error_model(x_tensor).squeeze().cpu().numpy()

            # target_score is the score required to stay under max_error
            # if score_required > threshold_in_calibration, we are confident.
            target_scores = max_error / sigmas_test
            
            # Use searchsorted on the whole batch at once
            cnt_better = np.searchsorted(self.calibrated_scores, target_scores)
            confidences = cnt_better / len(self.calibrated_scores)
            return confidences