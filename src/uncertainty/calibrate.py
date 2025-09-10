import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import cv2
import os
import json


DATA_DIR = 'data/processed/fer2013'
MODEL_PATH = 'models/fer_mobilenet_v2_dropout.pth'
TEMP_SAVE_PATH = 'models/temperature.json'
NUM_CLASSES = 7
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def opencv_loader(path):
    return cv2.imread(path)

def ece(preds, labels, n_bins=15):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = np.max(preds, axis=1), np.argmax(preds, axis=1)
    accuracies = (predictions == labels)

    ece_val = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece_val += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece_val

def brier_score(preds, labels):
    """Brier Score."""
    one_hot_labels = np.eye(NUM_CLASSES)[labels]
    return np.mean(np.sum((preds - one_hot_labels) ** 2, axis=1))

class ModelWithTemperature(nn.Module):
    """
    A thin wrapper for a model to tune the temperature parameter.
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.scale_logits(logits)

    def scale_logits(self, logits):
        return logits / self.temperature

def main():
    print(f"Using device: {DEVICE}")

    model = models.mobilenet_v2()
    model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(1280, NUM_CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), val_transform, loader=opencv_loader)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            all_logits.append(logits)
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    uncalibrated_probs = torch.softmax(all_logits, dim=1).cpu().numpy()
    ece_before = ece(uncalibrated_probs, all_labels.cpu().numpy())
    brier_before = brier_score(uncalibrated_probs, all_labels.cpu().numpy())
    print(f"Before Calibration - ECE: {ece_before:.4f}, Brier Score: {brier_before:.4f}")


    scaled_model = ModelWithTemperature(model)
    optimizer = optim.LBFGS(scaled_model.parameters(), lr=0.01, max_iter=50)
    nll_criterion = nn.CrossEntropyLoss()

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(scaled_model(next(iter(val_loader))[0].to(DEVICE)), next(iter(val_loader))[1].to(DEVICE))
        loss.backward()
        return loss

    optimizer.step(eval)
    optimal_temp = scaled_model.temperature.item()
    
    print(f"Optimal Temperature: {optimal_temp:.4f}")
    with open(TEMP_SAVE_PATH, 'w') as f:
        json.dump({'temperature': optimal_temp}, f)
    print(f"Saved temperature to {TEMP_SAVE_PATH}")

    calibrated_logits = all_logits / optimal_temp
    calibrated_probs = torch.softmax(calibrated_logits, dim=1).cpu().numpy()
    ece_after = ece(calibrated_probs, all_labels.cpu().numpy())
    brier_after = brier_score(calibrated_probs, all_labels.cpu().numpy())
    print(f"After Calibration - ECE: {ece_after:.4f}, Brier Score: {brier_after:.4f}")
    
    if ece_after < ece_before:
        print("\nSuccess! ECE was reduced, indicating better calibration.")

if __name__ == '__main__':
    main()

