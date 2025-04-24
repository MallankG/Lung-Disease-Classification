#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Classification with 224x224 Input Size Models

This script focuses specifically on pre-trained models that use 224x224 input size
for image classification.
"""

# -----------------------------------------------------------------------------
# Section 1: Imports & Setup
# -----------------------------------------------------------------------------
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from tqdm import tqdm
import pandas as pd

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# Section 2: Dataset Preparation (224x224 Input Size)
# -----------------------------------------------------------------------------
# Dataset paths
data_dir = "Dataset-1"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

# Transformations for 224x224 input size
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Class names and sizes
sample_dataset = datasets.ImageFolder(train_dir)  # For class names
class_names = sample_dataset.classes
num_classes = len(class_names)
print(f"Classes detected: {class_names}")

# Dataset and Dataloader for 224x224
dataloaders = {}
batch_size = 32

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

dataloaders["train"] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
dataloaders["val"] = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
dataloaders["test"] = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print("✅ Data loaders created for 224x224 input size.")

# -----------------------------------------------------------------------------
# Section 3: Model Definitions (224x224 Models)
# -----------------------------------------------------------------------------
# Utility to freeze all except classifier head
def freeze_layers(model, classifier_attr=None):
    for name, param in model.named_parameters():
        param.requires_grad = False
    if classifier_attr is not None:
        head = getattr(model, classifier_attr)
        for param in head.parameters():
            param.requires_grad = True
    return model

# Adjust classifier head, apply dropout and freeze layers
def modify_model(name, base_model, num_classes=4, dropout_p=0.5):
    if any(x in name for x in ["resnet", "resnext", "wide_resnet"]):
        in_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )
        model = freeze_layers(base_model, classifier_attr="fc")

    elif "densenet" in name:
        in_features = base_model.classifier.in_features
        base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )
        model = freeze_layers(base_model, classifier_attr="classifier")

    elif "vgg" in name:
        in_features = base_model.classifier[-1].in_features
        base_model.classifier[-2] = nn.Dropout(p=dropout_p)
        base_model.classifier[-1] = nn.Linear(in_features, num_classes)
        model = freeze_layers(base_model, classifier_attr="classifier")

    elif "mobilenetv2" in name:
        in_features = base_model.classifier[-1].in_features
        base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )
        model = freeze_layers(base_model, classifier_attr="classifier")

    elif any(x in name for x in ["res2net", "darknet"]):
        if hasattr(base_model, 'reset_classifier'):
            base_model.reset_classifier(num_classes=num_classes)
            for param in base_model.get_classifier().parameters():
                param.requires_grad = True
            for pname, param in base_model.named_parameters():
                if not any(key in pname for key in ['classifier', 'fc', '_fc']):
                    param.requires_grad = False
            model = base_model
        else:
            in_features = base_model.get_classifier().in_features
            base_model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes)
            )
            model = freeze_layers(base_model, classifier_attr="classifier")

    else:
        raise ValueError(f"Model type not recognized for: {name}")

    return model

# 224x224 input size models
models_224 = {
    "resnet34": models.resnet34(pretrained=True),
    "resnet50": models.resnet50(pretrained=True),
    "resnet101": models.resnet101(pretrained=True),
    "densenet201": models.densenet201(pretrained=True),
    "vgg16": models.vgg16(pretrained=True),
    "vgg19": models.vgg19(pretrained=True),
    "mobilenetv2": models.mobilenet_v2(pretrained=True),
    "resnext50_32x4d": models.resnext50_32x4d(pretrained=True),
    "wide_resnet50_2": models.wide_resnet50_2(pretrained=True),
    "res2net50": timm.create_model('res2net50_26w_4s', pretrained=True),
    "darknet53": timm.create_model('darknet53', pretrained=True),
}

# Apply head modifications and freeze layers
models = {name: modify_model(name, model, num_classes=num_classes).to(device) for name, model in models_224.items()}

print("Models ready for training:")
print(list(models.keys()))

# -----------------------------------------------------------------------------
# Section 4: Loss Functions, Optimizer Setup & Metrics
# -----------------------------------------------------------------------------
# Custom Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        return self.alpha * (1 - pt) ** self.gamma * ce_loss

# Loss Functions Dictionary
loss_functions = {
    'crossentropy': nn.CrossEntropyLoss(),
    'label_smoothing': nn.CrossEntropyLoss(label_smoothing=0.1),
    'focal': FocalLoss()
}

# Optimizer Factory
def get_optimizer(model, name, lr, wd):
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

# Metric Calculation Function
def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }

# Hyperparameter Combinations
combos = [
    {"opt": "adam", "loss": "crossentropy", "lr": 1e-4, "wd": 1e-4},
    {"opt": "sgd", "loss": "label_smoothing", "lr": 1e-3, "wd": 5e-4},
    {"opt": "adamw", "loss": "focal", "lr": 1e-4, "wd": 1e-5},
    {"opt": "adam", "loss": "crossentropy", "lr": 5e-5, "wd": 1e-6}
]

print("✅ Loss functions, optimizers, and combos set.")

# -----------------------------------------------------------------------------
# Section 5: Training Loop — Train and Validate Model
# -----------------------------------------------------------------------------
def train_and_validate(model, model_name, combo_id, combo, dataloaders, device, class_names):
    optimizer = get_optimizer(model, combo["opt"], combo["lr"], combo["wd"])
    loss_fn = loss_functions[combo["loss"]]

    results_dir = f"results-1/combo_{combo_id}"
    os.makedirs(results_dir, exist_ok=True)

    best_val_acc = -1
    patience = 5
    counter = 0
    num_epochs = 10

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    if not any(p.requires_grad for p in model.parameters()):
        print(f"⚠️ Warning: All parameters are frozen for {model_name}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        for inputs, labels in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1} Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        train_loss /= len(dataloaders['train'])
        train_losses.append(train_loss)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloaders['val'], desc=f"Epoch {epoch+1} Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(dataloaders['val'])
        val_losses.append(val_loss)

        val_acc = accuracy_score(all_labels, all_preds)
        val_accuracies.append(val_acc)

        print(f"📊 Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early Stopping Checkpoint
        model_path = f"{results_dir}/{model_name}_c{combo_id}.pt"
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            counter += 1
            if counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

    # Save training history
    history_df = pd.DataFrame({
        "epoch": range(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "train_accuracy": train_accuracies,
        "val_loss": val_losses,
        "val_accuracy": val_accuracies
    })
    history_df.to_csv(f"{results_dir}/{model_name}_c{combo_id}_history.csv", index=False)

    return model_path, history_df

# -----------------------------------------------------------------------------
# Section 6: Test Evaluation, Plots & Confusion Matrix
# -----------------------------------------------------------------------------
def evaluate_and_visualize(model, model_name, combo_id, model_path, dataloaders, class_names, device):
    results_dir = f"results-1/combo_{combo_id}"
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test'], desc=f"Testing {model_name}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_labels, all_preds)

    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title(f"{model_name} Combo {combo_id} — Confusion Matrix")
    plt.savefig(f"{results_dir}/{model_name}_c{combo_id}_confmatrix.png")
    plt.close()

    return metrics

def plot_training_curves(history_df, model_name, combo_id):
    results_dir = f"results-1/combo_{combo_id}"

    # Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['epoch'], history_df['train_loss'], label="Train Loss")
    plt.plot(history_df['epoch'], history_df['val_loss'], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Combo {combo_id} Loss Curve")
    plt.legend()
    plt.savefig(f"{results_dir}/{model_name}_c{combo_id}_loss.png")
    plt.close()

    # Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['epoch'], history_df['train_accuracy'], label="Train Accuracy", color='blue')
    plt.plot(history_df['epoch'], history_df['val_accuracy'], label="Validation Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Combo {combo_id} Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f"{results_dir}/{model_name}_c{combo_id}_acc.png")
    plt.close()

# -----------------------------------------------------------------------------
# Section 7: Main Function to Run All Models with All Hyperparameter Combinations
# -----------------------------------------------------------------------------
def main():
    # Make sure results directory exists
    os.makedirs("results-1", exist_ok=True)
    
    summary = []
    
    # === TRAINING LOOP ===
    for model_name, model in models.items():
        for combo_id, combo in enumerate(combos, 1):
            print(f"🔥 Training {model_name} | Combo {combo_id}")
            model.to(device)
            trained_model_path, history = train_and_validate(
                model, model_name, combo_id, combo, dataloaders, device, class_names
            )
            plot_training_curves(history, model_name, combo_id)

    # === EVALUATION LOOP ===
    for model_name, model in models.items():
        print(f"🧠 Evaluating {model_name}")
        for combo_id, combo in enumerate(combos, 1):
            results_dir = f"results-1/combo_{combo_id}"
            model_path = os.path.join(results_dir, f"{model_name}_c{combo_id}.pt")
            if not os.path.exists(model_path):
                print(f"⚠️ Skipping {model_name} (Combo {combo_id}) — no saved model found.")
                continue
            metrics = evaluate_and_visualize(
                model, model_name, combo_id, model_path, dataloaders, class_names, device
            )
            summary.append({
                "Model": model_name,
                "InputSize": 224,
                "Combo": combo_id,
                **metrics
            })

    # Save the results summary
    metrics_df = pd.DataFrame(summary)
    metrics_df.to_csv("results-1/metrics_224x224_summary.csv", index=False)
    print("✅ Training and Evaluation completed for all 224x224 models.")


if __name__ == "__main__":
    main()
