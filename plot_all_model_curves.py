import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing combo folders
RESULTS_DIR = 'results-1'
COMBO_PREFIX = 'combo_'

# Find all combo directories
combo_dirs = [os.path.join(RESULTS_DIR, d) for d in os.listdir(RESULTS_DIR) if d.startswith(COMBO_PREFIX)]
combo_dirs.sort()

# Find all models by scanning one combo dir for *_history.csv
model_names = set()
if combo_dirs:
    for fname in os.listdir(combo_dirs[0]):
        if fname.endswith('_history.csv'):
            model_names.add('_'.join(fname.split('_')[:-2]))  # removes _c1_history
model_names = sorted(model_names)

# For each model, gather all history files from all combos
for model in model_names:
    histories = []
    combo_labels = []
    for combo_dir in combo_dirs:
        combo_num = os.path.basename(combo_dir).split('_')[-1]
        pattern = os.path.join(combo_dir, f'{model}_c{combo_num}_history.csv')
        if os.path.exists(pattern):
            df = pd.read_csv(pattern)
            histories.append(df)
            combo_labels.append(f'Combo {combo_num}')
    if not histories:
        continue
    # Plot Loss
    plt.figure(figsize=(10,6))
    for df, label in zip(histories, combo_labels):
        plt.plot(df['epoch'], df['train_loss'], label=f'{label} Train')
        plt.plot(df['epoch'], df['val_loss'], label=f'{label} Val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model} - Training & Validation Loss (All Combos)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model}_all_combos_loss.png'))
    plt.close()
    # Plot Accuracy
    plt.figure(figsize=(10,6))
    for df, label in zip(histories, combo_labels):
        plt.plot(df['epoch'], df['train_accuracy'], label=f'{label} Train')
        plt.plot(df['epoch'], df['val_accuracy'], label=f'{label} Val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model} - Training & Validation Accuracy (All Combos)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model}_all_combos_accuracy.png'))
    plt.close()
print('✅ Combined plots saved for all models.')
