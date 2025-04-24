# Lung X-Ray Classification and Analysis

This repository contains code for classifying lung X-ray images using deep learning models, along with scripts for preprocessing, evaluation, and visualization.

## Project Structure

- `classification-224.py`, `classification-299.py`, `classification-331.py`, etc.: Model training scripts for different input sizes.
- `plot_all_model_curves.py`, `plot_all_model_roc_auc.py`: Scripts for plotting model performance metrics and ROC/AUC curves.
- `Preprocessing.ipynb`: Data preprocessing and exploration notebook.
- `results-1/`, `results-2/`, `224-results-1/`, `224-results-2/`, `331-results-1/`, `331-results-2/`: Output folders containing model results and plots.
- `Chest_X-Ray_Image/`, `Data/`, `Dataset-1/`, `Dataset-2/`: Datasets for training and testing.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- timm
- efficientnet_pytorch
- scikit-learn
- matplotlib
- numpy

Install dependencies with:
```sh
pip install torch torchvision timm efficientnet_pytorch scikit-learn matplotlib numpy
```

## Usage

1. Prepare your datasets in the specified folders.
2. Train models using the provided classification scripts.
3. Use the plotting scripts to generate performance curves and ROC/AUC plots.

Example:
```sh
python classification-224.py
python plot_all_model_roc_auc.py
```

## Results

All results, including accuracy, loss, and ROC/AUC plots, are saved in the results folders.

## License

This project is for academic and research purposes only.
