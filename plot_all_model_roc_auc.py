import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torchvision import datasets, transforms, models
import timm
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
import torch.nn as nn

# 1. Define model dictionaries and modification utilities

def freeze_layers(model, classifier_attr=None):
    for name, param in model.named_parameters():
        param.requires_grad = False
    if classifier_attr is not None:
        head = getattr(model, classifier_attr)
        for param in head.parameters():
            param.requires_grad = True
    return model

def modify_model(name, base_model, num_classes=3, dropout_p=0.5):
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
    elif "inceptionv3" in name:
        in_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )
        if base_model.aux_logits:
            aux_in = base_model.AuxLogits.fc.in_features
            base_model.AuxLogits.fc = nn.Linear(aux_in, num_classes)
        model = freeze_layers(base_model, classifier_attr="fc")
    elif "efficientnet" in name:
        in_features = base_model._fc.in_features
        base_model._fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )
        model = freeze_layers(base_model, classifier_attr="_fc")
    elif "nasnet" in name:
        if hasattr(base_model, 'reset_classifier'):
            base_model.reset_classifier(num_classes=num_classes)
            for pname, param in base_model.named_parameters():
                param.requires_grad = False
            for param in base_model.get_classifier().parameters():
                param.requires_grad = True
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

def get_fresh_model(model_name):
    if model_name == "mobilenetv2":
        return models.mobilenet_v2(pretrained=True)
    if model_name in models_224:
        if model_name in ["res2net50", "darknet53"]:
            if model_name == "res2net50":
                return timm.create_model('res2net50_26w_4s', pretrained=True)
            if model_name == "darknet53":
                return timm.create_model('darknet53', pretrained=True)
        return getattr(models, model_name)(pretrained=True)
    if model_name in models_299:
        return models.inception_v3(pretrained=True, aux_logits=True)
    if model_name in models_331:
        if 'efficientnet' in model_name:
            from efficientnet_pytorch import EfficientNet
            return EfficientNet.from_name(model_name)
        if 'nasnet' in model_name:
            return timm.create_model('nasnetalarge', pretrained=True)
    raise ValueError(f"Unknown model: {model_name}")

# Model dictionaries
models_224 = {
    # "resnet34": models.resnet34(pretrained=True),
    # "resnet50": models.resnet50(pretrained=True),
    # "resnet101": models.resnet101(pretrained=True),
    # "densenet201": models.densenet201(pretrained=True),
    # "vgg16": models.vgg16(pretrained=True),
    # "vgg19": models.vgg19(pretrained=True),
    "mobilenetv2": models.mobilenet_v2(pretrained=True),
    "resnext50_32x4d": models.resnext50_32x4d(pretrained=True),
    "wide_resnet50_2": models.wide_resnet50_2(pretrained=True),
    "res2net50": timm.create_model('res2net50_26w_4s', pretrained=True),
    "darknet53": timm.create_model('darknet53', pretrained=True),
}
models_299 = {
    "inceptionv3": models.inception_v3(pretrained=True, aux_logits=True),
}
models_331 = {
    "efficientnet-b0": EfficientNet.from_name("efficientnet-b0"),
    "efficientnet-b1": EfficientNet.from_name("efficientnet-b1"),
    "efficientnet-b2": EfficientNet.from_name("efficientnet-b2"),
    "nasnetmobile": timm.create_model('nasnetalarge', pretrained=True),
}

# 2. Prepare test transforms for each size
transforms_dict = {
    224: transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]),
    299: transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ]),
    331: transforms.Compose([
        transforms.Resize((331, 331)),
        transforms.ToTensor()
    ]),
}

# 3. Model input size mapping
model_size_map = {}
for k in models_224: model_size_map[k] = 224
for k in models_299: model_size_map[k] = 299
for k in models_331: model_size_map[k] = 331

# 4. List all combos and models
RESULTS_DIR = 'results-2'
COMBO_PREFIX = 'combo_'
TEST_DIR = 'Dataset-2/test'
NUM_CLASSES = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

combo_dirs = [os.path.join(RESULTS_DIR, d) for d in os.listdir(RESULTS_DIR) if d.startswith(COMBO_PREFIX)]
combo_dirs.sort()

all_models = {**models_224, **models_299, **models_331}

# 5. Main ROC plotting loop
for model_name in all_models:
    input_size = model_size_map[model_name]
    test_transform = transforms_dict[input_size]
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    plt.figure(figsize=(10, 8))
    any_curve = False
    for combo_dir in combo_dirs:
        combo_num = os.path.basename(combo_dir).split('_')[-1]
        pt_path = os.path.join(combo_dir, f'{model_name}_c{combo_num}.pt')
        if not os.path.exists(pt_path):
            continue
        print(f"Testing {model_name} combo {combo_num}")
        # Re-instantiate and modify model
        base_model = get_fresh_model(model_name)
        model = modify_model(model_name, base_model, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())
        if not all_labels or not all_probs:
            print(f"No predictions for {model_name} combo {combo_num}")
            continue
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        for i in range(NUM_CLASSES):
            y_true = (all_labels == i).astype(int)
            y_score = all_probs[:, i]
            if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
                print(f"Warning: Cannot compute ROC for class {i} in combo {combo_num} (only one class present).")
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Combo {combo_num} Class {i} (AUC = {roc_auc:.2f})')
            any_curve = True
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {model_name} (All Combos)')
    if any_curve:
        plt.legend(loc='lower right', fontsize='small')
    else:
        print(f"No valid ROC curves plotted for {model_name}.")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_all_combos_roc_auc.png'))
    plt.close()
print('✅ ROC/AUC plots saved and displayed for all models.')
