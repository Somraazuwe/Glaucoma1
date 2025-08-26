import kagglehub

# Download latest version
path = kagglehub.dataset_download("deathtrooper/multichannel-glaucoma-benchmark-dataset")

print("Path to dataset files:", path)

# === IMPORTS AND SETUP ===
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
import warnings
import seaborn as sns
ignore_warnings = warnings.filterwarnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# === UPDATED DATASET PATH ===
root_dir = "/kaggle/input/multichannel-glaucoma-benchmark-dataset"

# Load and validate metadata
metadata = pd.read_csv(os.path.join(root_dir, "metadata - standardized.csv"))

# Filter for glaucoma classification (keep only 0 and 1, exclude -1 which is glaucoma suspect)
metadata = metadata[metadata['types'].isin([0, 1])].reset_index(drop=True)
print(f"✅ Total samples for glaucoma classification: {len(metadata)}")
print(f"   Non-glaucoma (0): {len(metadata[metadata['types'] == 0])}")
print(f"   Glaucoma (1): {len(metadata[metadata['types'] == 1])}")

def apply_clahe_and_median(image_np):
    """
    Apply CLAHE + median filter to RGB image (numpy array) for better contrast and noise reduction.
    """
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    merged_lab = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    # Median filter to reduce noise while preserving edges
    denoised_rgb = cv2.medianBlur(enhanced_rgb, ksize=3)

    return denoised_rgb

# === GLAUCOMA CLASSIFICATION DATASET CLASS ===
class GlaucomaClassificationDataset(Dataset):
    def __init__(self, metadata, root_dir, image_size=224, augment=False):
        self.metadata = metadata
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment

        # Data augmentation for classification
        self.aug_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-20, 20), p=0.5),
            A.GaussianBlur(blur_limit=(1, 3), p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        self.simple_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        fundus_path = os.path.join(self.root_dir, "full-fundus" + row['fundus'])

        # Load and preprocess image
        raw_image = np.array(Image.open(fundus_path).convert("RGB"))
        fundus = apply_clahe_and_median(raw_image)

        # Get glaucoma label from 'types' column (0=non-glaucoma, 1=glaucoma)
        label = int(row['types'])

        if self.augment:
            transformed = self.aug_transform(image=fundus)
            fundus = transformed['image']
        else:
            fundus = self.simple_transform(Image.fromarray(fundus))

        return fundus, torch.tensor(label, dtype=torch.long)

# === GLAUCOMA CLASSIFIER MODEL ===
class GlaucomaClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3, pretrained=True):
        super().__init__()

        # Load pre-trained ResNet-50 as feature extractor
        self.feature_extractor = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)

        # Remove the final classification layers
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate/2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)

        # Global pooling
        pooled = self.global_pool(features)

        # Flatten
        flattened = torch.flatten(pooled, 1)

        # Classification
        output = self.classifier(flattened)

        return output

# === TRAINING FUNCTION FOR GLAUCOMA CLASSIFICATION ===
def train_glaucoma_classifier(model, train_loader, val_loader, epochs=5, patience=7):
    """
    Training loop for glaucoma classification
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [], 'precision': [], 'recall': [], 'lr': []
    }

    best_val_acc = 0.0
    patience_counter = 0

    model.to(device)
    print(f"🚀 Starting glaucoma classification training for {epochs} epochs")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for img, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            img, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate training metrics
        train_acc = 100 * correct / total
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for img, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                img, labels = img.to(device), labels.to(device)
                outputs = model(img)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calculate validation metrics
        val_acc = 100 * val_correct / val_total
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='binary', zero_division=0
        )

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)

        # Store history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['precision'].append(val_precision)
        history['recall'].append(val_recall)
        history['lr'].append(current_lr)

        # Print metrics
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"         | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        print(f"         | Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f}")
        print(f"         | Val F1: {val_f1:.4f} | LR: {current_lr:.2e}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "glaucoma_classifier_best.pt")
            patience_counter = 0
            print("✅ New best model saved!")
        else:
            patience_counter += 1
            print(f"⏳ Early stopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("🛑 Early stopping triggered.")
                break

    return history

# === EVALUATE GLAUCOMA CLASSIFIER ===
def evaluate_glaucoma_classifier(model, test_loader):
    """
    Evaluate the glaucoma classifier on test set
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    print("🧪 Evaluating Glaucoma Classifier on test set...")

    with torch.no_grad():
        for img, labels in tqdm(test_loader):
            img, labels = img.to(device), labels.to(device)
            outputs = model(img)

            # Get predictions and probabilities
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, [p[1] for p in all_probs])
    roc_auc = auc(fpr, tpr)

    print("=" * 60)
    print("🎯 GLAUCOMA CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🔍 Recall: {recall:.4f}")
    print(f"⭐ F1 Score: {f1:.4f}")
    print(f"📈 AUC-ROC: {roc_auc:.4f}")
    print(f"📋 Confusion Matrix:\n{cm}")
    print("=" * 60)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ROC curve
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    axes[1].set_title('Confusion Matrix')

    plt.tight_layout()
    plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': roc_auc,
        'confusion_matrix': cm
    }

# === DATA PREPARATION ===
def validate_image_paths(df, root_dir):
    """Validate that all required files exist."""
    valid_rows = []
    print("🔍 Validating image paths...")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        fundus_path = os.path.join(root_dir, "full-fundus" + row['fundus'])

        if os.path.exists(fundus_path):
            valid_rows.append(i)

    return df.iloc[valid_rows].reset_index(drop=True)

# Validate dataset
metadata_filtered = validate_image_paths(metadata, root_dir)
print(f"✅ Total valid samples: {len(metadata_filtered)}")

# Create train/val/test splits
indices = np.arange(len(metadata_filtered))
np.random.seed(42)
np.random.shuffle(indices)

train_idx = indices[:-2000]  # 80% for training
val_idx = indices[-2000:-1000]  # 10% for validation
test_idx = indices[-1000:]  # 10% for testing

print(f"📊 Dataset split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# Create datasets with appropriate augmentation
train_set = GlaucomaClassificationDataset(metadata_filtered.iloc[train_idx], root_dir, image_size=224, augment=True)
val_set = GlaucomaClassificationDataset(metadata_filtered.iloc[val_idx], root_dir, image_size=224, augment=False)
test_set = GlaucomaClassificationDataset(metadata_filtered.iloc[test_idx], root_dir, image_size=224, augment=False)

# Create data loaders
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

# === TRAIN THE MODEL ===
print("🏗️ Initializing Glaucoma Classifier...")
glaucoma_model = GlaucomaClassifier(num_classes=2, dropout_rate=0.3, pretrained=True)

print("🎯 Starting glaucoma classification training...")
cls_history = train_glaucoma_classifier(glaucoma_model, train_loader, val_loader, epochs=5, patience=7)

# Load best model and evaluate
print("🔄 Loading best model for evaluation...")
glaucoma_model = GlaucomaClassifier(num_classes=2, dropout_rate=0.3, pretrained=True)
glaucoma_model.load_state_dict(torch.load("glaucoma_classifier_best.pt"))

# Evaluate on test set
cls_results = evaluate_glaucoma_classifier(glaucoma_model, test_loader)

# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0,0].plot(cls_history['train_loss'], label='Train Loss', color='blue', linewidth=2)
axes[0,0].plot(cls_history['val_loss'], label='Validation Loss', color='red', linewidth=2)
axes[0,0].set_title('Training and Validation Loss')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Accuracy
axes[0,1].plot(cls_history['train_acc'], label='Train Accuracy', color='green', linewidth=2)
axes[0,1].plot(cls_history['val_acc'], label='Validation Accuracy', color='orange', linewidth=2)
axes[0,1].set_title('Training and Validation Accuracy')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Accuracy (%)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# F1 Score
axes[1,0].plot(cls_history['train_f1'], label='Train F1', color='purple', linewidth=2)
axes[1,0].plot(cls_history['val_f1'], label='Validation F1', color='brown', linewidth=2)
axes[1,0].set_title('Training and Validation F1 Score')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('F1 Score')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Learning Rate
axes[1,1].plot(cls_history['lr'], color='black', linewidth=2)
axes[1,1].set_title('Learning Rate Schedule')
axes[1,1].set_xlabel('Epoch')
axes[1,1].set_ylabel('Learning Rate')
axes[1,1].set_yscale('log')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n🏆 Glaucoma Classification Completed Successfully!")
print(f"📁 Best model saved as: glaucoma_classifier_best.pt")
print(f"🎯 Final Test Performance:")
print(f"   📊 Accuracy: {cls_results['accuracy']:.4f}")
print(f"   🎯 Precision: {cls_results['precision']:.4f}")
print(f"   🔍 Recall: {cls_results['recall']:.4f}")
print(f"   ⭐ F1 Score: {cls_results['f1']:.4f}")
print(f"   📈 AUC-ROC: {cls_results['auc_roc']:.4f}")

# === INFERENCE FUNCTION ===
def predict_glaucoma(image_path, model, device):
    """
    Predict if an image has glaucoma
    """
    model.eval()

    # Load and preprocess image exactly like during training
    raw_image = np.array(Image.open(image_path).convert("RGB"))

    # Apply the same enhancement used during training
    enhanced_image = apply_clahe_and_median(raw_image)

    # Convert back to PIL Image for transformation
    pil_image = Image.fromarray(enhanced_image)

    # Use the same transform as validation (no augmentation)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        prediction = torch.argmax(output, 1).item()
        confidence = probabilities[0][prediction].item()

    return prediction, confidence

# === TEST THE INFERENCE FUNCTION ===
def test_inference():
    """
    Test the inference function with a sample image from the dataset
    """
    # Find a sample image from the test set
    sample_row = metadata_filtered.iloc[test_idx[0]]
    sample_image_path = os.path.join(root_dir, "full-fundus" + sample_row['fundus'])
    true_label = sample_row['types']

    print(f"Testing inference on: {sample_image_path}")
    print(f"True label: {true_label} ({'Glaucoma' if true_label == 1 else 'Non-Glaucoma'})")

    # Make prediction
    prediction, confidence = predict_glaucoma(sample_image_path, glaucoma_model, device)

    print(f"Prediction: {prediction} ({'Glaucoma' if prediction == 1 else 'Non-Glaucoma'})")
    print(f"Confidence: {confidence:.2%}")

    # Display the image
    img = Image.open(sample_image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"True: {'Glaucoma' if true_label == 1 else 'Non-Glaucoma'} | "
              f"Pred: {'Glaucoma' if prediction == 1 else 'Non-Glaucoma'} ({confidence:.2%})")
    plt.axis('off')
    plt.show()

    return prediction == true_label

# Test the inference
test_inference()

# === BATCH INFERENCE FUNCTION ===
def batch_predict_glaucoma(image_paths, model, device, batch_size=16):
    """
    Predict glaucoma for multiple images in batches
    """
    model.eval()
    predictions = []
    confidences = []

    # Create dataset and dataloader for batch processing
    class InferenceDataset(Dataset):
        def __init__(self, image_paths):
            self.image_paths = image_paths
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            raw_image = np.array(Image.open(image_path).convert("RGB"))
            enhanced_image = apply_clahe_and_median(raw_image)
            pil_image = Image.fromarray(enhanced_image)
            return self.transform(pil_image), image_path

    dataset = InferenceDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    with torch.no_grad():
        for batch, paths in tqdm(dataloader, desc="Processing images"):
            batch = batch.to(device)
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(batch)):
                predictions.append(preds[i].item())
                confidences.append(probs[i][preds[i]].item())

    return predictions, confidences
