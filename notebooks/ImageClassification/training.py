import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
import copy

from CustomCNN import Conv_Net
from ResNet import Res_Net


BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5

class BirdImageDataset(torch.utils.data.Dataset):

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        image = self.images[idx]
        
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.copy()).float()
        
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name="model"):
    model.to(device)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model = None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [{model_name}-Train]") as t:
            for inputs, targets in t:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item() * inputs.size(0)
                train_loss += batch_loss
                
                pred_binary = (outputs > 0.5).float()
                
                correct = (pred_binary == targets).sum().item()
                total = targets.numel()
                train_correct += correct
                train_total += total
                
                t.set_postfix(loss=batch_loss/inputs.size(0), acc=correct/total)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [{model_name}-Val]") as t:
                for inputs, targets in t:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    batch_loss = loss.item() * inputs.size(0)
                    val_loss += batch_loss
                    
                    pred_binary = (outputs > 0.5).float()
                    
                    correct = (pred_binary == targets).sum().item()
                    total = targets.numel()
                    val_correct += correct
                    val_total += total
                    
                    all_targets.append(targets.cpu().numpy())
                    all_predictions.append(pred_binary.cpu().numpy())
                    
                    t.set_postfix(loss=batch_loss/inputs.size(0), acc=correct/total)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            best_model = copy.deepcopy(model.state_dict())
            print(f"New best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs (no improvement for {patience} epochs)")
                break
    
    
    if best_model is not None:
        model.load_state_dict(best_model)
    
    
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), f"models/{model_name}_final.pth")
    print(f"Model saved as 'models/{model_name}_final.pth'")
    
    return history

def calculate_metrics(model, data_loader, species_list, device):
    model.eval()
    
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating model"):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_preds = (all_outputs > 0.5).astype(int)
    
    
    metrics = {}
    for i, species in enumerate(species_list):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]
        y_prob = all_outputs[:, i]
        
        
        accuracy = accuracy_score(y_true, y_pred)
        
        
        if len(np.unique(y_true)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            try:
                auc = roc_auc_score(y_true, y_prob)
            except:
                auc = 0.5
        else:
            precision = recall = f1 = 0.0
            auc = 0.5
        
        
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        
        metrics[species] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': np.array([[tn, fp], [fn, tp]])
        }
    
    return metrics

def prepare_data(images, labels, test_size=0.2):
    
    print(f"Image data shape: {images.shape}")
    
    
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.float32)
    
    
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=test_size, random_state=42,
        stratify=np.argmax(labels, axis=1) if labels.shape[1] > 1 else None
    )
    
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    
    train_dataset = BirdImageDataset(X_train, y_train, transform=train_transform)
    val_dataset = BirdImageDataset(X_val, y_val, transform=val_transform)
    
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    return train_loader, val_loader

def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png', dpi=300)
    plt.close()

def create_balanced_loss_function(labels, device):

    num_samples = len(labels)
    num_species = labels.shape[1]
    
    
    pos_weights = torch.zeros(num_species, device=device)
    
    for i in range(num_species):
        
        pos_count = labels[:, i].sum()
        pos_ratio = pos_count / num_samples
        
        
        if pos_ratio > 0:
            pos_weights[i] = (1 - pos_ratio) / max(pos_ratio, 1e-8)
        else:
            
            pos_weights[i] = 10.0
    
    print("Class weights for balanced loss function:")
    for i in range(num_species):
        print(f"  Species {i}: {pos_weights[i].item():.4f}")
    
    
    return nn.BCEWithLogitsLoss(pos_weight=pos_weights)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for multi-label classification
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        
        probs = torch.sigmoid(inputs)
        
        
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        
        focal_weight = (1 - pt) ** self.gamma
        
        
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        
        focal_loss = alpha_t * focal_weight * bce_loss
        
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_models(train_loader, val_loader, species_list, device):
    num_species = len(species_list)
    
    
    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    
    models_to_train = [
        ("CNN", Conv_Net(num_species)),
        ("ResNet", Res_Net(num_species))
    ]
    
    all_histories = {}
    
    for model_name, model in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_name} model...")
        print(f"{'='*50}")
        
        
        
        
        criterion = nn.BCELoss()
        
        
        
        
        optimizer = optim.Adam(
            model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY
        )
        
        
        history = train_model(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            EPOCHS, 
            device, 
            model_name=model_name
        )
        all_histories[model_name] = history
        
        
        plot_training_history(history, model_name)
        
        
        print(f"\nCalculating metrics for {model_name}...")
        metrics = calculate_metrics(model, val_loader, species_list, device)
        
        
        print(f"\n{model_name} Metrics Summary:")
        print("-" * 40)
        for species in species_list:
            m = metrics[species]
            print(f"{species}: Acc={m['accuracy']:.4f}, Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, F1={m['f1']:.4f}")
    
    print("\nTraining complete!")
    return all_histories

def analyze_dataset_distribution(labels, species_list):
    """Analyze the distribution of species in the dataset"""
    print("\nDataset Distribution Analysis:")
    print("-" * 40)
    
    
    total_samples = len(labels)
    species_counts = labels.sum(axis=0)
    
    print(f"Total samples: {total_samples}")
    
    
    for i, species in enumerate(species_list):
        present = int(species_counts[i])
        absent = total_samples - present
        percent = (present / total_samples) * 100
        
        print(f"{species}: Present in {present}/{total_samples} samples ({percent:.2f}%)")
        print(f"  Class ratio: 1:{absent/max(present, 1):.1f} (positive:negative)")
    
    
    avg_presence = species_counts.mean()
    print(f"\nAverage presence per species: {avg_presence:.2f} samples")
    print(f"Most common species: {species_list[np.argmax(species_counts)]} ({int(species_counts.max())} samples)")
    print(f"Rarest species: {species_list[np.argmin(species_counts)]} ({int(species_counts.min())} samples)")

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    species_list = ["amecro", "easblu", "pilwoo", "blujay"]
    
    try:
        
        from data import generate_dataset
        
        
        locations = [
            
            [-73.2, 44.5, -72.7, 45.0],  
            [-68.9, 45.5, -68.4, 46.0],  
            [-82.5, 35.5, -82.0, 36.0],  
            
            
            [-71.8, 42.5, -71.3, 43.0],  
            [-76.5, 40.8, -76.0, 41.3],  
            [-83.6, 32.8, -83.1, 33.3],  
            
            
            [-74.0, 40.7, -73.5, 41.2],  
            [-87.9, 41.8, -87.4, 42.3],  
            [-122.4, 47.5, -121.9, 48.0],  
            
            
            [-84.5, 39.0, -84.0, 39.5],  
            [-70.8, 42.2, -70.3, 42.7],  
            [-88.2, 43.0, -87.7, 43.5],  
            
            
            [-86.2, 39.7, -85.7, 40.2],  
            [-95.4, 29.7, -94.9, 30.2],  
            [-77.2, 38.8, -76.7, 39.3],  
            
            
            [-99.2, 41.5, -98.7, 42.0],  
            [-104.8, 40.5, -104.3, 41.0],  
            [-116.5, 43.5, -116.0, 44.0],  
            
            
            [-69.1, 45.2, -68.6, 45.7],  
            [-73.9, 42.7, -73.4, 43.2],  
            [-123.8, 47.8, -123.3, 48.3],  
            
            
            [-107.8, 37.5, -107.3, 38.0],  
            [-110.8, 43.7, -110.3, 44.2],  
            [-71.3, 44.2, -70.8, 44.7],  
            
            
            [-80.3, 25.7, -79.8, 26.2],  
            [-93.3, 45.0, -92.8, 45.5],  
            [-118.4, 34.0, -117.9, 34.5],  
        ]
        
        date_ranges = [
            ("2023-01-01", "2023-01-31"),
            ("2023-02-01", "2023-02-28"),
            ("2023-03-01", "2023-03-31"),
            ("2023-04-01", "2023-04-30"),
            ("2023-05-01", "2023-05-31"),            
        ]
        
        images, labels, metadata = generate_dataset(locations, date_ranges, species_list)

        
        analyze_dataset_distribution(labels, species_list)
        
        if len(images) == 0:
            raise ValueError("No data was generated")
            
    except Exception as e:
        print(f"Error generating real dataset: {e}")
        print("Creating dummy dataset instead")
        
        
        num_samples = 100
        image_size = 224
        images = np.random.rand(num_samples, 3, image_size, image_size).astype(np.float32)
        labels = np.random.randint(0, 2, size=(num_samples, len(species_list))).astype(np.float32)
    
    
    train_loader, val_loader = prepare_data(images, labels)
    
    
    train_models(train_loader, val_loader, species_list, device)

if __name__ == "__main__":
    main()