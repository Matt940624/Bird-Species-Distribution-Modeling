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

# Constants
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5

class BirdImageDataset(torch.utils.data.Dataset):
    """Dataset for bird habitat prediction"""
    def __init__(self, images, labels, transform=None):
        """
        Initialize the dataset
        
        Parameters:
        -----------
        images : numpy.ndarray
            Array of images (num_samples, channels, height, width)
        labels : torch.Tensor
            Tensor of binary labels (num_samples, num_species)
        transform : callable, optional
            Optional transform to be applied on images
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and convert to tensor
        image = self.images[idx]
        
        # Convert to torch tensor if it's numpy array
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.copy()).float()
        
        # Apply transforms if available
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name="model"):
    """Train the model"""
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
        # Training phase
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
        
        # Validation phase
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
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping - only save the best model
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
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Save only the final model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), f"models/{model_name}_final.pth")
    print(f"Model saved as 'models/{model_name}_final.pth'")
    
    return history

def calculate_metrics(model, data_loader, species_list, device):
    """Calculate performance metrics for the model"""
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
    
    # Calculate metrics
    metrics = {}
    for i, species in enumerate(species_list):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]
        y_prob = all_outputs[:, i]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle cases where a class might not be present
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
        
        # Confusion matrix elements
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
    """Prepare data for training and validation"""
    # Ensure images are in the correct format
    print(f"Image data shape: {images.shape}")
    
    # Convert labels to torch tensors if they're not already
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.float32)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=test_size, random_state=42,
        stratify=np.argmax(labels, axis=1) if labels.shape[1] > 1 else None
    )
    
    # Define transformations - simpler for faster training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BirdImageDataset(X_train, y_train, transform=train_transform)
    val_dataset = BirdImageDataset(X_val, y_val, transform=val_transform)
    
    # Create data loaders
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
    """Plot training history"""
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
    """
    Create a weighted BCE loss that balances classes to approximate a 50/50 split
    """
    num_samples = len(labels)
    num_species = labels.shape[1]
    
    # Calculate class weights for each species
    pos_weights = torch.zeros(num_species, device=device)
    
    for i in range(num_species):
        # Count positive samples for this species
        pos_count = labels[:, i].sum()
        pos_ratio = pos_count / num_samples
        
        # Calculate weight to balance to 50/50
        if pos_ratio > 0:
            pos_weights[i] = (1 - pos_ratio) / max(pos_ratio, 1e-8)
        else:
            # Special case for species with no positive examples
            pos_weights[i] = 10.0
    
    print("Class weights for balanced loss function:")
    for i in range(num_species):
        print(f"  Species {i}: {pos_weights[i].item():.4f}")
    
    # Create the loss function with calculated weights
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
        # Use BCE with logits as the base loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate pt (probability of the truth)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Calculate focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Calculate loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_models(train_loader, val_loader, species_list, device):
    """Train both CNN and ResNet models"""
    num_species = len(species_list)
    
    # Get labels from train_loader for calculating weights
    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    
    models_to_train = [
        ("CNN", Conv_Net(num_species)),
        # ("ResNet", Res_Net(num_species))
    ]
    
    all_histories = {}
    
    for model_name, model in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_name} model...")
        print(f"{'='*50}")
        
        # Use one of the following loss functions:
        
        # OPTION 1: Weighted BCE Loss
        criterion = nn.BCELoss()
        
        # OPTION 2: Focal Loss 
        # criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        optimizer = optim.Adam(
            model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY
        )
        
        # Train model
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
        
        # Plot training history
        plot_training_history(history, model_name)
        
        # Calculate metrics on validation data
        print(f"\nCalculating metrics for {model_name}...")
        metrics = calculate_metrics(model, val_loader, species_list, device)
        
        # Print summary of metrics
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
    
    # Overall statistics
    total_samples = len(labels)
    species_counts = labels.sum(axis=0)
    
    print(f"Total samples: {total_samples}")
    
    # Per-species analysis
    for i, species in enumerate(species_list):
        present = int(species_counts[i])
        absent = total_samples - present
        percent = (present / total_samples) * 100
        
        print(f"{species}: Present in {present}/{total_samples} samples ({percent:.2f}%)")
        print(f"  Class ratio: 1:{absent/max(present, 1):.1f} (positive:negative)")
    
    # Summary of imbalance
    avg_presence = species_counts.mean()
    print(f"\nAverage presence per species: {avg_presence:.2f} samples")
    print(f"Most common species: {species_list[np.argmax(species_counts)]} ({int(species_counts.max())} samples)")
    print(f"Rarest species: {species_list[np.argmin(species_counts)]} ({int(species_counts.min())} samples)")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example species list
    species_list = ["amecro", "easblu", "pilwoo", "blujay"]
    
    try:
        # Load the dataset
        from data import generate_dataset
        
        # Define locations and date ranges
        locations = [
            # Locations good for Pileated Woodpecker (mature forests with large trees)
            [-73.2, 44.5, -72.7, 45.0],  # Northern Vermont forests
            [-68.9, 45.5, -68.4, 46.0],  # Northern Maine (mature woods)
            [-82.5, 35.5, -82.0, 36.0],  # Smoky Mountains (old growth forest)
            
            # Locations good for Eastern Bluebird (open meadows, farmland)
            [-71.8, 42.5, -71.3, 43.0],  # New Hampshire farmland
            [-76.5, 40.8, -76.0, 41.3],  # Pennsylvania agricultural areas
            [-83.6, 32.8, -83.1, 33.3],  # Central Georgia farmland
            
            # Locations good for American Crow (widespread, adaptable)
            [-74.0, 40.7, -73.5, 41.2],  # NYC/Long Island suburban areas
            [-87.9, 41.8, -87.4, 42.3],  # Chicago area (urban/suburban mix)
            [-122.4, 47.5, -121.9, 48.0],  # Seattle area (urban/suburban)
            
            # Locations good for Blue Jay (deciduous forests, suburban areas)
            [-84.5, 39.0, -84.0, 39.5],  # Southern Ohio woodlands
            [-70.8, 42.2, -70.3, 42.7],  # Eastern Massachusetts mixed woods
            [-88.2, 43.0, -87.7, 43.5],  # Southern Wisconsin oak forests
            
            # Locations for Northern Cardinal (woodland edges, shrubby areas)
            [-86.2, 39.7, -85.7, 40.2],  # Central Indiana thickets
            [-95.4, 29.7, -94.9, 30.2],  # Houston area (southern range)
            [-77.2, 38.8, -76.7, 39.3],  # DC/Maryland suburban areas
            
            # Unsuitable for Pileated Woodpecker (open areas without large trees)
            [-99.2, 41.5, -98.7, 42.0],  # Nebraska grasslands
            [-104.8, 40.5, -104.3, 41.0],  # Eastern Colorado plains
            [-116.5, 43.5, -116.0, 44.0],  # Idaho sagebrush plains
            
            # Unsuitable for Eastern Bluebird (dense forests)
            [-69.1, 45.2, -68.6, 45.7],  # Dense Maine forests
            [-73.9, 42.7, -73.4, 43.2],  # Adirondack forest
            [-123.8, 47.8, -123.3, 48.3],  # Olympic Peninsula rainforest
            
            # Unsuitable for Northern Cardinal (high elevations, northern areas)
            [-107.8, 37.5, -107.3, 38.0],  # Colorado Rockies (high elevation)
            [-110.8, 43.7, -110.3, 44.2],  # Wyoming mountains
            [-71.3, 44.2, -70.8, 44.7],  # White Mountains NH (northern edge)
            
            # Seasonal variations (to capture migration patterns)
            [-80.3, 25.7, -79.8, 26.2],  # South Florida (winter grounds)
            [-93.3, 45.0, -92.8, 45.5],  # Minnesota (summer breeding)
            [-118.4, 34.0, -117.9, 34.5],  # Southern California (edge of range)
        ]
        
        date_ranges = [
            ("2023-01-01", "2023-01-31"),
            ("2023-02-01", "2023-02-28"),
            ("2023-03-01", "2023-03-31"),
            ("2023-04-01", "2023-04-30"),
            ("2023-05-01", "2023-05-31"),
            # ("2023-06-01", "2023-06-30"),
            # ("2023-07-01", "2023-07-31"),
            # ("2023-08-01", "2023-08-31"),
            # ("2023-09-01", "2023-09-30"),
            # ("2023-10-01", "2023-10-31"),
            # ("2023-11-01", "2023-11-30"),
            # ("2023-12-01", "2023-12-31"),
            # ("2024-01-01", "2024-01-31"),
            # ("2024-02-01", "2024-02-29"),
            # ("2024-03-01", "2024-03-31"),
            # ("2024-04-01", "2024-04-30"),
            # ("2024-05-01", "2024-05-31"),
            # ("2024-06-01", "2024-06-30"),
            # ("2024-07-01", "2024-07-31"),
            # ("2024-08-01", "2024-08-31"),
            # ("2024-09-01", "2024-09-30"),
            # ("2024-10-01", "2024-10-31"),
            # ("2024-11-01", "2024-11-30"),
            # ("2024-12-01", "2024-12-31"),
            # ("2025-01-01", "2025-01-31"),
            # ("2025-02-01", "2025-02-28"),
            # ("2025-03-01", "2025-03-31"),
        ]
        
        images, labels, metadata = generate_dataset(locations, date_ranges, species_list)

        # After loading or generating dataset
        analyze_dataset_distribution(labels, species_list)
        
        if len(images) == 0:
            raise ValueError("No data was generated")
            
    except Exception as e:
        print(f"Error generating real dataset: {e}")
        print("Creating dummy dataset instead")
        
        # Create dummy data
        num_samples = 100
        image_size = 224
        images = np.random.rand(num_samples, 3, image_size, image_size).astype(np.float32)
        labels = np.random.randint(0, 2, size=(num_samples, len(species_list))).astype(np.float32)
    
    # Prepare data
    train_loader, val_loader = prepare_data(images, labels)
    
    # Train models
    train_models(train_loader, val_loader, species_list, device)

if __name__ == "__main__":
    main()