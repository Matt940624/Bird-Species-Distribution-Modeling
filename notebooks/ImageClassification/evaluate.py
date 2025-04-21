import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import pandas as pd
import cv2
from torch.utils.data import DataLoader


from CustomCNN import Conv_Net
from ResNet import Res_Net
from data import generate_dataset
from training import BirdImageDataset  

def load_model(model_path, model_type, num_species, device):
    """Load a trained model"""
    if model_type == 'CNN':
        model = Conv_Net(num_species=num_species)
    else:  
        model = Res_Net(num_species=num_species)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, data_loader, species_list, device):
    all_targets = []
    all_outputs = []
    all_preds = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating model"):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
            all_preds.append((outputs > 0.5).cpu().numpy())
    
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    
    metrics = {}
    for i, species in enumerate(species_list):
        
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]
        y_score = all_outputs[:, i]
        
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
        except:
            
            roc_auc = 0.5
            fpr, tpr = [0, 1], [0, 1]
        
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        
        accuracy = accuracy_score(y_true, y_pred)
        
        
        metrics[species] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'roc': {
                'fpr': fpr,
                'tpr': tpr
            },
            'counts': {
                'tn': tn, 
                'fp': fp, 
                'fn': fn, 
                'tp': tp
            }
        }
    
    return metrics

def plot_confusion_matrices(metrics, species_list, species_names, model_name, output_dir):
    
    n_species = len(species_list)
    fig, axes = plt.subplots(1, n_species, figsize=(n_species * 5, 5))
    
    if n_species == 1:
        axes = [axes]  
        
    for i, species in enumerate(species_list):
        ax = axes[i]
        cm = metrics[species]['confusion_matrix']
        
        
        cm_sum = np.sum(cm)
        if cm_sum > 0:  
            cm_normalized = cm.astype('float') / cm_sum
        else:
            cm_normalized = cm.astype('float')
        
        
        sns.heatmap(cm, annot=False, cmap='Blues', ax=ax,
                    xticklabels=['Pred Neg', 'Pred Pos'],
                    yticklabels=['Act Neg', 'Act Pos'])
        
        
        colormap = plt.cm.Blues
        
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                
                
                bg_intensity = cm_normalized[i, j]
                
                
                bg_color = colormap(bg_intensity)
                
                
                
                luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                
                
                
                
                text_color = 'black' if luminance > 0.6 else 'white'
                
                
                text = f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})"
                
                
                ax.text(j + 0.5, i + 0.5, text,
                        ha='center', va='center', 
                        color=text_color,
                        fontsize=10)
                
        
        ax.set_title(f'{model_name} - {species_names.get(species, species)}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrices.png'), dpi=300)
    plt.close()

def plot_roc_curves(metrics, species_list, species_names, model_name, output_dir):
    plt.figure(figsize=(10, 8))
    
    for species in species_list:
        fpr = metrics[species]['roc']['fpr']
        tpr = metrics[species]['roc']['tpr']
        roc_auc = metrics[species]['auc']
        
        plt.plot(fpr, tpr, lw=2, label=f'{species_names.get(species, species)} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_roc_curves.png'), dpi=300)
    plt.close()

def plot_metrics_comparison(cnn_metrics, resnet_metrics, species_list, species_names, output_dir):
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 3 * len(metrics_to_plot)))
    
    x = np.arange(len(species_list))
    width = 0.35
    
    for i, metric in enumerate(metrics_to_plot):
        cnn_values = [cnn_metrics[species][metric] for species in species_list]
        resnet_values = [resnet_metrics[species][metric] for species in species_list]
        
        ax = axes[i]
        ax.bar(x - width/2, cnn_values, width, label='CNN')
        ax.bar(x + width/2, resnet_values, width, label='ResNet')
        
        ax.set_ylabel(metric.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels([species_names.get(species, species) for species in species_list])
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()

def compute_feature_importance(model, sample_images, species_list, device):
    model.eval()
    importance_maps = {}
    
    for species_idx, species in enumerate(species_list):
        importance_maps[species] = []
        
        for img_idx, image in enumerate(sample_images):
            
            tensor_image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
            tensor_image.requires_grad_()
            
            
            model.zero_grad()
            output = model(tensor_image)
            
            
            output[0, species_idx].backward()
            
            
            gradients = tensor_image.grad.abs().cpu().numpy()[0]
            
            
            importance = np.sum(gradients, axis=0)
            
            
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
            
            importance_maps[species].append({
                'image': image,
                'importance': importance
            })
    
    return importance_maps

def plot_feature_importance(importance_maps, species_list, species_names, model_name, output_dir, num_samples=3):
    n_species = len(species_list)
    n_samples = min(num_samples, len(importance_maps[species_list[0]]))
    
    plt.figure(figsize=(n_species * 4, n_samples * 3))
    
    for i, species in enumerate(species_list):
        for j in range(n_samples):
            
            sample = importance_maps[species][j]
            image = sample['image']
            importance = sample['importance']
            
            
            img_display = np.transpose(image, (1, 2, 0))
            img_display = np.clip(img_display, 0, 1)
            
            
            plt.subplot(n_samples, n_species, j * n_species + i + 1)
            plt.imshow(img_display)
            plt.imshow(importance, cmap='hot', alpha=0.5)
            
            if j == 0:
                plt.title(species_names.get(species, species))
            
            plt.axis('off')
    
    plt.suptitle(f'{model_name} Feature Importance', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(output_dir, f'{model_name}_feature_importance.png'), dpi=300)
    plt.close()

def create_metrics_table(cnn_metrics, resnet_metrics, species_list, species_names):
    data = []
    
    
    for species in species_list:
        species_name = species_names.get(species, species)
        
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            cnn_value = cnn_metrics[species][metric]
            resnet_value = resnet_metrics[species][metric]
            
            data.append({
                'Species': species_name,
                'Metric': metric.capitalize(),
                'CNN': f"{cnn_value:.4f}",
                'ResNet': f"{resnet_value:.4f}",
                'Difference': f"{resnet_value - cnn_value:+.4f}"
            })
    
    
    df = pd.DataFrame(data)
    
    return df

def main():
    
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    species_list = ["amecro", "easblu", "pilwoo", "blujay"]
    species_names = {
        "amecro": "American Crow",
        "easblu": "Eastern Bluebird",
        "pilwoo": "Pileated Woodpecker",
        "blujay": "Blue Jay"
    }
    
    
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
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
        ("2025-01-01", "2025-01-31"),
        ("2025-02-01", "2025-02-28"),
        ("2025-03-01", "2025-03-31"),
    ]
    
    try:
        print("Loading dataset...")
        
        
        images, labels, metadata = generate_dataset(locations, date_ranges, species_list)
        
        
        test_dataset = BirdImageDataset(images, labels, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        
        print("Creating dummy test dataset instead")
        num_samples = 50
        image_size = 224
        images = np.random.rand(num_samples, 3, image_size, image_size).astype(np.float32)
        labels = np.random.randint(0, 2, size=(num_samples, len(species_list))).astype(np.float32)
        
        test_dataset = BirdImageDataset(images, labels, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    
    models = {
        'CNN': load_model('models/CNN_final.pth', 'CNN', len(species_list), device),
        'ResNet': load_model('models/ResNet_final.pth', 'ResNet', len(species_list), device)
    }
    
    
    print("Evaluating models...")
    metrics = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        metrics[model_name] = evaluate_model(model, test_loader, species_list, device)
        
        
        plot_confusion_matrices(
            metrics[model_name], 
            species_list, 
            species_names, 
            model_name, 
            output_dir
        )
        
        
        plot_roc_curves(
            metrics[model_name], 
            species_list, 
            species_names, 
            model_name, 
            output_dir
        )
    
    
    plot_metrics_comparison(
        metrics['CNN'], 
        metrics['ResNet'], 
        species_list, 
        species_names, 
        output_dir
    )
    
    
    metrics_table = create_metrics_table(
        metrics['CNN'], 
        metrics['ResNet'], 
        species_list, 
        species_names
    )
    
    
    metrics_table.to_csv(os.path.join(output_dir, 'metrics_comparison.csv'), index=False)
    print(f"Metrics table saved to {os.path.join(output_dir, 'metrics_comparison.csv')}")
    
    
    print("\nModel Performance Comparison:")
    print(metrics_table.to_string(index=False))
    
    
    print("\nComputing feature importance...")
    
    
    sample_indices = np.random.choice(len(images), min(5, len(images)), replace=False)
    sample_images = images[sample_indices]
    
    importance_maps = {}
    for model_name, model in models.items():
        print(f"Computing feature importance for {model_name}...")
        importance_maps[model_name] = compute_feature_importance(
            model, 
            sample_images, 
            species_list, 
            device
        )
        
        
        plot_feature_importance(
            importance_maps[model_name], 
            species_list, 
            species_names, 
            model_name, 
            output_dir
        )
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")

if __name__ == "__main__":
    
    import torchvision.transforms as transforms
    main()