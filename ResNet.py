import torchvision.models as models
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Res_Net(nn.Module):
    def __init__(self, num_species, dropout_rate=0.5):
        super(Res_Net, self).__init__()
        self.backbone = models.resnet34(pretrained=True)

        for p in list(self.backbone.parameters())[:-8]:
            p.requires_grad = False
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_species),
            nn.Sigmoid()  
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x



