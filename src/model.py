import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalModel(nn.Module):
    def __init__(self, num_tabular_features=12):
        super(MultimodalModel, self).__init__()
        
        
        self.image_branch = models.resnet18(weights=None)
        self.image_branch.fc = nn.Identity()

        
        self.tabular_branch = nn.Sequential(
            nn.Linear(num_tabular_features, 256),
            nn.BatchNorm1d(256),                
            nn.ReLU(),                            
            nn.Dropout(0.3),                     
            nn.Linear(256, 128),                 
            nn.BatchNorm1d(128),                
            nn.ReLU(),                            
            nn.Linear(128, 64),                 
            nn.ReLU()                             
        )

     
        self.final_fc = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, image, tabular):
        img_feat = self.image_branch(image)
        tab_feat = self.tabular_branch(tabular)
        combined = torch.cat((img_feat, tab_feat), dim=1)
        return self.final_fc(combined)