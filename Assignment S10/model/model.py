import torch.nn as nn
import torch.nn.functional as F

##### Model for Assignment S10 #####
class CustomResNet(nn.Module):
    
    dropout_value = 0.1
    
    def __init__(self):
        super().__init__()
        # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        
        # Layer1 - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        
        # Layer1 - R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
        self.layer1_resblock = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        # Layer2 - Conv 3x3 [256k], MaxPooling2D, BN, ReLU
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        # Layer3 - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        # Layer3 - R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        self.layer3_resblock = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        # MaxPooling with Kernel Size 4
        self.maxpool = nn.MaxPool2d(kernel_size=4)
        
        # FC Layer 
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        # PrepLayer
        x = self.prep_layer(x)
        
        # Layer 1
        x = self.layer1(x)
        r1 = self.layer1_resblock(x)
        x = x + r1
        
        # Layer 2
        x = self.layer2(x)
        
        # Layer 3
        x = self.layer3(x)
        r2 = self.layer3_resblock(x)
        x = x + r2
        
        # MaxPooling
        x = self.maxpool(x)
        
        # FC Layer 
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        
        # Softmax
        return F.log_softmax(x, dim=-1)
