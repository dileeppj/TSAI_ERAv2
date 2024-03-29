import torch
import torch.nn as nn
import torch.nn.functional as F


dropout_value = 0.1

'''
    Model Architecture => C1 [] => C2 [] => C3 [] => C4 [] =>
    Input Size == Output Size -> for a Block
'''
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Convolution Block 1   [C1 + C2 + C3]
        self.convBlock1 = nn.Sequential(
            # Convolution 1                     32x32x3 -> 32x32x8 -> RF 3
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # Convolution 2                     32x32x8 -> 32x32x32 -> RF 5
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # Convolution 3                     32x32x32 -> 30x30x32 -> RF 9  ( 5+(5-1)*1 = 9 )   => Dilation of 2 makes K=5
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )


        # Convolution Block 2   [C4 + C5 + C6]
        self.convBlock2 = nn.Sequential(
            # Convolution 4                     30x30x8 -> 30x30x8 -> RF 11
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # Convolution 5                     30x30x8 -> 30x30x8 -> RF 13
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # Convolution 6                     30x30x8 -> 15x15x8 -> RF 15 ( 13+(3-1)*1 = 15 ) Jout = 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )


        # Convolution Block 3   [C7 + C8 + C9]
        self.convBlock3 = nn.Sequential(
            # Convolution 7                     15x15x8 -> 15x15x8 -> RF 19 ( 15+(3-1)*2 = 19 )
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # Convolution 8                     15x15x8 -> 15x15x8 -> RF 23
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # Convolution 9                     15x15x8 -> 7x7x8 -> RF 27 ( 23+(3-1)*2 = 27 ) Jout = 4
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )


        # Convolution Block 4   [C10 + C11 + C12]
        self.convBlock4 = nn.Sequential(
            # # Convolution 10                     7x7x8 -> 7x7x8 -> RF 35 ( 27+(3-1)*4 = 35 )
            # nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            # nn.BatchNorm2d(8),
            # nn.ReLU(),
            # nn.Dropout(dropout_value),

            # Convolution 10                     7x7x8 -> 7x7x8 -> RF 35 ( 27+(3-1)*4 = 35 )    => Depthwise Separable Convolution
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, groups=16, bias=False),  # Depthwise Convolution
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),            # Pointwise Convolution
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            # Convolution 11                     7x7x8 -> 7x7x8 -> RF 43
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            # Convolution 12                     7x7x8 -> 7x7x8 -> RF 51
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )


        # GAP Layer [GAP]
        self.gap = nn.Sequential(
            # Global Average Pooling            7x7x64 -> 1x1x64 -> RF 51
            nn.AvgPool2d(kernel_size=7)
        )


        # Output Block [c13]
        self.outputBlock = nn.Sequential(
            # Convolution 13                    1x1x64 -> 1x1x10 -> RF 51
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )


    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = self.gap(x)
        x = self.outputBlock(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)