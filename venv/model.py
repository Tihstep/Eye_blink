from torch import nn

class Eye_CNN(nn.Module):
    def __init__(self):
        super(Eye_CNN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),             #24x24
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding= 0),                #22x22
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=0),                #20x20
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),   #10x10
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding= 0),              #8x8
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),   #4x4
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),          #4x4
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),          #4x4
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
           )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*512, 1024),
            nn.Dropout(p=0.5, inplace=False),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.Dropout(p=0.5, inplace=False),
            nn.ReLU(),
            nn.Linear(128, 12),
        )

    def forward(self, x):
        x = self.feature(x)
        coordinates = self.classifier(x)
        return coordinates