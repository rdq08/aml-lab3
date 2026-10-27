
from torch import nn

# Define the custom neural network architecture
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64,
            kernel_size=3, padding=1, stride=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128,
            kernel_size=3, padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256,
            kernel_size=3, padding=1
        )

        # Pooling layers
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Classifier
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(in_features=256, out_features=200)

    def forward(self, x):
        # Input: B x 3 x 224 x 224

        x = self.maxpool(self.conv1(x).relu())
        # B x 64 x 112 x 112

        x = self.maxpool(self.conv2(x).relu())
        # B x 128 x 56 x 56

        x = self.maxpool(self.conv3(x).relu())
        # B x 256 x 28 x 28

        x = self.pool(x)
        # B x 256 x 1 x 1

        x = self.flatten(x)
        # B x 256

        x = self.fc1(x)
        # B x 200

        return x
