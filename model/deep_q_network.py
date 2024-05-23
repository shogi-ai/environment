"""
Deep Q-Network (DQN) model for processing input images and producing output
predictions. This model includes convolutional layers, batch normalization,
and fully connected layers, with an optional masking layer.
"""

from torch import nn

from model.mask_layer import MaskLayer


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model for processing input images and producing output
    predictions. This model includes convolutional layers, batch normalization,
    and fully connected layers, with an optional masking layer.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization layer after conv1.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization layer after conv2.
        conv3 (nn.Conv2d): Third convolutional layer.
        bn3 (nn.BatchNorm2d): Batch normalization layer after conv3.
        conv4 (nn.Conv2d): Fourth convolutional layer.
        bn4 (nn.BatchNorm2d): Batch normalization layer after conv4.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        mask (MaskLayer): Optional masking layer.
    """

    def __init__(self):
        """
        Initializes the DQN model with four convolutional layers, batch
        normalization, two fully connected layers, and an optional MaskLayer.
        """
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(14, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 9 * 9, 128 * 81)
        self.fc2 = nn.Linear(128 * 81, 81 * 81)
        self.mask = MaskLayer()

    def forward(self, x, mask=None):
        """
        Defines the forward pass of the DQN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 14, height, width).
            mask (torch.Tensor, optional): Optional mask tensor to apply. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = nn.Flatten()(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        if mask is not None:
            x = self.mask(x, mask)

        return x
