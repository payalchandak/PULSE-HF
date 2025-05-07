import torch
from torch import nn

def conv15x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=stride, padding=7)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm1d
        self.conv1 = conv15x1(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv15x1(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_outputs=1, num_channels=12, norm_layer=nn.BatchNorm1d, embed_only=True, dropout_prob=0.5):

        super(ResNet, self).__init__()
        """
        Args:
        - block (nn.Module): The type of block to use (e.g., BasicBlock).
        - layers (list of int): List containing the number of blocks in each of the 4 layers of the network.
        - num_outputs (int, optional): Number of neurons in the final output layer. Default is 1.
        - num_channels (int, optional): Number of input channels. Default is 12.
        - groups (int, optional): Number of groups for group normalization. Not used here, but kept for possible extensions.
        - width_per_group (int, optional): Width of each group. Not used here, but kept for possible extensions.
        - replace_stride_with_dilation (bool, optional): If true, replaces stride with dilation. Default is None.
        - norm_layer (torch.nn.Module, optional): Type of normalization layer to use. Default is BatchNorm1d.
        """
        self.num_outputs = num_outputs
        self.num_channels = num_channels
        self._norm_layer = norm_layer
        self.embed_only = embed_only
        # Initial convolutional layer
        self.inplanes = 32
        self.conv1 = nn.Conv1d(self.num_channels, self.inplanes, kernel_size=15, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        # ResNet blocks
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        # Adaptive average pooling and final fully connected layer to produce a fixed size output tensor.
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if not self.embed_only: 
            self.fc = nn.Sequential(nn.Linear(256, self.num_outputs))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        Create a layer of blocks.

        Args:
        - block (nn.Module): Type of block to use (e.g., BasicBlock).
        - planes (int): Number of output channels for this layer.
        - blocks (int): Number of blocks in the layer.
        - stride (int, optional): Stride for the first block in the layer. Default is 1.
        - dilate (bool, optional): Whether to replace stride with dilation. Default is False.

        Returns:
        - torch.nn.Module: A sequence of blocks.
        """
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                    norm_layer(planes),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if not self.embed_only: x = self.fc(x)
        return x

def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def ResNet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)

def ResNet34(**kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)