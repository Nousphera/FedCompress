"""Paper backbones: CIFAR ResNet-20 and the MobileNet-v1 YAMNet backbone.

YAMNet topology adapted from Google/TensorFlow AudioSet (2019), Apache-2.0.
The PyTorch port retains asymmetric SAME padding and offset-only audio BN.
"""

import torch
from torch import nn
from torch.nn import functional as F

CLASSES = {"cifar10": 10, "cifar100": 100, "pathmnist": 9, "spcm": 12, "voxforge": 6}
AUDIO = {"spcm", "voxforge"}


class SameConv(nn.Conv2d):
    def forward(self, x):
        pads = []
        for size, kernel, stride, dilation in reversed(
            list(zip(x.shape[-2:], self.kernel_size, self.stride, self.dilation))
        ):
            total = max(
                0,
                ((size + stride - 1) // stride - 1) * stride
                + dilation * (kernel - 1)
                + 1
                - size,
            )
            pads.extend((total // 2, total - total // 2))
        return F.conv2d(
            F.pad(x, pads),
            self.weight,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )


class OffsetBatchNorm(nn.BatchNorm2d):
    def __init__(self, channels):
        super().__init__(channels, eps=1e-4, momentum=0.01)
        self.weight.requires_grad_(False)


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.conv1 = SameConv(inputs, outputs, 3, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(outputs, eps=1e-3, momentum=0.01)
        self.conv2 = SameConv(outputs, outputs, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(outputs, eps=1e-3, momentum=0.01)
        self.stride, self.extra = stride, outputs - inputs

    def forward(self, x):
        shortcut = x[:, :, :: self.stride, :: self.stride]
        if self.extra:
            shortcut = torch.cat(
                (
                    shortcut,
                    shortcut.new_zeros(
                        shortcut.shape[0], self.extra, *shortcut.shape[2:]
                    ),
                ),
                dim=1,
            )
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(out + shortcut)


class ResNet20(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.stem = nn.Sequential(
            SameConv(3, 16, 3, bias=False),
            nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        blocks, inputs = [], 16
        for stage, outputs in enumerate((16, 32, 64)):
            for i in range(3):
                blocks.append(
                    ResidualBlock(inputs, outputs, 2 if stage and i == 0 else 1)
                )
                inputs = outputs
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(64, classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def features(self, x):
        return self.blocks(self.stem(x)).mean((2, 3))

    def forward(self, x):
        return self.head(self.features(x))


class YAMNet(nn.Module):
    def __init__(self, classes):
        super().__init__()

        def conv(inputs, outputs, kernel, stride=1, groups=1):
            return nn.Sequential(
                SameConv(
                    inputs, outputs, kernel, stride=stride, groups=groups, bias=False
                ),
                OffsetBatchNorm(outputs),
                nn.ReLU(),
            )

        blocks = [conv(1, 32, 3, 2)]
        inputs = 32
        for outputs, stride in [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            *(5 * [(512, 1)]),
            (512, 1),
            (1024, 2),
            (1024, 1),
        ]:
            # The first 512-channel stage downsamples; subsequent five do not.
            if outputs == 512 and inputs != 512:
                stride = 2
            blocks.append(
                nn.Sequential(
                    conv(inputs, inputs, 3, stride, inputs), conv(inputs, outputs, 1)
                )
            )
            inputs = outputs
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(1024, classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def features(self, x):
        return self.blocks(x).mean((2, 3))

    def forward(self, x):
        return self.head(self.features(x))


def make_model(dataset, seed=42):
    if dataset not in CLASSES:
        raise ValueError(f"Unknown dataset: {dataset}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return (
            YAMNet(CLASSES[dataset]) if dataset in AUDIO else ResNet20(CLASSES[dataset])
        )
