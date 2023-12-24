import torch.nn as nn


class MNISTToy(nn.Module):
    def __init__(self, image_size, num_classes):
        super(ExampleModel, self).__init__()
        in_planes = image_size[0] * image_size[1] * image_size[2]
        self.seq_blocks = nn.Sequential(
            nn.Linear(in_planes, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x):
        return self.seq_blocks(x.view(x.size(0), -1))
