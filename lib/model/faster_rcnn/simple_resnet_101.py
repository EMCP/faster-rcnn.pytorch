"""
simple_resnet.py is a wrapper around the PyTorch Vision provided ResNet101 neural network
"""
import torch.nn as nn
from torchvision.models.resnet import resnet101

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN


class SimpleResNet101(nn.Module):
    """
    See https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    We utilize the pre-trained 101 layer version by default.
    """
    def __init__(self, classes, class_agnostic):
        super(SimpleResNet101, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic

        self.network = resnet101(pretrained=True)

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # change last layer of ResNet pretrained
        num_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features=num_features, out_features=self.n_classes)



    def forward(self, x):  # pylint: disable=arguments-differ
        return self.network.forward(x)