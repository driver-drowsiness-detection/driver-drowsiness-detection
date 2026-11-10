# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# RT-BENE by Kevin Cortacero, Tobias Fischer and Yiannis Demiris.
# Source: Tobias-Fischer/rt_gene commit fa46a87e5f6fdc6f13be5d59572037c29ff9fc8b
# rt_gene_core/src/rt_bene/blink_estimation_models_pytorch.py
# Modification: retain only the ResNet18 inference architecture.
# License: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#! /usr/bin/env python

import torch
import torch.nn as nn
from torchvision import models


class BlinkEstimationAbstractModel(nn.Module):

    def __init__(self):
        super(BlinkEstimationAbstractModel, self).__init__()

    def _create_fc_layers(self, in_features, out_features, dropout_p=0.6):
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512, momentum=0.999, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, out_features)
        )

    @staticmethod
    def _init_weights(modules):
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, left_eye, right_eye):
        left_x = self.left_features(left_eye)
        left_x = torch.flatten(left_x, 1)

        right_x = self.right_features(right_eye)
        right_x = torch.flatten(right_x, 1)

        eyes_x = torch.cat((left_x, right_x), dim=1)
        fc_output = self.fc(eyes_x)

        return fc_output


class BlinkEstimationModelResnet18(BlinkEstimationAbstractModel):

    def __init__(self, num_out=1):
        super(BlinkEstimationModelResnet18, self).__init__()
        _left_model = models.resnet18(weights=None)
        _right_model = models.resnet18(weights=None)

        # remove the last ConvBRelu layer
        self.left_features = nn.Sequential(
            _left_model.conv1,
            _left_model.bn1,
            _left_model.relu,
            _left_model.maxpool,
            _left_model.layer1,
            _left_model.layer2,
            _left_model.layer3,
            _left_model.layer4,
            _left_model.avgpool
        )

        self.right_features = nn.Sequential(
            _right_model.conv1,
            _right_model.bn1,
            _right_model.relu,
            _right_model.maxpool,
            _right_model.layer1,
            _right_model.layer2,
            _right_model.layer3,
            _right_model.layer4,
            _right_model.avgpool
        )

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self._create_fc_layers(in_features=_left_model.fc.in_features + _right_model.fc.in_features,
                               out_features=num_out)
        self._init_weights(self.modules())
