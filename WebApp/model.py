import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models
import torch.optim as optim

class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        base_model = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(base_model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=1024,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=2048,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape

        x = x.view(batch_size*seq_len, c, h, w)
        x = self.cnn(x)
        x = self.avgpool(x)
        x = x.view(batch_size, seq_len, -1)

        gru_out, _ = self.gru(x)

        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        context = gru_out + attn_out

        context = torch.mean(context, dim=1)
        return self.classifier(context)