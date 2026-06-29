# TAMIL-Ember/models/autoencoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.ReLU())
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

class BaseAttention(nn.Module):

    def __init__(self):
        super(BaseAttention, self).__init__()

    def forward(self, x):
        encoded_x = self.encoder(x)
        reconstructed_x = self.decoder(encoded_x)
        return reconstructed_x


class SigmoidAlone(BaseAttention):

    def __init__(self):
        super(SigmoidAlone, self).__init__()
        self.encoder = nn.Sequential(
            nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.Identity())


class LinearSigmoid(BaseAttention):

    def __init__(self,  input_dims=512, code_dims=512):
        super(LinearSigmoid, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, code_dims),
            nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.Identity())


class AutoencoderSigmoid(BaseAttention):

    def __init__(self, input_dims=512, code_dims=256):
        super(AutoencoderSigmoid, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, code_dims),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(code_dims, input_dims),
            nn.Sigmoid())


class AutoencoderTanh(BaseAttention):

    def __init__(self, input_dims=512, code_dims=256):
        super(AutoencoderTanh, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, code_dims),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(code_dims, input_dims),
            nn.Tanh())


class AutoencoderRelu(BaseAttention):

    def __init__(self, input_dims=512, code_dims=256):
        super(AutoencoderRelu, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, code_dims),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(code_dims, input_dims),
            nn.ReLU())


