# TAMIL-Ember/models/tamil_ember.py
import torch
import torch.nn as nn
from models.autoencoder import AutoEncoder, AutoencoderRelu, AutoencoderSigmoid, AutoencoderTanh

class BackboneMLP(nn.Module):
    def __init__(self, input_dim=2381, hidden_dim=512, feat_dim=128, n_tasks=11, code_dim=64, n_classes=100, ae_type='relu'):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
            nn.ReLU(),
        )
        self.feat_dim = feat_dim

        AE_CLS = {
            'relu':   AutoencoderRelu,
            'sigmoid': AutoencoderSigmoid,
            'tanh':  AutoencoderTanh,
        }[ae_type]

        self.ae = nn.ModuleList([
            AE_CLS(input_dims=feat_dim, code_dims=code_dim)
            for _ in range(n_tasks)
        ])

        self.linear = nn.Linear(feat_dim, n_classes)

    #         AE_CLS(input_dims=feat_dim, code_dims=code_dim)
    #         for _ in range(n_tasks)
    #     ])
    #     self.linear = nn.Linear(feat_dim, n_classes)

    # def forward(self, x):
    #     return self.linear(self.features(x))

    # def features(self, x):
    #     return self.layers(x)

    def forward(self, x, task_id = None):
        feats = self.features(x)
        if task_id is None:
            return self.linear(feats)

        ae_output = self.ae[task_id](feats)
        combined = feats * ae_output  # TAMIL 논문 방식: 특징과 AE 출력의 elementwise product
        logits = self.linear(combined)
        return logits

    def features(self, x):
        return self.layers(x)

class TAMIL_Ember(nn.Module):
    def __init__(self, input_dim, hidden_dim, feat_dim, latent_dim, n_tasks, n_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
            nn.ReLU(),
        )
        self.task_aes = nn.ModuleList([
            AutoEncoder(feat_dim, latent_dim)
            for _ in range(n_tasks)
        ])
        self.classifier = nn.Linear(feat_dim, n_classes)
        self.n_tasks = n_tasks
        self.current_task = 0   

    def forward(self, x, task_id):
        if task_id is None:
            task_id = self.current_task

        feats = self.backbone(x)
        ae_out = self.task_aes[task_id](feats)

        combined = feats * ae_out
        logits = self.classifier(combined)
        return logits
    
    def features(self, x):
        return self.backbone(x)

    def set_task(self, task_id):
        self.current_task = task_id

    def observe(self, x, y, bx=None, by=None, buffer_weight=1.0):
        logits = self.forward(x, self.current_task)
        main_loss = nn.CrossEntropyLoss()(logits, y)
        # print(f"[DEBUG] Main loss: {main_loss.item():.4f}")
        total_loss = main_loss

        if bx is not None and by is not None:
            buffer_logits = self.forward(bx, self.current_task)  
            buffer_loss = nn.CrossEntropyLoss()(buffer_logits, by)
            #print(f"[DEBUG] Buffer loss: {buffer_loss.item():.4f}")
            total_loss = total_loss + buffer_weight * buffer_loss

        return total_loss