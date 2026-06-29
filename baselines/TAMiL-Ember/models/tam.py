# TAMIL-Ember/models/tam.py

from models.buffer import Buffer
from torch.nn import functional as F
from models.continual_model import ContinualModel
from utils.args import *
import torch
from copy import deepcopy
from argparse import ArgumentParser


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via TAM')
                                       # ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class TAM(ContinualModel):
    NAME = 'tam'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(TAM, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task_id = 0
        self.exclude_layers_start_with = ['ae', 'linear']
        self.ema_model = deepcopy(self.net).to(self.device)
        # set regularization weight
        self.reg_weight = self.args.reg_weight
        # set parameters for ema model
        self.ema_update_freq = self.args.ema_update_freq
        self.ema_alpha = self.args.ema_alpha
        self.consistency_loss = torch.nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0

    def forward(self, x):
        feats = self.net.features(x)
        ae_output = self.net.ae[self.task_id](feats) # 현재 task의 AE 사용
        combined = feats * ae_output  # TAMIL 논문 방식: 특징과 AE 출력의 elementwise product
        logits = self.net.linear(combined)
        return logits

    def forward_with_task(self, x, task_id):
        feats = self.net.features(x)
        ae_output = self.net.ae[task_id](feats)  # 지정된 task의 AE 사용
        combined = feats * ae_output  # TAMIL 논문 방식: 특징과 AE 출력의 elementwise product
        logits = self.net.linear(combined)
        return logits
    
    def update_ema_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.ema_alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1-alpha)

    def end_task(self, dataset):
        self.task_id += 1

    def buffer_through_ae(self):
        buf_inputs, buf_labels, buf_logits, task_labels = self.buffer.get_data(
            self.args.minibatch_size, transform=self.transform)
        
        if buf_inputs is None:
            return None, None, None
        
        buf_inputs = buf_inputs.to(self.device)
        buf_labels = buf_labels.to(self.device)
        task_labels = task_labels.to(self.device)
        
        buf_feats = self.net.features(buf_inputs)
        buf_feats_ema = self.ema_model.features(buf_inputs)
        
        buf_output_ae = torch.zeros((self.args.minibatch_size, self.task_id + 1, buf_feats.shape[-1]),
                                      device=self.device)
        buf_output_ae_ema = torch.zeros((self.args.minibatch_size, self.task_id + 1, buf_feats.shape[-1]),
                                      device=self.device)
        err_ae = torch.zeros((self.args.minibatch_size, self.task_id + 1), device=self.device)
        
        for i in range(self.task_id + 1):
            out_ae_i = self.net.ae[i](buf_feats)
            recon_e = F.mse_loss(out_ae_i, buf_feats, reduction='none')
            err_ae[:, i] = torch.mean(recon_e, dim=1)
            buf_output_ae[:, i, :] = out_ae_i
            
            out_ae_i_ema = self.ema_model.ae[i](buf_feats_ema.detach())
            buf_output_ae_ema[:, i, :] = out_ae_i_ema

        # current model
        indices = torch.argmin(err_ae, dim=1)
        mask = F.one_hot(indices, self.task_id + 1)
        mask = mask.unsqueeze(2).expand(-1, -1, buf_feats.shape[-1])
        buf_output_ae_selected = torch.sum(buf_output_ae * mask, keepdim=True, dim=1).squeeze()
        buf_outputs = self.net.linear(buf_feats * buf_output_ae_selected)
        
        # EMA model
        mask_ema = F.one_hot(task_labels.long(), self.task_id + 1)
        mask_ema = mask_ema.unsqueeze(2).expand(-1, -1, buf_feats_ema.shape[-1])
        buf_output_ae_ema_selected = torch.sum(buf_output_ae_ema * mask_ema, keepdim=True, dim=1).squeeze()
        buf_logits_out = self.ema_model.linear(buf_feats_ema * buf_output_ae_ema_selected)

        return buf_outputs, buf_logits_out, buf_labels

    def observe(self, inputs, labels, not_aug_inputs):

        if inputs is None or labels is None:
            return 0,0, 0.0

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        not_aug_inputs = not_aug_inputs.to(self.device) if not_aug_inputs is not None else inputs

        self.opt.zero_grad()
        # outputs = self.net(inputs)
        # CE for current task samples
        # loss = 0

        # current task loss
        feats = self.net.features(inputs)
        outputs_encoder = self.net.ae[self.task_id].encoder(feats)
        outputs_ae = self.net.ae[self.task_id].decoder(outputs_encoder)
        outputs = self.net.linear(outputs_ae * feats)
        
        loss = self.loss(outputs, labels)

        # if self.args.use_pairwise_loss_after_ae or self.args.load_best_args:
        if getattr(self.args, "use_pairwise_loss_after_ae", False) or getattr(self.args, "load_best_args", False):
            softmax = torch.nn.Softmax(dim=-1)
            for i in range(self.task_id):
                outputs_i = self.net.ae[i](feats)
                pairwise_dist = torch.pairwise_distance(softmax(outputs_i.detach()),
                                                        softmax(outputs_ae), p=1).mean()
                loss = loss - self.args.pairwise_weight * (pairwise_dist)

        # Buffer consistency loss
        loss_consistency = torch.tensor(0.0, device=self.device)
        if not self.buffer.is_empty():

            # buf_size = self.buffer.num_seen_examples
            # print(f"[DEBUG] Buffer size: {buf_size}")
            # _, buf_labels, _, _ = self.buffer.get_data(10)           
            # if buf_labels is not None:
            #     print(f"[DEBUG] Buffer label dist: {torch.bincount(buf_labels.cpu())}")
            
            buf_outputs_1, buf_logits_1, _ = self.buffer_through_ae()
            if buf_outputs_1 is not None and buf_logits_1 is not None:
                loss_consistency = self.reg_weight * F.mse_loss(buf_outputs_1, buf_logits_1.detach())
                loss = loss + loss_consistency

                # CE for buffered images
                buf_outputs_2, _, buf_labels_2 = self.buffer_through_ae()
                if buf_outputs_2 is not None and buf_labels_2 is not None:
                    loss = loss + self.args.beta * self.loss(buf_outputs_2, buf_labels_2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.opt.step()

        task_labels = torch.ones(labels.shape[0], device=self.device) * self.task_id
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             task_labels=task_labels)

        # Update the ema model
        self.global_step += 1
        if torch.rand(1).item() < self.ema_update_freq:
            self.update_ema_model_variables()

        return loss.item(), loss_consistency.item()
    
    def set_task(self, task_id):
        self.current_task = task_id 
        self.task_id = task_id
