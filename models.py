import numpy as np
import os
import os.path as opth
import argparse
import torch
import torch.nn as nn
from copy import deepcopy


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.input_features = 2381
        self.output_dim = 50
        self.drop_prob = 0.5

        self.block1 = nn.Sequential(
            nn.Conv1d(self.input_features, 512, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 3, 3, 1),
            nn.BatchNorm1d(256),
            nn.Dropout(self.drop_prob),
            nn.ReLU(),
            nn.MaxPool1d(3, 3, 1)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.Dropout(self.drop_prob),
            nn.ReLU()
        )
        
        self.fc1_f = nn.Flatten()
        self.fc1 = nn.Linear(128, self.output_dim)
        self.fc1_bn1 = nn.BatchNorm1d(self.output_dim)
        self.fc1_drop1 = nn.Dropout(self.drop_prob)
        self.fc1_act1 = nn.ReLU()
        

        self.softmax = nn.Softmax()

    def forward(self, x):
     
        # Get the original shape of the input tensor
        original_shape = x.size()

        # Reshape input data based on whether it's training or testing
        if len(original_shape) == 2:
            batch_size = original_shape[0]
        elif len(original_shape) == 3:
            batch_size = original_shape[0] * original_shape[1]
            x = x.view(batch_size, self.input_features)

        x = x.view(batch_size, self.input_features, -1)
        x = self.block1(x)
        x = self.block2(x)

        x = self.fc1_f(x)
        x = self.fc1(x)
        x = self.fc1_bn1(x)
        x = self.fc1_drop1(x)
        x = self.fc1_act1(x)

        x = self.softmax(x)   

        # If testing, reshape the output tensor back to the original shape
        if len(original_shape) == 3:
            x = x.view(original_shape[0], original_shape[1], -1)

        return x

    def expand_output_layer(self, init_classes, nb_inc, task):

        old_fc1 = self.fc1
        old_fc1_bn1 = self.fc1_bn1
        self.output_dim = init_classes + nb_inc * task

        self.fc1 = nn.Linear(old_fc1.in_features, self.output_dim)
        self.fc1_bn1 = nn.BatchNorm1d(self.output_dim)

        with torch.no_grad():
            self.fc1.weight[:old_fc1.out_features].copy_(old_fc1.weight.data)
            self.fc1.bias[:old_fc1.out_features].copy_(old_fc1.bias.data)
            self.fc1_bn1.weight[:old_fc1_bn1.num_features].copy_(old_fc1_bn1.weight.data)
            self.fc1_bn1.bias[:old_fc1_bn1.num_features].copy_(old_fc1_bn1.bias.data)
        return self


    def predict(self, x_data):
        result = self.forward(x_data)
        
        return result

    def get_logits(self, x):
        
        # Get the original shape of the input tensor
        original_shape = x.size()

        if len(original_shape) == 2:
            batch_size = original_shape[0]
        elif len(original_shape) == 3:
            batch_size = original_shape[0] * original_shape[1]
            x = x.view(batch_size, self.input_features)

        x = x.view(batch_size, self.input_features, -1)
        x = self.block1(x)
        x = self.block2(x)

        logits = self.fc1_f(x)
        
        return logits.detach()
# import torch
# from scipy.interpolate import interp1d, CubicSpline
# import numpy as np
# from copy import deepcopy

# def interpolate_models(old_model, new_model, alpha=0.5, method='linear'):
#     """
#     Interpolates between shared parameters of two models using the specified method.
    
#     Parameters:
#         old_model (nn.Module): Model from the previous task (slow learner).
#         new_model (nn.Module): Model trained on the current task (fast learner).
#         alpha (float): Interpolation weight (0.0 = only old, 1.0 = only new).
#         method (str): 'linear', 'polynomial', or 'spline'.

#     Returns:
#         nn.Module: Interpolated model.
#     """

#     def interpolate_tensor(old_tensor, new_tensor, method, alpha):
#         """
#         Interpolates between two tensors using the specified method.
#         Only supports 1D and 2D tensors (weights, biases).
#         """
#         old_np = old_tensor.detach().cpu().numpy()
#         new_np = new_tensor.detach().cpu().numpy()

#         if method == 'linear':
#             interpolated_np = (1 - alpha) * old_np + alpha * new_np

#         elif method == 'polynomial':
#             # Degree-2 polynomial (quadratic) interpolation
#             x_vals = [0, 0.5, 1]
#             y_vals = np.stack([old_np, (old_np + new_np) / 2, new_np], axis=0)
#             poly_interp = interp1d(x_vals, y_vals, axis=0, kind='quadratic', fill_value="extrapolate")
#             interpolated_np = poly_interp(alpha)

#         elif method == 'spline':
#             x_vals = [0, 1]
#             y_vals = np.stack([old_np, new_np], axis=0)
#             spline = CubicSpline(x_vals, y_vals, axis=0)
#             interpolated_np = spline(alpha)

#         else:
#             raise ValueError(f"Unknown interpolation method: {method}")

#         return torch.tensor(interpolated_np, dtype=new_tensor.dtype, device=new_tensor.device)

#     # Begin interpolation
#     interpolated_state = {}
#     old_state = old_model.state_dict()
#     new_state = new_model.state_dict()

#     for name, new_param in new_state.items():
#         if name in old_state:
#             old_param = old_state[name].to(new_param.device)
#             if old_param.shape == new_param.shape:
#                 interpolated_state[name] = interpolate_tensor(old_param, new_param, method, alpha)
#             else:
#                 interpolated_state[name] = new_param.clone()
#         else:
#             interpolated_state[name] = new_param.clone()

#     # Load interpolated parameters into a model copy
#     interpolated_model = deepcopy(new_model)
#     interpolated_model.load_state_dict(interpolated_state)
#     return interpolated_model


# def interpolate_models_fisher(old_model, new_model, fisher_old, fisher_new, alpha=0.5):
#     """
#     Interpolate model parameters using Fisher-weighted importance.
#     Handles expanding output layers by only interpolating shared dimensions.
#     """
#     interpolated_model = deepcopy(new_model)
#     new_state = new_model.state_dict()
#     old_state = old_model.state_dict()

#     interpolated_state = {}

#     for name, new_param in new_state.items():
#         if name in old_state and name in fisher_old and name in fisher_new:
#             old_param = old_state[name]
#             f_old = fisher_old[name].to(new_param.device)
#             f_new = fisher_new[name].to(new_param.device)

#             if new_param.shape == old_param.shape:
#                 # Full interpolation
#                 weight = alpha * f_new / (f_old + f_new + 1e-8)
#                 interpolated_param = (1 - weight) * old_param + weight * new_param
#             else:
#                 # Partial interpolation: handle only shared part
#                 interpolated_param = new_param.clone()
#                 shared_dims = tuple(min(o, n) for o, n in zip(old_param.shape, new_param.shape))

#                 # Handle 2D and 1D tensors
#                 if len(shared_dims) == 2:
#                     f_old_part = f_old[:shared_dims[0], :shared_dims[1]]
#                     f_new_part = f_new[:shared_dims[0], :shared_dims[1]]
#                     weight = alpha * f_new_part / (f_old_part + f_new_part + 1e-8)
#                     interpolated_param[:shared_dims[0], :shared_dims[1]] = (
#                         (1 - weight) * old_param[:shared_dims[0], :shared_dims[1]] +
#                         weight * new_param[:shared_dims[0], :shared_dims[1]]
#                     )
#                 elif len(shared_dims) == 1:
#                     f_old_part = f_old[:shared_dims[0]]
#                     f_new_part = f_new[:shared_dims[0]]
#                     weight = alpha * f_new_part / (f_old_part + f_new_part + 1e-8)
#                     interpolated_param[:shared_dims[0]] = (
#                         (1 - weight) * old_param[:shared_dims[0]] +
#                         weight * new_param[:shared_dims[0]]
#                     )
#                 else:
#                     raise ValueError(f"Unsupported tensor shape for interpolation: {new_param.shape}")

#         else:
#             # Copy new parameter if no match or new layer
#             interpolated_param = new_param.clone()

#         interpolated_state[name] = interpolated_param

#     interpolated_model.load_state_dict(interpolated_state)
#     return interpolated_model


