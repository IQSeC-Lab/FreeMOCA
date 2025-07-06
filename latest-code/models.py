import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from convkan_layer import ConvKAN
from kanlinear import KANLinear


class KANClassifier(nn.Module):
    def __init__(self):
        super(KANClassifier, self).__init__()

        self.input_features = 2381
        self.output_dim = 50
        self.drop_prob = 0.5

        self.block1 = nn.Sequential(
            ConvKAN(self.input_features, 512, kernel_size=(3, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
            ConvKAN(512, 256, kernel_size=(3, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.Dropout(self.drop_prob),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=(1, 0))
        )

        self.block2 = nn.Sequential(
            ConvKAN(256, 128, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.Dropout(self.drop_prob)
        )

        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = KANLinear(128, self.output_dim)
        self.fc1_bn1 = nn.BatchNorm1d(self.output_dim)
        self.fc1_drop1 = nn.Dropout(self.drop_prob)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        original_shape = x.size()

        if len(original_shape) == 2:
            batch_size = original_shape[0]
        elif len(original_shape) == 3:
            batch_size = original_shape[0] * original_shape[1]
            x = x.view(batch_size, self.input_features)

        x = x.view(batch_size, self.input_features, -1, 1)  # 4D for ConvKAN
    
        x = self.block1(x)

        x = self.block2(x)
        

        x = self.flatten(x)
       
        x = self.fc1(x)
        x = self.fc1_bn1(x)
        
        x = self.fc1_drop1(x)
        x = self.softmax(x)

        if len(original_shape) == 3:
            x = x.view(original_shape[0], original_shape[1], -1)

        return x

    
    def expand_output_layer(self, init_classes, nb_inc, task):
        old_fc1 = self.fc1
        old_fc1_bn1 = self.fc1_bn1
        self.output_dim = init_classes + nb_inc * task

        self.fc1 = KANLinear(old_fc1.in_features, self.output_dim)
        self.fc1_bn1 = nn.BatchNorm1d(self.output_dim)

        with torch.no_grad():
            self.fc1.base_weight[:, :old_fc1.out_features, :].copy_(old_fc1.base_weight.data)
            self.fc1.spline_weight[:, :old_fc1.out_features, :, :].copy_(old_fc1.spline_weight.data)
            # old_basis = old_fc1.spline_weight.size(-1)
            # new_basis = self.fc1.spline_weight.size(-1)
            # shared_basis = min(old_basis, new_basis)

            # self.fc1.spline_weight[:, :old_fc1.out_features, :, :shared_basis] \
            #     .copy_(old_fc1.spline_weight[:, :, :, :shared_basis])
            if old_fc1.enable_standalone_scale_spline:
                self.fc1.spline_scaler[:, :old_fc1.out_features, :].copy_(old_fc1.spline_scaler.data)

            self.fc1_bn1.weight[:old_fc1_bn1.num_features].copy_(old_fc1_bn1.weight.data)
            self.fc1_bn1.bias[:old_fc1_bn1.num_features].copy_(old_fc1_bn1.bias.data)

        return self


    def predict(self, x_data):
        return self.forward(x_data)

    def get_logits(self, x):
        original_shape = x.size()

        if len(original_shape) == 2:
            batch_size = original_shape[0]
        elif len(original_shape) == 3:
            batch_size = original_shape[0] * original_shape[1]
            x = x.view(batch_size, self.input_features)

        x = x.view(batch_size, self.input_features, -1, 1)
        x = self.block1(x)
        x = self.block2(x)

        logits = self.flatten(x)
        return logits.detach()


def interpolate_models(old_model, new_model, alpha=0.5, method='linear'):
    """
    Interpolates between shared parameters of two models using the specified method.
    
    Parameters:
        old_model (nn.Module): Model from the previous task (slow learner).
        new_model (nn.Module): Model trained on the current task (fast learner).
        alpha (float): Interpolation weight (0.0 = only old, 1.0 = only new).
        method (str): 'linear', 'polynomial', or 'spline'.

    Returns:
        nn.Module: Interpolated model.
    """

    def interpolate_tensor(old_tensor, new_tensor, method, alpha):
        """
        Interpolates between two tensors using the specified method.
        Only supports 1D and 2D tensors (weights, biases).
        """
        old_np = old_tensor.detach().cpu().numpy()
        new_np = new_tensor.detach().cpu().numpy()

        if method == 'linear':
            interpolated_np = (1 - alpha) * old_np + alpha * new_np

        elif method == 'polynomial':
            # Degree-2 polynomial (quadratic) interpolation
            x_vals = [0, 0.5, 1]
            y_vals = np.stack([old_np, (old_np + new_np) / 2, new_np], axis=0)
            poly_interp = interp1d(x_vals, y_vals, axis=0, kind='quadratic', fill_value="extrapolate")
            interpolated_np = poly_interp(alpha)

        elif method == 'spline':
            x_vals = [0, 1]
            y_vals = np.stack([old_np, new_np], axis=0)
            spline = CubicSpline(x_vals, y_vals, axis=0)
            interpolated_np = spline(alpha)

        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        return torch.tensor(interpolated_np, dtype=new_tensor.dtype, device=new_tensor.device)

    # Begin interpolation
    interpolated_state = {}
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()

    for name, new_param in new_state.items():
        if name in old_state:
            old_param = old_state[name].to(new_param.device)
            if old_param.shape == new_param.shape:
                interpolated_state[name] = interpolate_tensor(old_param, new_param, method, alpha)
            else:
                interpolated_state[name] = new_param.clone()
        else:
            interpolated_state[name] = new_param.clone()

    # Load interpolated parameters into a model copy
    interpolated_model = deepcopy(new_model)
    interpolated_model.load_state_dict(interpolated_state)
    return interpolated_model

import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import torch
from copy import deepcopy

def interpolate_models_KAN(old_model, new_model, alpha=0.5, method='linear'):
    import numpy as np
    from scipy.interpolate import interp1d, CubicSpline
    from copy import deepcopy
    import torch

    def interpolate_tensor(old_tensor, new_tensor, method, alpha):
        old_np = old_tensor.detach().cpu().numpy()
        new_np = new_tensor.detach().cpu().numpy()

        if method == 'linear':
            interpolated_np = (1 - alpha) * old_np + alpha * new_np
        elif method == 'polynomial':
            x_vals = [0, 0.5, 1]
            y_vals = np.stack([old_np, (old_np + new_np) / 2, new_np], axis=0)
            interpolated_np = interp1d(x_vals, y_vals, axis=0, kind='quadratic', fill_value="extrapolate")(alpha)
        elif method == 'spline':
            x_vals = [0, 1]
            y_vals = np.stack([old_np, new_np], axis=0)
            interpolated_np = CubicSpline(x_vals, y_vals, axis=0)(alpha)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        return torch.tensor(interpolated_np, dtype=new_tensor.dtype, device=new_tensor.device)

    interpolated_state = {}
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()
    for name, new_param in new_state.items():
        if name in old_state:
            old_param = old_state[name].to(new_param.device)

            # Case 1: Fully matched → interpolate whole tensor
            if old_param.shape == new_param.shape:
                interpolated_state[name] = interpolate_tensor(old_param, new_param, method, alpha)

            # Case 2: Interpolate matching prefix along dim=1 (output_dim)
            elif (
                old_param.dim() == new_param.dim() and
                old_param.shape[0] == new_param.shape[0] and  # typically groups = 1
                old_param.shape[2:] == new_param.shape[2:] and
                old_param.shape[1] < new_param.shape[1]       # output_dim grew
            ):
                d1 = old_param.shape[1]
                interpolated = new_param.clone()
                interpolated[:, :d1, ...] = interpolate_tensor(
                    old_param, new_param[:, :d1, ...], method, alpha
                )
                interpolated_state[name] = interpolated

            # Case 3: Interpolate flat vectors like fc1_bn1.weight
            elif (
                old_param.dim() == 1 and
                old_param.shape[0] < new_param.shape[0]
            ):
                d0 = old_param.shape[0]
                interpolated = new_param.clone()
                interpolated[:d0] = interpolate_tensor(old_param, new_param[:d0], method, alpha)
                interpolated_state[name] = interpolated

            else:
                print(f"Shape mismatch for {name}: skipping interpolation.")
                interpolated_state[name] = new_param.clone()
        else:
            interpolated_state[name] = new_param.clone()


    interpolated_model = deepcopy(new_model)
    interpolated_model.load_state_dict(interpolated_state)
    return interpolated_model
