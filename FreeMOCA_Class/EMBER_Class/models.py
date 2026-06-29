import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb
from copy import deepcopy
from convkan_layer import ConvKAN
from kanlinear import KANLinear
import numpy as np
from scipy.interpolate import interp1d, CubicSpline

###EMBER
class Classifier(nn.Module):

    def __init__(self, output_dim):
        super(Classifier, self).__init__()

        self.input_features = 2381
        # self.input_features = 624384
        self.output_dim = output_dim
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
     
        # print("DEBUG x.shape before view:", x.shape)

        # Get the original shape of the input tensor
        original_shape = x.size()

        # Reshape input data based on whether it's training or testing
        if len(original_shape) == 2:
            batch_size = original_shape[0]
        elif len(original_shape) == 3:
            batch_size = original_shape[0] * original_shape[1]
            # x = x.view(batch_size, self.input_features)
            x = x.flatten(start_dim=1)


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



###AZ 
import torch
import torch.nn as nn


# class Classifier(nn.Module):

#     def __init__(self, input_features=2381, init_classes=50, drop_prob=0.5):
#         super(Classifier, self).__init__()

#         self.input_features = input_features
#         self.output_dim = init_classes
#         self.drop_prob = drop_prob

#         # -------- Conv Blocks --------
#         self.block1 = nn.Sequential(
#             nn.Conv1d(self.input_features, 512, kernel_size=1),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(self.drop_prob)
#         )

#         self.block2 = nn.Sequential(
#             nn.Conv1d(512, 128, kernel_size=1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(self.drop_prob)
#         )

#         # -------- Classifier Head --------
#         self.fc = nn.Linear(128, self.output_dim)

#     def forward(self, x):
#         """
#         x: [B, F] or [B, T, F]
#         """

#         original_shape = x.shape

#         # Handle temporal input
#         if x.dim() == 3:
#             B, T, F = x.shape
#             x = x.view(B * T, F)

#         # Convert to Conv1D format: [B, C, L=1]
#         x = x.unsqueeze(-1)  # [B, F, 1]

#         x = self.block1(x)
#         x = self.block2(x)

#         # Remove length dimension
#         x = x.squeeze(-1)  # [B, 128]

#         logits = self.fc(x)

#         # Restore temporal shape if needed
#         if len(original_shape) == 3:
#             logits = logits.view(B, T, -1)

#         return logits

#     # -------- Incremental Learning --------
#     def expand_output_layer(self, init_classes, nb_inc, task):

#         old_fc = self.fc
#         new_out_dim = init_classes + nb_inc * task

#         self.fc = nn.Linear(old_fc.in_features, new_out_dim)

#         with torch.no_grad():
#             self.fc.weight[:old_fc.out_features].copy_(old_fc.weight)
#             self.fc.bias[:old_fc.out_features].copy_(old_fc.bias)

#         self.output_dim = new_out_dim
#         return self

#     def get_logits(self, x):
#         return self.forward(x).detach()

#     def predict(self, x):
#         return self.forward(x)




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





def compute_layerwise_lambdas(old_model, new_model,
                              lambda_min=0.3,
                              lambda_max=0.7,
                              eps=1e-12):
    """
    Compute adaptive λ_l for each parameter tensor based on how much
    it changed between old_model and new_model.

    Returns:
        layer_lambdas: dict[name -> lambda_l]
    """
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()

    change_scores = {}

    # 1) Compute change score per parameter (only for floating-point tensors)
    for name, new_param in new_state.items():
        if name in old_state:
            old_param = old_state[name]
            # Skip if shapes mismatch
            if old_param.shape != new_param.shape:
                continue
            # Skip non-floating tensors (e.g., num_batches_tracked: LongTensor)
            if not (torch.is_floating_point(new_param) and torch.is_floating_point(old_param)):
                continue

            old_fp = old_param.to(new_param.device).float()
            new_fp = new_param.to(new_param.device).float()
            diff = (new_fp - old_fp).abs().mean().item()
            change_scores[name] = diff

    if not change_scores:
        # Fallback: no matching float params, just return empty
        return {}

    scores = list(change_scores.values())
    s_min, s_max = min(scores), max(scores)

    layer_lambdas = {}

    # 2) Normalize and map to [lambda_min, lambda_max]
    if s_max - s_min < eps:
        # All layers changed about the same → use middle of range
        mid = 0.5 * (lambda_min + lambda_max)
        for name in change_scores.keys():
            layer_lambdas[name] = mid
    else:
        for name, s in change_scores.items():
            # normalized_change in [0,1]
            normalized_change = (s - s_min) / (s_max - s_min)

            # Inverse mapping – large change → smaller λ (more stable)
            norm = 1.0 - normalized_change
            lambda_l = lambda_min + norm * (lambda_max - lambda_min)
            layer_lambdas[name] = float(lambda_l)

            # Direct mapping – large change → larger λ
            # lambda_l = lambda_min + normalized_change * (lambda_max - lambda_min)
            # layer_lambdas[name] = float(lambda_l)

    return layer_lambdas



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



def interpolate_models(old_model, new_model,
                       alpha=None,        # optional global fallback
                       method='linear',
                       lambda_min=0.3,
                       lambda_max=0.7):
    """
    Interpolates between shared parameters of two models using the specified method,
    with adaptive per-layer λ_l computed from how much each layer changed
    between old_model and new_model.

    Parameters:
        old_model (nn.Module): Model from the previous task.
        new_model (nn.Module): Model trained on the current task.
        alpha (float or None): Optional global interpolation weight; if None,
                               use layerwise λ_l only.
        method (str): 'linear', 'polynomial', or 'spline'.
        lambda_min, lambda_max: Range for λ_l.

    Returns:
        nn.Module: Interpolated model.
    """

    def interpolate_tensor(old_tensor, new_tensor, method, alpha_val):
        """
        Interpolates between two tensors using the specified method.
        """
        old_np = old_tensor.detach().cpu().numpy()
        new_np = new_tensor.detach().cpu().numpy()

        if method == 'linear':
            interpolated_np = (1 - alpha_val) * old_np + alpha_val * new_np

        elif method == 'polynomial':
            # --- FIX: true quadratic curve ---
            x_vals = [0.0, 0.5, 1.0]

            # Nonlinear midpoint (off the straight line)
            mid = 0.5 * (old_np + new_np) + 0.1 * (new_np - old_np)

            y_vals = np.stack([old_np, mid, new_np], axis=0)

            poly_interp = interp1d(
                x_vals,
                y_vals,
                axis=0,
                kind='quadratic',
                fill_value="extrapolate"
            )
            interpolated_np = poly_interp(alpha_val)

        elif method == 'spline':
            # --- FIX: true cubic spline (needs ≥4 points) ---
            x_vals = [0.0, 0.33, 0.66, 1.0]

            y_vals = np.stack([
                old_np,
                old_np + 0.25 * (new_np - old_np),
                old_np + 0.75 * (new_np - old_np),
                new_np
            ], axis=0)

            spline = CubicSpline(x_vals, y_vals, axis=0)
            interpolated_np = spline(alpha_val)

        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        return torch.tensor(
            interpolated_np,
            dtype=new_tensor.dtype,
            device=new_tensor.device
        )


    old_state = old_model.state_dict()
    new_state = new_model.state_dict()
    interpolated_state = {}

    # 1) Compute λ_l per parameter name (based on change between old and new)
    layer_lambdas = compute_layerwise_lambdas(
        old_model, new_model,
        lambda_min=lambda_min,
        lambda_max=lambda_max
    )

    # 2) Interpolate with per-layer λ_l (or global alpha fallback)
    for name, new_param in new_state.items():
        if name in old_state:
            old_param = old_state[name].to(new_param.device)

            if old_param.shape == new_param.shape and torch.is_floating_point(new_param):
                # Choose layer-specific λ_l if available; else fallback
                if alpha is not None:
                    alpha_val = alpha
                else:
                    alpha_val = layer_lambdas.get(name, 0.5)  # default if missing

                interpolated_state[name] = interpolate_tensor(
                    old_param, new_param, method, alpha_val
                )

            else:
                # Shape mismatch or non-float → just take new param as-is
                interpolated_state[name] = new_param.clone()
        else:
            # New parameter → keep as is
            interpolated_state[name] = new_param.clone()

    # Load interpolated parameters into a model copy
    interpolated_model = deepcopy(new_model)
    interpolated_model.load_state_dict(interpolated_state)
    return interpolated_model


import torch
from copy import deepcopy


def average_two_models(old_model, new_model):
    """
    Compute a simple per-parameter average between two subsequent task models:
        theta_avg = 0.5 * theta_prev + 0.5 * theta_current

    Args:
        old_model (nn.Module): model from previous task
        new_model (nn.Module): model from current task

    Returns:
        nn.Module: averaged model, same architecture as new_model
    """
    # Use new_model as template
    avg_model = deepcopy(new_model)
    avg_state = avg_model.state_dict()
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()

    for name, new_param in new_state.items():
        if name in old_state:
            old_param = old_state[name]

            # Only average if shapes match and both are floating tensors
            if (
                old_param.shape == new_param.shape
                and torch.is_floating_point(old_param)
                and torch.is_floating_point(new_param)
            ):
                avg_state[name] = 0.5 * old_param.to(new_param.device) + 0.5 * new_param
            else:
                # Shape mismatch (e.g., expanded classifier) or non-float tensors:
                # keep the current task's parameter
                avg_state[name] = new_param.clone()
        else:
            # Parameter only exists in new_model (e.g., newly added) → keep as is
            avg_state[name] = new_param.clone()

    avg_model.load_state_dict(avg_state)
    return avg_model


# def interpolate_models_KAN(old_model, new_model, alpha=0.5, method='linear'):
    

#     def interpolate_tensor(old_tensor, new_tensor, method, alpha):
#         old_np = old_tensor.detach().cpu().numpy()
#         new_np = new_tensor.detach().cpu().numpy()

#         if method == 'linear':
#             interpolated_np = (1 - alpha) * old_np + alpha * new_np
#         elif method == 'polynomial':
#             x_vals = [0, 0.5, 1]
#             y_vals = np.stack([old_np, (old_np + new_np) / 2, new_np], axis=0)
#             interpolated_np = interp1d(x_vals, y_vals, axis=0, kind='quadratic', fill_value="extrapolate")(alpha)
#         elif method == 'spline':
#             x_vals = [0, 1]
#             y_vals = np.stack([old_np, new_np], axis=0)
#             interpolated_np = CubicSpline(x_vals, y_vals, axis=0)(alpha)
#         else:
#             raise ValueError(f"Unknown interpolation method: {method}")

#         return torch.tensor(interpolated_np, dtype=new_tensor.dtype, device=new_tensor.device)


#     # 1) Compute λ_l per parameter name (based on change between old and new)
#     layer_lambdas = compute_layerwise_lambdas(
#         old_model, new_model,
#         lambda_min=lambda_min,
#         lambda_max=lambda_max
#     )
#     interpolated_state = {}
#     old_state = old_model.state_dict()
#     new_state = new_model.state_dict()
#     for name, new_param in new_state.items():
#         if name in old_state:
#             old_param = old_state[name].to(new_param.device)

#             # Case 1: Fully matched → interpolate whole tensor
#             if old_param.shape == new_param.shape:
            
#                 interpolated_state[name] = interpolate_tensor(old_param, new_param, method, alpha)

#             # Case 2: Interpolate matching prefix along dim=1 (output_dim)
#             elif (
#                 old_param.dim() == new_param.dim() and
#                 old_param.shape[0] == new_param.shape[0] and  # typically groups = 1
#                 old_param.shape[2:] == new_param.shape[2:] and
#                 old_param.shape[1] < new_param.shape[1]       # output_dim grew
#             ):
#                 d1 = old_param.shape[1]
#                 interpolated = new_param.clone()
#                 interpolated[:, :d1, ...] = interpolate_tensor(
#                     old_param, new_param[:, :d1, ...], method, alpha
#                 )
#                 interpolated_state[name] = interpolated

#             # Case 3: Interpolate flat vectors like fc1_bn1.weight
#             elif (
#                 old_param.dim() == 1 and
#                 old_param.shape[0] < new_param.shape[0]
#             ):
#                 d0 = old_param.shape[0]
#                 interpolated = new_param.clone()
#                 interpolated[:d0] = interpolate_tensor(old_param, new_param[:d0], method, alpha)
#                 interpolated_state[name] = interpolated

#             else:
#                 print(f"Shape mismatch for {name}: skipping interpolation.")
#                 interpolated_state[name] = new_param.clone()
#         else:
#             interpolated_state[name] = new_param.clone()


#     interpolated_model = deepcopy(new_model)
#     interpolated_model.load_state_dict(interpolated_state)
#     return interpolated_model

# def interpolate_models_KAN(
#     old_model,
#     new_model,
#     alpha=None,           # optional global fallback
#     method='linear',
#     lambda_min=0.3,
#     lambda_max=0.7,
# ):
#     """
#     Interpolates between parameters of two KAN models using the specified method,
#     with adaptive per-parameter λ_l computed from how much each parameter changed
#     between old_model and new_model.

#     Handles:
#       - full shape matches
#       - partial matches for expanded KAN heads (prefix interpolation)
#       - falls back to new_param when shapes are incompatible
#     """

#     def interpolate_tensor(old_tensor, new_tensor, method, alpha_val):
#         old_np = old_tensor.detach().cpu().numpy()
#         new_np = new_tensor.detach().cpu().numpy()

#         if method == 'linear':
#             interpolated_np = (1 - alpha_val) * old_np + alpha_val * new_np
#         elif method == 'polynomial':
#             x_vals = [0, 0.5, 1]
#             y_vals = np.stack([old_np, (old_np + new_np) / 2, new_np], axis=0)
#             interpolated_np = interp1d(
#                 x_vals, y_vals, axis=0, kind='quadratic', fill_value="extrapolate"
#             )(alpha_val)
#         elif method == 'spline':
#             x_vals = [0, 1]
#             y_vals = np.stack([old_np, new_np], axis=0)
#             interpolated_np = CubicSpline(x_vals, y_vals, axis=0)(alpha_val)
#         else:
#             raise ValueError(f"Unknown interpolation method: {method}")

#         return torch.tensor(
#             interpolated_np,
#             dtype=new_tensor.dtype,
#             device=new_tensor.device,
#         )

#     interpolated_state = {}
#     old_state = old_model.state_dict()
#     new_state = new_model.state_dict()

#     # 1) Compute λ_l per parameter name (only for exact shape matches & float tensors)
#     layer_lambdas = compute_layerwise_lambdas(
#         old_model,
#         new_model,
#         lambda_min=lambda_min,
#         lambda_max=lambda_max,
#     )

#     # 2) Interpolate with per-layer λ_l (or global alpha fallback)
#     for name, new_param in new_state.items():
#         if name in old_state:
#             old_param = old_state[name].to(new_param.device)

#             # Case 1: Fully matched → use layerwise λ_l (or alpha/global fallback)
#             if (
#                 old_param.shape == new_param.shape
#                 and torch.is_floating_point(new_param)
#                 and torch.is_floating_point(old_param)
#             ):
#                 if alpha is not None:
#                     alpha_val = alpha
#                 else:
#                     alpha_val = layer_lambdas.get(name, 0.5)  # fallback if missing

#                 interpolated_state[name] = interpolate_tensor(
#                     old_param, new_param, method, alpha_val
#                 )

#             # Case 2: Interpolate matching prefix along dim=1 (output_dim grew)
#             elif (
#                 old_param.dim() == new_param.dim()
#                 and old_param.shape[0] == new_param.shape[0]  # e.g., groups
#                 and old_param.shape[2:] == new_param.shape[2:]
#                 and old_param.shape[1] < new_param.shape[1]   # expanded out_features
#                 and torch.is_floating_point(new_param)
#                 and torch.is_floating_point(old_param)
#             ):
#                 if alpha is not None:
#                     alpha_val = alpha
#                 else:
#                     alpha_val = layer_lambdas.get(name, 0.5)

#                 d1 = old_param.shape[1]
#                 interpolated = new_param.clone()
#                 interpolated[:, :d1, ...] = interpolate_tensor(
#                     old_param, new_param[:, :d1, ...], method, alpha_val
#                 )
#                 interpolated_state[name] = interpolated

#             # Case 3: Interpolate flat vectors like fc1_bn1.weight, bias with prefix
#             elif (
#                 old_param.dim() == 1
#                 and old_param.shape[0] < new_param.shape[0]
#                 and torch.is_floating_point(new_param)
#                 and torch.is_floating_point(old_param)
#             ):
#                 if alpha is not None:
#                     alpha_val = alpha
#                 else:
#                     alpha_val = layer_lambdas.get(name, 0.5)

#                 d0 = old_param.shape[0]
#                 interpolated = new_param.clone()
#                 interpolated[:d0] = interpolate_tensor(
#                     old_param, new_param[:d0], method, alpha_val
#                 )
#                 interpolated_state[name] = interpolated

#             else:
#                 # Shape mismatch or non-float → just copy current task parameter
#                 print(f"[KAN interp] Shape/type mismatch for {name}: "
#                       f"{old_param.shape} vs {new_param.shape} → keep new_param")
#                 interpolated_state[name] = new_param.clone()
#         else:
#             # New parameter → keep as is
#             interpolated_state[name] = new_param.clone()

#     interpolated_model = deepcopy(new_model)
#     interpolated_model.load_state_dict(interpolated_state)
#     return interpolated_model


# In models.py

def interpolate_models_KAN(
    old_model,
    new_model,
    alpha=None,
    method='linear',
    lambda_min=0.3,
    lambda_max=0.7,
):
    # ... (Keep helper function interpolate_tensor as is) ...
    def interpolate_tensor(old_tensor, new_tensor, method, alpha_val):
        # ... (same as your original code) ...
        old_np = old_tensor.detach().cpu().numpy()
        new_np = new_tensor.detach().cpu().numpy()

        if method == 'linear':
            interpolated_np = (1 - alpha_val) * old_np + alpha_val * new_np
        elif method == 'polynomial':
            x_vals = [0, 0.5, 1]
            y_vals = np.stack([old_np, (old_np + new_np) / 2, new_np], axis=0)
            interpolated_np = interp1d(
                x_vals, y_vals, axis=0, kind='quadratic', fill_value="extrapolate"
            )(alpha_val)
        elif method == 'spline':
            x_vals = [0, 1]
            y_vals = np.stack([old_np, new_np], axis=0)
            interpolated_np = CubicSpline(x_vals, y_vals, axis=0)(alpha_val)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        return torch.tensor(
            interpolated_np,
            dtype=new_tensor.dtype,
            device=new_tensor.device,
        )

    interpolated_state = {}
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()

    # 1) Compute lambdas
    layer_lambdas = compute_layerwise_lambdas(
        old_model,
        new_model,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )

    # 2) Interpolate
    for name, new_param in new_state.items():
        if name in old_state:
            old_param = old_state[name].to(new_param.device)

            # Case 1: Exact Match (Same shape + Floating Point)
            if (
                old_param.shape == new_param.shape
                and torch.is_floating_point(new_param)
                and torch.is_floating_point(old_param)
            ):
                if alpha is not None:
                    alpha_val = alpha
                else:
                    alpha_val = layer_lambdas.get(name, 0.5)

                interpolated_state[name] = interpolate_tensor(
                    old_param, new_param, method, alpha_val
                )

            # Case 2: Expanded Output (Prefix Interpolation)
            # FIX: Added `and old_param.dim() >= 2` to prevent crash on scalars
            elif (
                old_param.dim() == new_param.dim()
                and old_param.dim() >= 2                  # <--- ADD THIS CHECK
                and old_param.shape[0] == new_param.shape[0]
                and old_param.shape[2:] == new_param.shape[2:]
                and old_param.shape[1] < new_param.shape[1]
                and torch.is_floating_point(new_param)
                and torch.is_floating_point(old_param)
            ):
                if alpha is not None:
                    alpha_val = alpha
                else:
                    alpha_val = layer_lambdas.get(name, 0.5)

                d1 = old_param.shape[1]
                interpolated = new_param.clone()
                
                # Interpolate the matching part
                interpolated[:, :d1, ...] = interpolate_tensor(
                    old_param, new_param[:, :d1, ...], method, alpha_val
                )
                interpolated_state[name] = interpolated

            # Case 3: 1D Vectors (Bias/BatchNorm weights)
            elif (
                old_param.dim() == 1
                and old_param.shape[0] < new_param.shape[0]
                and torch.is_floating_point(new_param)
                and torch.is_floating_point(old_param)
            ):
                if alpha is not None:
                    alpha_val = alpha
                else:
                    alpha_val = layer_lambdas.get(name, 0.5)

                d0 = old_param.shape[0]
                interpolated = new_param.clone()
                interpolated[:d0] = interpolate_tensor(
                    old_param, new_param[:d0], method, alpha_val
                )
                interpolated_state[name] = interpolated

            else:
                # Shape mismatch or non-float (e.g. num_batches_tracked) -> Keep New
                # No print needed usually, or keep it for debug
                interpolated_state[name] = new_param.clone()
        else:
            # New parameter
            interpolated_state[name] = new_param.clone()

    interpolated_model = deepcopy(new_model)
    interpolated_model.load_state_dict(interpolated_state)
    return interpolated_model 



import torch
from copy import deepcopy

@torch.no_grad()
def evaluate_loss_acc(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for inputs, labels in dataloader:
        inputs = inputs.float().to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        # If temporal: outputs [B,T,C] and labels likely [B,T,C] or [B,T]
        if outputs.dim() == 3:
            B, T, C = outputs.shape
            outputs_flat = outputs.reshape(-1, C)

            if labels.dim() == 3:  # one-hot over time [B,T,C]
                labels_flat = labels.reshape(-1, C)
            else:                  # indices over time [B,T]
                labels_flat = labels.reshape(-1)

        else:
            outputs_flat = outputs
            labels_flat = labels

        # ----- Loss: match your training target format -----
        # Case A: one-hot / soft labels -> must be FLOAT and same shape as outputs
        if labels_flat.dim() == 2 and labels_flat.shape == outputs_flat.shape:
            loss = criterion(outputs_flat, labels_flat.float())
            hard_labels = labels_flat.argmax(dim=1)

        # Case B: hard class indices -> must be LONG and shape [N]
        else:
            if labels_flat.dtype != torch.long:
                labels_flat = labels_flat.long()
            loss = criterion(outputs_flat, labels_flat)
            hard_labels = labels_flat

        total_loss += loss.item() * hard_labels.size(0)

        preds = outputs_flat.argmax(dim=1)
        total_correct += (preds == hard_labels).sum().item()
        total_n += hard_labels.size(0)

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)




def _make_layerwise_gammas(layer_lambdas, lambda_min, lambda_max, k=3.0, eps=1e-12):
    denom = max(lambda_max - lambda_min, eps)
    gammas = {}
    for name, lam in layer_lambdas.items():
        lam_norm = (lam - lambda_min) / denom  # in [0,1]
        gammas[name] = 1.0 + k * (1.0 - lam_norm)  # in [1, 1+k]
    return gammas


@torch.no_grad()
def interpolate_state_layerwise_path(model_A, model_B, s,
                                     layer_lambdas,
                                     lambda_min=0.3, lambda_max=0.7,
                                     k=3.0):
    """
    Path between model_A (s=0) and model_B (s=1) with per-parameter schedules:
      alpha_l(s) = s^(gamma_l), gamma_l derived from lambda_l.

    Only interpolates matching floating tensors. Others copied from model_B.
    """
    A = model_A.state_dict()
    B = model_B.state_dict()
    gammas = _make_layerwise_gammas(layer_lambdas, lambda_min, lambda_max, k=k)

    out = {}
    s = float(s)

    for name, Bp in B.items():
        if name in A:
            Ap = A[name].to(Bp.device)
            if (Ap.shape == Bp.shape and torch.is_floating_point(Ap) and torch.is_floating_point(Bp)):
                gamma = gammas.get(name, 1.0)
                alpha = s ** gamma  # 0->1
                out[name] = (1.0 - alpha) * Ap + alpha * Bp
            else:
                out[name] = Bp.clone()
        else:
            out[name] = Bp.clone()

    return out


@torch.no_grad()
def loss_barrier_between_two_finals(model_prev_final, model_curr_final,
                                   dataloader, criterion, device,
                                   compute_layerwise_lambdas_fn,
                                   lambda_min=0.3, lambda_max=0.7,
                                   k=3.0,
                                   num_points=11):
    """
    Barrier along a layerwise-adaptive path between:
      s=0 -> model_prev_final (tilde theta_{t-1})
      s=1 -> model_curr_final (tilde theta_t)
    """
    # lambdas computed FROM the two endpoints you are connecting
    layer_lambdas = compute_layerwise_lambdas_fn(
        model_prev_final, model_curr_final,
        lambda_min=lambda_min, lambda_max=lambda_max
    )

    probe = deepcopy(model_curr_final).to(device)

    s_values = torch.linspace(0.0, 1.0, steps=num_points).tolist()
    curve = []

    for s in s_values:
        sd = interpolate_state_layerwise_path(
            model_prev_final, model_curr_final, s,
            layer_lambdas=layer_lambdas,
            lambda_min=lambda_min, lambda_max=lambda_max,
            k=k
        )
        probe.load_state_dict(sd, strict=True)
        L, A = evaluate_loss_acc(probe, dataloader, criterion, device)
        curve.append({'s': float(s), 'loss': float(L), 'acc': float(A)})

    L0 = curve[0]['loss']
    L1 = curve[-1]['loss']
    Lmax = max(p['loss'] for p in curve)
    barrier = Lmax - 0.5 * (L0 + L1)

    return curve, float(barrier), layer_lambdas
