import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from scipy.interpolate import interp1d, CubicSpline

class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        # --------------------------------------------------
        # Dataset-specific input dimensionality
        # --------------------------------------------------
        if config.dataset == "EMBER":
            self.input_features = 2381
        elif config.dataset == "AZ":
            # AZ Class-IL vs Domain-IL differ in preprocessing
            self.input_features = 2439 if config.scenario == "class" else 1789
        else:
            raise ValueError(f"Unknown dataset {config.dataset}")

        # --------------------------------------------------
        # Output dimensionality
        # --------------------------------------------------
        self.output_dim = config.init_classes
        self.drop_prob = 0.5

        # --------------------------------------------------
        # Normalization choice (CRITICAL)
        # --------------------------------------------------
        if config.scenario == "class":
            Norm1 = lambda c: nn.BatchNorm1d(c)
            Norm2 = lambda c: nn.BatchNorm1d(c)
        else:  # Domain-IL
            Norm1 = lambda c: nn.LayerNorm([c, 1])
            Norm2 = lambda c: nn.LayerNorm([c, 1])

        # --------------------------------------------------
        # Backbone
        # --------------------------------------------------
        self.block1 = nn.Sequential(
            nn.Conv1d(self.input_features, 512, kernel_size=3, stride=3, padding=1),
            Norm1(512),
            nn.ReLU(),

            nn.Conv1d(512, 256, kernel_size=3, stride=3, padding=1),
            Norm1(256),
            nn.Dropout(self.drop_prob),
            nn.ReLU(),

            nn.MaxPool1d(3, 3, 1)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, stride=2, padding=1),
            Norm2(128),
            nn.Dropout(self.drop_prob),
            nn.ReLU()
        )

        # --------------------------------------------------
        # Classifier head
        # --------------------------------------------------
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, self.output_dim)

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------
    def forward(self, x):
        original_shape = x.size()

        # Handle [B, T, F] case
        if len(original_shape) == 3:
            x = x.view(original_shape[0] * original_shape[1], self.input_features)

        x = x.view(x.size(0), self.input_features, -1)

        x = self.block1(x)
        x = self.block2(x)

        x = self.flatten(x)
        logits = self.fc(x)

        if len(original_shape) == 3:
            logits = logits.view(original_shape[0], original_shape[1], -1)

        return logits

    # ------------------------------------------------------
    # Class-IL ONLY
    # ------------------------------------------------------
    def expand_output_layer(self, init_classes, n_inc, task):
        """
        Expands classifier head for Class-IL.
        Safe no-op if output size does not change.
        """
        new_dim = init_classes + n_inc * task
        if new_dim == self.output_dim:
            return self

        old_fc = self.fc
        self.output_dim = new_dim
        self.fc = nn.Linear(old_fc.in_features, self.output_dim)

        with torch.no_grad():
            self.fc.weight[:old_fc.out_features].copy_(old_fc.weight)
            self.fc.bias[:old_fc.out_features].copy_(old_fc.bias)

        return self



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

    return layer_lambdas



def interpolate_models(old_model, new_model,
                       Lambda=None,        # optional global fallback
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
        Lambda (float or None): Optional global interpolation weight; if None,
                               use layerwise λ_l only.
        method (str): 'linear', 'polynomial', or 'spline'.
        lambda_min, lambda_max: Range for λ_l.

    Returns:
        nn.Module: Interpolated model.
    """

    def interpolate_tensor(old_tensor, new_tensor, method, Lambda_val):
        """
        Interpolates between two tensors using the specified method.
        """
        old_np = old_tensor.detach().cpu().numpy()
        new_np = new_tensor.detach().cpu().numpy()

        if method == 'linear':
            interpolated_np = (1 - Lambda_val) * old_np + Lambda_val * new_np

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
            interpolated_np = poly_interp(Lambda_val)

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
            interpolated_np = spline(Lambda_val)

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

    # 2) Interpolate with per-layer λ_l (or global Lambda fallback)
    for name, new_param in new_state.items():
        if name in old_state:
            old_param = old_state[name].to(new_param.device)

            if old_param.shape == new_param.shape and torch.is_floating_point(new_param):
                # Choose layer-specific λ_l if available; else fallback
                if Lambda is not None:
                    Lambda_val = Lambda
                else:
                    Lambda_val = layer_lambdas.get(name, 0.5)  # default if missing
               

                interpolated_state[name] = interpolate_tensor(
                    old_param, new_param, method, Lambda_val
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
