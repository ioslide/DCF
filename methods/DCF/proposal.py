import math
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as log

from core.model.build import split_up_model, ImageNetXMaskingLayer
from methods.DCF.MCL import TransportPrototype
from methods.DCF.fourieraug import GeneralFourierOnline
from methods.DCF.transformers_cotta import get_tta_transforms

__all__ = ["setup"]


def gaussian_lowpass_mask(h, w, sigma_f, device, dtype):
    fy = torch.fft.fftfreq(h, d=1.0, device=device).reshape(h, 1)
    fx = torch.fft.fftfreq(w, d=1.0, device=device).reshape(1, w)
    f2 = fy**2 + fx**2
    mask = torch.exp(-0.5 * f2 / (sigma_f**2 + 1e-12))
    return mask.to(dtype=dtype)

def lowpass_filter(x, sigma_f=0.1):
    B, C, H, W = x.shape
    X = torch.fft.fft2(x, dim=(-2, -1))
    M = gaussian_lowpass_mask(H, W, sigma_f, x.device, X.dtype).view(1,1,H,W)
    Xf = X * M
    x_lp = torch.fft.ifft2(Xf, dim=(-2, -1)).real
    return x_lp

def softmax_clamp(logits: torch.Tensor) -> torch.Tensor:
    """Apply softmax to logits and clamp the probabilities.
    """
    logits = F.softmax(logits, dim=1)
    logits = torch.clamp(logits, min=0.0, max=0.99)
    return logits

class SupSoftLikelihoodRatio(nn.Module):
    def __init__(self, gamma=1e-5):
        super(SupSoftLikelihoodRatio, self).__init__()
        self.gamma = gamma
        self.eps=1e-5

    def __call__(self, logits, target_logits):
        logits = softmax_clamp(logits)
        target_logits = softmax_clamp(target_logits)

        return - (logits * torch.log(
            (target_logits  * (1- self.gamma)) / ((1 - target_logits) + self.eps)  + self.gamma
        ) / (1- self.gamma)).sum(1)

@torch.jit.script
def consistency(x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * y.log_softmax(1)).sum(1)

@torch.jit.script
def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute the entropy of the softmax probabilities.
    """
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)

def configure_model(model: nn.Module, cfg):
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()
            module.requires_grad_(True)
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            module.requires_grad_(True)
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.requires_grad_(True)
    return model

def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if 'encoder_layer_9' in nm:
            continue
        if 'encoder_layer_10' in nm:
            continue
        if 'encoder_layer_11' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


class DCF(nn.Module):
    """DCF with aligned geometry repair and curvature-aware retention."""

    def __init__(self, cfg, model: nn.Module):
        """Initialize the DCF adaptation method.
        """
        super().__init__()
        params, param_names = collect_params(model)
        log.info(f"==>> param_names:  {param_names}")
        self.params = params
        self.param_names = param_names
        self.clr_mode = str(cfg.ADAPTER.DCF.CLR_MODE)
        self.clr_eps = float(cfg.ADAPTER.DCF.CLR_EPS)
        self.clr_after_step = bool(cfg.ADAPTER.DCF.CLR_AFTER_STEP)
        self.rgr_weight = float(cfg.ADAPTER.DCF.RGR_WEIGHT)
        self.output_weak_weight = float(cfg.ADAPTER.DCF.OUTPUT_WEAK_WEIGHT)
        self.output_space = str(cfg.ADAPTER.DCF.OUTPUT_SPACE)
        if self.clr_mode != "paper" or cfg.ADAPTER.DCF.RGR_MODE != "aligned":
            raise ValueError("This DCF release uses CLR_MODE=paper and RGR_MODE=aligned")
        if cfg.OPTIM.METHOD != "SGD":
            raise ValueError("The default DCF profile requires SGD")
        if self.clr_eps <= 0:
            raise ValueError("DCF.CLR_EPS must be positive")
        if not 0 <= self.output_weak_weight <= 1 or self.output_space not in {"logits", "probs"}:
            raise ValueError("Invalid DCF prediction ensemble configuration")
        model.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        if cfg.MODEL.ARCH == "resnet50_gn":
            lr = (cfg.OPTIM.LR / 64) * cfg.TEST.BATCH_SIZE * 2 if cfg.TEST.BATCH_SIZE < 32 else cfg.OPTIM.LR
        elif cfg.MODEL.ARCH == "vit_b_16":
            lr = (0.001 / 64) * cfg.TEST.BATCH_SIZE
        elif cfg.MODEL.ARCH == "resnet50":
            lr = (cfg.OPTIM.LR / 64) * cfg.TEST.BATCH_SIZE * 2 if cfg.TEST.BATCH_SIZE < 32 else cfg.OPTIM.LR
        else:
            lr = cfg.OPTIM.LR

        if cfg.TEST.BATCH_SIZE == 1:
            lr = 2 * lr

        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(cfg.OPTIM.MOMENTUM),
            weight_decay=float(cfg.OPTIM.WD)
        )
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, "DCF requires >= 1 step(s) to forward and update"

        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())
        self.MU = self.cfg.ADAPTER.DCF.MU
        self.trust_loss = {"slr": SupSoftLikelihoodRatio(0.001),
                           "softent": consistency}[cfg.ADAPTER.DCF.TRUST_LOSS]

        self.batch_index = 0
        self.entropy_margin = math.log(self.cfg.CORRUPTION.NUM_CLASS) * 0.6
        self.feature_extractor, self.classifier = split_up_model(
            self.model, cfg.MODEL.ARCH, cfg.CORRUPTION.DATASET
        )
        self.sigma_f = self.cfg.ADAPTER.DCF.SIGMA_F

        self.transforms = get_tta_transforms(
            cfg=cfg,
            padding_mode="reflect",
            cotta_augs=True,
            soft=bool(cfg.ADAPTER.DCF.SOFT_AUG),
        )

        freq_lo, freq_hi = map(int, cfg.ADAPTER.DCF.FREQ_RANGE)
        if not 1 <= freq_lo <= freq_hi:
            raise ValueError("DCF.FREQ_RANGE must contain positive increasing bounds")
        if cfg.ADAPTER.DCF.LAMBDA <= 0 or cfg.ADAPTER.DCF.PROBE_DRAWS < 1:
            raise ValueError("DCF probe strength and draw count must be positive")
        self.strong_transforms = GeneralFourierOnline(
            img_size=cfg.INPUT.SIZE[0],
            groups=range(freq_lo, freq_hi + 1),
            phases=(0.0, 1.0),
            f_cut=int(cfg.ADAPTER.DCF.PROBE_DRAWS),
            phase_cut=1,
            min_str=float(cfg.ADAPTER.DCF.MIN_STR),
            mean_str=32.0 * float(cfg.ADAPTER.DCF.LAMBDA),
            granularity=448
        )

        self.prototype = self._make_prototype()
        self.device = next(self.model.parameters()).device
        self.theta_0 = self.model_state
        self.FIM_0 = None
        self.fim_clip_value = torch.tensor(0.0001)

    def _make_prototype(self):
        classifier = next(m for m in self.classifier.modules() if isinstance(m, nn.Linear))
        weights = classifier.weight
        for module in self.classifier.modules():
            if isinstance(module, ImageNetXMaskingLayer):
                weights = weights[module.mask]
        return TransportPrototype(weights, self.cfg.ADAPTER.DCF)

    def forward(self, x, y=None, adapt=True):
        if adapt:
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x)
            if self.output_weak_weight > 0:
                weak = self._last_weak_logits
                if self.output_space == "probs":
                    outputs, weak = outputs.softmax(1), weak.softmax(1)
                outputs = (1 - self.output_weak_weight) * outputs + self.output_weak_weight * weak
        else:
            with torch.no_grad():
                outputs = self.model(x)

        self.batch_index += 1
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        total_loss = 0
        x_aug = self.transforms(x)
        x_strong = self.strong_transforms(x_aug)
        strong_feats = self.feature_extractor(x_strong)

        x_aug = lowpass_filter(x_aug, sigma_f=self.sigma_f)
        weak_feats = self.feature_extractor(x_aug)
        orig_feats = self.feature_extractor(x)

        weak_logits = self.classifier(weak_feats)
        strong_logits = self.classifier(strong_feats)
        orig_logits = self.classifier(orig_feats)
        self._last_weak_logits = weak_logits.detach()

        with torch.no_grad():
            _logits_A = orig_logits
            _logits_B = strong_logits
            entropy = softmax_entropy(_logits_A.detach())
            _entropy_idx = torch.where(entropy < self.entropy_margin)[0]

            prob_outputs = _logits_A[_entropy_idx].softmax(1)
            prob_outputs_strong = _logits_B[_entropy_idx].softmax(1)
            cls1 = prob_outputs.argmax(dim=1)
            pcs = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_strong, dim=1, index=cls1.reshape(-1,1))
            pcs = pcs.reshape(-1)
            pcs = pcs.abs()
            _pcs_idx = torch.where(pcs > self.cfg.ADAPTER.DCF.PCS_T)[0]
            _fillter_idx = _entropy_idx[_pcs_idx]

            mask = torch.zeros_like(entropy, dtype=torch.bool)
            mask[_fillter_idx] = True

        # Check before reducing the trusted loss: an empty mean is NaN.
        if not mask.any():
            log.debug("DCF: no trusted samples; skipping this adaptation update")
            return orig_logits
        pcs_weights = self.get_pcs_weights(_logits_B, _logits_A)
        weights = torch.exp(pcs_weights) ** self.cfg.ADAPTER.DCF.PCS_W
        trust_loss = ((
            self.trust_loss(orig_logits, orig_logits)[mask] * weights[mask] \
            + self.trust_loss(orig_logits, weak_logits)[mask] * weights[mask] \
            + self.trust_loss(orig_logits, strong_logits)[mask] * weights[mask]
        )).mean(0)

        total_loss += trust_loss
        ot_loss = self.prototype.loss(weak_feats[~mask], strong_feats[~mask], orig_logits)
        total_loss += self.rgr_weight * ot_loss
        if not torch.isfinite(total_loss):
            log.warning("DCF: nonfinite adaptation loss; skipping update")
            return orig_logits

        total_loss.backward()
        if self.cfg.ADAPTER.DCF.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(self.params, self.cfg.ADAPTER.DCF.GRAD_CLIP, error_if_nonfinite=True)
        if not self.clr_after_step:
            self.param_robustness(x)
        self.optimizer.step()
        if self.clr_after_step:
            self.param_robustness(x)

        return orig_logits

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.batch_index = 0
        self.model.zero_grad(set_to_none=True)
        self.sigma_f = self.cfg.ADAPTER.DCF.SIGMA_F

        self.prototype = self._make_prototype()

        self.FIM_0 = None


        log.info("DCF model, optimizer, and other states have been reset.")

    def get_pcs_weights(self, strong_logits: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prob_outputs = logits.softmax(1)
            cls1 = prob_outputs.argmax(dim=1)
            prob_outputs_strong = strong_logits.softmax(1)
            strong_pcs = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_strong, dim=1, index=cls1.reshape(-1,1))
            pcs = strong_pcs
            pcs = pcs.reshape(-1)

        pcs = (pcs - pcs.min()) / (pcs.max() - pcs.min()).clamp_min(1e-12)
        return pcs

    def param_robustness(self, x=None):
        FIM_t = defaultdict(lambda: torch.tensor(0.0, device=self.device))
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                FIM_t[name] = torch.min(param.grad.square(), self.fim_clip_value)
        if self.FIM_0 is None:
            self.FIM_0 = deepcopy(FIM_t)
            return

        group_statistics = {}
        layer_groups = {}
        for layer_name, layer_module in self.model.named_modules():
            if isinstance(layer_module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                squared_l2_dist = torch.tensor(0.0, device=self.device)
                num_params_in_layer = 0

                for param_name, _ in layer_module.named_parameters():
                    full_param_name = f"{layer_name}.{param_name}"
                    if full_param_name in self.FIM_0 and full_param_name in FIM_t:
                        fim_0_val = self.FIM_0[full_param_name]
                        fim_t_val = FIM_t[full_param_name]

                        diff = fim_t_val - fim_0_val
                        squared_l2_dist += torch.sum(diff.pow(2))
                        num_params_in_layer += fim_t_val.numel()

                if num_params_in_layer > 0:
                    group_name = layer_name
                    if self.cfg.ADAPTER.DCF.GRANULARITY == "stage":
                        group_name = next((part for part in layer_name.split('.')
                                           if part in ('layer1', 'layer2', 'layer3', 'layer4')), layer_name)
                    layer_groups[layer_name] = group_name
                    if group_name not in group_statistics:
                        group_statistics[group_name] = [squared_l2_dist, num_params_in_layer]
                    else:
                        group_statistics[group_name][0] += squared_l2_dist
                        group_statistics[group_name][1] += num_params_in_layer

        group_gates = {}
        for group_name, (distance, dimension) in group_statistics.items():
            if self.cfg.ADAPTER.DCF.CLR_GATE_NORMALIZE:
                distance = distance / dimension
            distance = distance * self.cfg.ADAPTER.DCF.MU_TEMP
            group_gates[group_name] = torch.exp(-distance) * self.MU

        for layer_name, layer_module in self.model.named_modules():
            if layer_name in layer_groups:
                mu = group_gates[layer_groups[layer_name]]
                for param_name, param in layer_module.named_parameters():
                    if not param.requires_grad:
                        continue

                    full_param_name = f"{layer_name}.{param_name}"
                    if full_param_name in self.FIM_0 and full_param_name in FIM_t:
                        theta_0_val = self.theta_0[full_param_name].to(self.device)
                        fim_0_val = self.FIM_0[full_param_name]
                        theta_t_val = param.data
                        fim_t_val = FIM_t[full_param_name]
                        current = fim_t_val + self.clr_eps
                        source = fim_0_val + self.clr_eps
                        alpha = mu * current / (mu * current + (1 - mu) * source)
                        stabilized_theta = theta_0_val + alpha * (theta_t_val - theta_0_val)
                        param.data = stabilized_theta.to(param.dtype)


def setup(model: nn.Module, cfg):
    log.info("Setup TTA method: DCF")
    return DCF(cfg, configure_model(model, cfg))
