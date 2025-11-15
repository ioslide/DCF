import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os

from copy import deepcopy
from loguru import logger as log
from collections import defaultdict
from core.model.build import split_up_model
from methods.DCF.MCL import Prototype
from methods.DCF.fourieraug import GeneralFourierOnline
from methods.DCF.transformers_cotta import get_tta_transforms
import torchvision
from einops import rearrange
from core.model.build import split_up_model, ResNetDomainNet126
from torch.nn.utils.weight_norm import WeightNorm

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

@torch.no_grad()
def ema_update_model(model_to_update, model_to_merge, momentum, update_all=False):
    if momentum < 1.0:
        for param_to_update, param_to_merge in zip(model_to_update.parameters(), model_to_merge.parameters()):
            if param_to_update.requires_grad or update_all:
                param_to_update.data = momentum * param_to_update.data + (1 - momentum) * param_to_merge.data.cuda()
    return model_to_update

def pairwise_cosine_sim(a, b):
    assert len(a.shape) == 2
    assert a.shape[1] == b.shape[1]
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    mat = a @ b.t()
    return mat

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

def copy_model(model):
    if isinstance(model, ResNetDomainNet126):  # https://github.com/pytorch/pytorch/issues/28594
        for module in model.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, WeightNorm):
                    if hasattr(module, hook.name):
                        delattr(module, hook.name)
        coppied_model = deepcopy(model)
        for module in model.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, WeightNorm):
                    hook(module, None)
    else:
        coppied_model = deepcopy(model)
    return coppied_model
    
@torch.jit.script
def consistency(x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * y.log_softmax(1)).sum(1)

@torch.jit.script
def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute the entropy of the softmax probabilities.
    """
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)

def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Compute the Gaussian kernel between two tensors.
    """
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return torch.exp(-((x - y) ** 2).sum(2) / (2 * sigma ** 2))

def gauss_kl_divergence(mean1, var1, mean2, var2, eps=1e-3):
    d1 = (torch.log(var2 + eps) - torch.log(var1 + eps))/2. + \
        (var1 + eps + (mean1 - mean2)**2) / 2. / (var2 + eps) - 0.5
    return d1

def copy_model_and_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer):
    """Create deep copies of the model and optimizer states.
    """
    model_state = deepcopy(model.state_dict())
    src_model = copy_model(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    for param in src_model.parameters():
        param.detach_()
    return model_state, optimizer_state, src_model

def load_model_and_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer,
                             model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

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

def compute_input_gradients(model, imgs):
    imgs.requires_grad = True
    logits = model(imgs)
    entropies = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    _entropy_idx = torch.where(entropies < math.log(1000) * 0.4)[0]
    entropies = entropies[_entropy_idx]
    loss = entropies.mean(0)
    input_gradients = torch.autograd.grad(outputs=loss, inputs=imgs, create_graph=True)[0].detach()
    imgs.requires_grad = False
    model.zero_grad()
    return input_gradients, entropies, logits

class DCF(nn.Module):
    """Prototype Rectification (DCF) adaptation method implementation."""

    def __init__(self, cfg, model: nn.Module):
        """Initialize the DCF adaptation method.
        """
        super().__init__()
        params, param_names = collect_params(model)
        log.info(f"==>> param_names:  {param_names}")
        self.params = params
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

        if cfg.OPTIM.METHOD == "SGD":
            optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=float(cfg.OPTIM.MOMENTUM),
                weight_decay=float(cfg.OPTIM.WD)
            )
        elif cfg.OPTIM.METHOD == "Adam":
            optimizer = torch.optim.Adam(
                params,
                lr=lr,
                weight_decay=float(cfg.OPTIM.WD)
            )
        else:
            raise ValueError(f"Invalid optimizer method: DCF")

        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, "DCF requires >= 1 step(s) to forward and update"
                        
        self.model_state, self.optimizer_state, self.src_model = copy_model_and_optimizer(self.model, self.optimizer)
        self.hidden_model = copy_model(self.model)
        self.hidden_model.train()

        self.ema_model = copy_model(self.model)
        self.ema_model.train()
        self.MU = self.cfg.ADAPTER.DCF.MU
        self.EMA_TRIGGER = False
        self.sup_slr = SupSoftLikelihoodRatio(0.001)

        self.eps = 1e-6
        self.batch_index = 0
        self.batch_size = cfg.TEST.BATCH_SIZE
        self.entropy_margin = math.log(self.cfg.CORRUPTION.NUM_CLASS) * 0.6
        self.feature_extractor, self.classifier = split_up_model(
            self.model, cfg.MODEL.ARCH, cfg.CORRUPTION.DATASET
        )
        self.sigma_f = self.cfg.ADAPTER.DCF.SIGMA_F

        self.current_model_probs = None
        self.transforms = get_tta_transforms(
            cfg=cfg,
            padding_mode="reflect",
            cotta_augs=True
        )

        self.strong_transforms = GeneralFourierOnline(
            img_size=cfg.INPUT.SIZE[0],
            groups=range(1, 225),
            phases=(0.0, 1.0),
            f_cut=1,
            phase_cut=1,
            min_str=0,
            mean_str=5,
            granularity=448
        )

        self.prototype = Prototype(
            C=self.cfg.CORRUPTION.NUM_CLASS,
            dim= 256 if cfg.CORRUPTION.DATASET == "domainnet126" else cfg.MODEL.PROJECTION.EMB_DIM,
        )
        self.param_group_names = []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.param_group_names.append(name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.theta_0 = deepcopy(self.model.state_dict())
        self.FIM_0 = None
        self.fim_clip_value = torch.tensor(0.0001)
        
        self.layer_names_to_track = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                 self.layer_names_to_track.append(name)
        log.info(f"Tracking layer-wise drift for: {self.layer_names_to_track}")

    def forward(self, x, y=None, adapt=True):
        if adapt:
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x)
        else:
            with torch.no_grad():
                outputs = self.model(x)

        self.batch_index += 1
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        if 'using gradient entropy' in self.cfg.NOTE:
            input_gradients, entropies, logits = compute_input_gradients(self.model, x)

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

        with torch.no_grad():
            self.prototype.update(
                orig_logits,
                orig_feats,
                momentum = 0.99,
                norm=True
            )
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
            if 'lower_pcs_type' in self.cfg.NOTE:
                _pcs_idx = torch.where(pcs < self.cfg.ADAPTER.DCF.PCS_T)[0]
            elif 'mean_pcs_type' in self.cfg.NOTE:
                _pcs_idx = torch.where(pcs < pcs.mean())[0]
            else:
                _pcs_idx = torch.where(pcs > self.cfg.ADAPTER.DCF.PCS_T)[0]
            _fillter_idx = _entropy_idx[_pcs_idx]

            mask = torch.zeros_like(entropy, dtype=torch.bool)
            mask[_fillter_idx] = True
            
        pcs_weights = self.get_pcs_weights(_logits_B, _logits_A)
        weights = torch.exp(pcs_weights) ** self.cfg.ADAPTER.DCF.PCS_W
        slr_loss = ((
            self.sup_slr(orig_logits, orig_logits)[mask] * weights[mask] \
            + self.sup_slr(orig_logits, weak_logits)[mask] * weights[mask] \
            + self.sup_slr(orig_logits, strong_logits)[mask] * weights[mask]
        )).mean(0)

        total_loss += slr_loss 
        if len(_fillter_idx) == 0:
            log.warning("All samples are unreliable")
            if 'zeros_mask' in self.cfg.NOTE:
                mask = torch.zeros_like(entropy, dtype=torch.bool)
            elif 'ones_mask' in self.cfg.NOTE:
                mask = torch.ones_like(entropy, dtype=torch.bool)
            else:
                return orig_logits
        
        Q_st_weak = self.prototype.OT(
            self.prototype.feats_center, 
            weak_feats
        )
        Q_st_strong = self.prototype.OT(
            self.prototype.feats_center, 
            strong_feats
        )

        ot_loss1 = self.prototype.center_loss_cls(
            self.prototype.feats_center,
            strong_feats,
            Q_st = Q_st_weak,
            mask = mask,
            weights = pcs_weights,
            note = self.cfg.NOTE
        )
        ot_loss2 = self.prototype.center_loss_cls(
            self.prototype.feats_center,
            weak_feats,
            Q_st = Q_st_strong,
            mask = mask,
            weights = pcs_weights,
            note = self.cfg.NOTE
        )
        ot_loss = (ot_loss1 + ot_loss2) / 2
        total_loss += ot_loss
            
        total_loss.backward()
        self.param_robustness()
        self.optimizer.step()

        return orig_logits
    
    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.batch_index = -1
        self.sigma_f = self.cfg.ADAPTER.DCF.SIGMA_F
        
        self.prototype = Prototype(
            C=self.cfg.CORRUPTION.NUM_CLASS,
            dim= 256 if self.cfg.CORRUPTION.DATASET == "domainnet126" else self.cfg.MODEL.PROJECTION.EMB_DIM,
        )

        self.FIM_0 = None

        self.ema_model.load_state_dict(self.model_state)
        self.hidden_model.load_state_dict(self.model_state)

        log.info("DCF model, optimizer, and other states have been reset.")

    def get_pcs_weights(self, strong_logits: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prob_outputs = logits.softmax(1)
            cls1 = prob_outputs.argmax(dim=1)
            prob_outputs_strong = strong_logits.softmax(1)
            strong_pcs = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_strong, dim=1, index=cls1.reshape(-1,1))
            pcs = strong_pcs
            pcs = pcs.reshape(-1)

        pcs = (pcs - pcs.min()) / (pcs.max() - pcs.min())
        return pcs

    def param_robustness(self):
        FIM_t = defaultdict(lambda: torch.tensor(0.0, device=self.device))
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                FIM_t[name] = torch.pow(param.grad, 2)

        for name, fim_value in FIM_t.items():
            FIM_t[name] = torch.min(fim_value, self.fim_clip_value)

        if self.batch_index == 0:
            self.FIM_0 = deepcopy(FIM_t)
            return

        if self.FIM_0 is None:
            return
            
        layer_shift_scores = {}
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
                        num_params_in_layer += 1

                if num_params_in_layer > 0:
                    mu = torch.exp(-squared_l2_dist)
                    layer_shift_scores[layer_name] = mu

        for layer_name, layer_module in self.model.named_modules():
            if layer_name in layer_shift_scores:
                mu = layer_shift_scores[layer_name] * self.MU
                # layer_epsilon_par_t = 0.0
                for param_name, param in layer_module.named_parameters():
                    if not param.requires_grad:
                        continue
                        
                    full_param_name = f"{layer_name}.{param_name}"
                    if full_param_name in self.FIM_0 and full_param_name in FIM_t:
                        with torch.no_grad():
                            theta_0_val = self.theta_0[full_param_name].to(self.device)
                            theta_t_prime_val = param.data
                            fim_0_val = self.FIM_0[full_param_name]
                            # #  ||theta'_t - theta_0||^2_{F_0}
                            # param_dist_sq_fim = torch.sum(((theta_t_prime_val - theta_0_val) ** 2) * fim_0_val)
                            # param_contribution = 0.5 * ((1 - mu) ** 2) * param_dist_sq_fim
                            # layer_epsilon_par_t += param_contribution.item()
                        theta_t_val = param.data
                        fim_t_val = FIM_t[full_param_name]
                        numerator = mu * fim_t_val * theta_t_val + (1 - mu) * fim_0_val * theta_0_val
                        denominator = mu * fim_t_val + (1 - mu) * fim_0_val
                        stabilized_theta = numerator / (denominator + 1e-8)
                        param.data = stabilized_theta.to(param.dtype)

def setup(model: nn.Module, cfg):
    log.info("Setup TTA method: DCF")
    model = configure_model(model, cfg)
    tta_model = DCF(
        cfg,
        model
    )
    return tta_model