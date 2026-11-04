import math

import torch
import torch.nn.functional as F


class TransportPrototype:
    """Class-centroid OT with the dynamic marginals in the paper's RGR."""

    def __init__(self, source_weights, cfg):
        self.centers = F.normalize(source_weights.detach(), dim=1).clone()
        self.prior = source_weights.new_full((len(source_weights),), 1 / len(source_weights))
        self.prior_ema = float(cfg.PRIOR_EMA)
        self.prior_mix = float(cfg.PRIOR_RHO)
        self.reg = float(cfg.OT_REG)
        self.iterations = int(cfg.SINKHORN_ITERS)

    @torch.no_grad()
    def transport(self, cost):
        marginal = self.prior_mix * self.prior + (1 - self.prior_mix) / len(self.prior)
        kernel = -cost / self.reg
        log_v = marginal.log()
        for _ in range(self.iterations):
            log_u = -math.log(len(cost)) - torch.logsumexp(kernel + log_v[None, :], dim=1)
            log_v = marginal.log() - torch.logsumexp(kernel + log_u[:, None], dim=0)
        return (kernel + log_u[:, None] + log_v[None, :]).exp()

    def loss(self, weak, other, logits):
        with torch.no_grad():
            self.prior.lerp_(logits.detach().softmax(1).mean(0), 1 - self.prior_ema)
        if len(weak) == 0:
            return weak.sum()
        weak = F.normalize(weak, dim=1)
        strong = F.normalize(other, dim=1)
        centers = self.centers.clone()
        cost_weak = 1 - weak @ centers.t()
        cost_strong = 1 - strong @ centers.t()
        plan_weak = self.transport(cost_weak.detach())
        plan_strong = self.transport(cost_strong.detach())
        loss = (plan_weak * cost_strong + plan_strong * cost_weak).sum()
        with torch.no_grad():
            mass = (plan_weak + plan_strong).sum(0)
            means = (plan_weak.t() @ weak + plan_strong.t() @ strong) / mass[:, None].clamp_min(1e-12)
            means = torch.where(mass[:, None] > 1e-12, means, centers)
            self.centers = F.normalize(0.99 * centers + 0.01 * means, dim=1)
        return loss
