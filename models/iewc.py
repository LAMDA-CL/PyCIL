"""
IEWC: Improved Elastic Weight Consolidation.

D. Wiest, "Improved Elastic Weight Consolidation as an Optimization Constraint
for Continual Learning". https://arxiv.org/abs/2611.16039

IEWC keeps the EWC penalty (lamda / 2) * sum_i F_i (theta_i - theta*_i)^2 and
training loop from models/ewc.py and only changes how the diagonal importance
F is estimated. EWC's empirical Fisher sums squared loss gradients g_n, so each
old sample counts in proportion to its loss-gradient scale and well-fitted
samples barely count. IEWC divides each per-sample term by that scale:

    F = (1/N) sum_n h_n * g_n**2 / (s_n + tau),   s_n = ||r_n||^2,

where r_n is the gradient of the per-sample loss w.r.t. the logits,
tau = kappa * median{s_n : s_n > 0} is a task-relative damping, and h_n are
aggregation weights with mean one ("uniform": h_n = 1, "rdw": residual
diversity weighting, see ResidualDiversityWeights, or a callable
(x, y, flat_grad) -> non-negative weight, mean-normalized). The estimate is invariant
to a global rescaling of the loss.

Differences from models/ewc.py:
- per-sample gradients in eval mode (EWC uses batch gradients in train mode),
- no fishermax clamp (it is scale dependent),
- its own default lamda, since F has a different scale than EWC's. Without
  the clamp, the penalty curvature is lamda * F, so a large lamda can make SGD
  with lrate 0.1 diverge (lamda 100 does on CIFAR-100 with resnet32).

The importance pass caches the current task's (augmented) inputs on the CPU so
that both passes see identical images.

Hyper-parameters below are defaults; each can be overridden from the json
config with the same key. "output_metric" can also be given (when building the
learner from Python) as a callable mapping a batch of logit gradients r of
shape (B, C) to s of shape (B,), i.e. s_n = r_n^T G_n^{-1} r_n for a metric G_n.
"""

import logging
import torch
from torch.nn import functional as F
from models.ewc import EWC

lamda = 1
kappa = 1e-2
weighting = "uniform"  # "uniform", "rdw", or a callable (x, y, flat_grad) -> weight
rdw_max_basis = 64


def relative_damping(scales, kappa):
    """tau = kappa * median of the positive scales (0 if there are none)."""
    positive = scales[scales > 0]
    if positive.numel() == 0:
        return 0.0
    return kappa * torch.quantile(positive.double(), 0.5).item()


class ResidualDiversityWeights:
    """Streaming RDW weights h_n = ||(I - Q^T Q) u_n||^2 with u_n = g_n / ||g_n||.

    Q holds at most max_basis orthonormal residual directions of earlier samples
    (a residual is appended while the cap is not reached). Weights are returned
    unnormalized; the caller divides by their sum (mean normalization).
    """

    def __init__(self, max_basis=64, eps=1e-12):
        self.max_basis = max_basis
        self.eps = eps
        self.basis = None
        self.size = 0

    def __call__(self, grad):
        norm = grad.norm()
        if norm == 0:
            return 0.0
        residual = grad / norm
        if self.basis is None:
            # keep the basis off the GPU if it would take more than 1 GiB
            nbytes = self.max_basis * grad.numel() * grad.element_size()
            device = grad.device if nbytes <= 2**30 else "cpu"
            self.basis = torch.zeros(self.max_basis, grad.numel(), device=device)
        residual = residual.to(self.basis.device)
        if self.size > 0:
            q = self.basis[: self.size]
            residual = residual - q.t() @ (q @ residual)
        weight = residual.pow(2).sum().item()
        if weight > self.eps and self.size < self.max_basis:
            self.basis[self.size] = residual / weight**0.5
            self.size += 1
        return weight


class IEWC(EWC):
    def __init__(self, args):
        super().__init__(args)
        self.lamda = args.get("lamda", lamda)
        self.kappa = args.get("kappa", kappa)
        self.weighting = args.get("weighting", weighting)
        self.rdw_max_basis = args.get("rdw_max_basis", rdw_max_basis)
        self.output_metric = args.get("output_metric", None)
        assert callable(self.weighting) or self.weighting in (
            "uniform",
            "rdw",
        ), "Unknown weighting."

    def per_sample_loss(self, logits, targets):
        return F.cross_entropy(logits, targets, reduction="none")

    def output_scale(self, output_grad):
        if self.output_metric is not None:
            return self.output_metric(output_grad)
        return output_grad.pow(2).sum(dim=1)

    def getFisherDiagonal(self, train_loader):
        params = {
            n: p for n, p in self._network.named_parameters() if p.requires_grad
        }
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}
        was_training = self._network.training
        self._network.eval()

        # Pass 1: scales s_n from the logit gradients, and the damping tau.
        cache, scales = [], []
        for _, inputs, targets in train_loader:
            with torch.no_grad():
                logits = self._network(inputs.to(self._device))["logits"]
            logits.requires_grad_(True)
            loss = self.per_sample_loss(logits, targets.to(self._device)).sum()
            output_grad = torch.autograd.grad(loss, logits)[0]
            scales.append(self.output_scale(output_grad).detach().cpu())
            cache.append((inputs, targets))
        scales = torch.cat(scales)
        tau = relative_damping(scales, self.kappa)

        # Pass 2: per-sample parameter gradients on the same inputs.
        rdw = None
        if self.weighting == "rdw":
            rdw = ResidualDiversityWeights(self.rdw_max_basis)
        weights = []
        n = 0
        for inputs, targets in cache:
            for x, y in zip(inputs, targets):
                denom = scales[n].item() + tau
                n += 1
                self._network.zero_grad(set_to_none=True)
                logits = self._network(x.unsqueeze(0).to(self._device))["logits"]
                loss = self.per_sample_loss(logits, y.view(1).to(self._device))
                loss.sum().backward()
                if rdw is not None or callable(self.weighting):
                    flat = torch.cat(
                        [
                            p.grad.reshape(-1)
                            if p.grad is not None
                            else torch.zeros_like(p).reshape(-1)
                            for p in params.values()
                        ]
                    )
                    if rdw is not None:
                        weight = rdw(flat)
                    else:
                        weight = float(self.weighting(x, y, flat))
                else:
                    weight = 1.0
                weights.append(weight)
                if denom <= 0 or weight == 0:
                    continue
                for name, p in params.items():
                    if p.grad is not None:
                        fisher[name] += (weight / denom) * p.grad.pow(2)
        self._network.zero_grad(set_to_none=True)
        self._network.train(was_training)

        weights = torch.tensor(weights)
        total = weights.sum().item()
        if total > 0:
            for name in fisher:
                fisher[name] /= total
            weights = weights / weights.mean()
        self.importance_tau = tau
        self.importance_scales = scales
        self.importance_weights = weights
        logging.info(
            "IEWC importance: {} samples, tau {:.3e}, median scale {:.3e}".format(
                len(scales), tau, scales.median().item()
            )
        )
        return fisher
