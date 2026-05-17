import torch
from torch import Tensor
from .aux import zeropower_via_newtonschulz5, adam_update


def adamuon_update(grad: Tensor, momentum: Tensor, variance: Tensor, beta: float = 0.95, ns_steps: int = 5, nesterov=True, eps: float = 1e-8):
    momentum.mul_(beta).add_(grad)
    update = grad.add(momentum, alpha=beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(torch.sign(update), steps=ns_steps)
    variance.lerp_(update.square(), 1 - beta)
    update.div_(variance.view_as(update).sqrt() + eps)
    scale = 0.2 * (
        min(update.shape) * max(update.shape)
    ) ** 0.5 / (update.norm() + eps)
    update.mul_(scale)
    return update


class AdaMuon(torch.optim.Optimizer):
    def __init__(self, param_groups: list[dict]):
        for group in param_groups:
            assert "use_muon" in group, "param_groups must explicitly indicate 'use_muon'"
            if group["use_muon"]:
                group['lr'] = group.get('lr', 2 * 1e-2)
                group['momentum'] = group.get('momentum', 0.95)
                group['weight_decay'] = group.get('weight_decay', 0)
                group['eps'] = group.get('eps', 1e-8)
            else:
                group['lr'] = group.get('lr', 3e-4)
                group['betas'] = group.get('betas', (0.9, 0.95))
                group['eps'] = group.get('eps', 1e-10)
                group['weight_decay'] = group.get('weight_decay', 0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group['params']:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if (len(state) == 0):
                        state['momentum'] = torch.zeros_like(p)
                        if p.ndim == 4:
                            var_shape = (
                                p.shape[0], p.shape[1] * p.shape[2] * p.shape[3])
                        else:
                            var_shape = p.shape
                        state['variance'] = torch.zeros(
                            var_shape, device=p.device, dtype=p.dtype)
                    update = adamuon_update(
                        p.grad, state['momentum'], state['variance'], beta=group['momentum'])
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(update.reshape(p.shape), alpha=-group['lr'])

            else:
                for p in group['params']:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if (len(state) == 0):
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        state['step'] = 0
                    state['step'] += 1
                    update = adam_update(
                        p.grad, state['exp_avg'], state['exp_avg_sq'], state['step'], group['betas'], group['eps'])
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(update, alpha=-group['lr'])
        return loss
