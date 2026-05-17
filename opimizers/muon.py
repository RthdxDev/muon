import torch
from torch import Tensor
from .aux import zeropower_via_newtonschulz5, adam_update


def muon_update(grad: Tensor, momentum: Tensor, beta: float = 0.95, ns_steps: int = 5, nesterov=True):
    momentum.mul_(beta).add_(grad)
    update = grad.add(momentum, alpha=beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    # scale = max(1, update.size(-2) / update.size(-1)) ** 0.5
    scale = 0.2 * max(update.shape) ** 0.5  # kimi update
    update.mul_(scale)
    return update


class Muon(torch.optim.Optimizer):
    def __init__(self, param_groups: list[dict]):
        for group in param_groups:
            assert "use_muon" in group, "param_groups must explicitly indicate 'use_muon'"
            if group["use_muon"]:
                group['lr'] = group.get('lr', 2 * 1e-2)
                group['momentum'] = group.get('momentum', 0.95)
                group['weight_decay'] = group.get('weight_decay', 0)
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
                    update = muon_update(
                        p.grad, state['momentum'], beta=group['momentum'])
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
