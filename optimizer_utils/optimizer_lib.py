import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch
from torch import Tensor
from typing import List, Optional

import numpy as np

import copy
import os
import shutil

class omega_update(optim.Adam):

	def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *, foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False):
		super(omega_update, self).__init__(params)

	def __setstate__(self, state):
		super(omega_update, self).__setstate__(state)

	def step(self, reg_params, data_position, batch_size, args, closure = None):
		self._cuda_graph_capture_health_check()
		device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
		loss = None

		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			
			beta1, beta2 = group['betas']
			amsgrad = group['amsgrad']
			lr=args.cri_lr
			weight_decay = group['weight_decay']
			eps = group['eps']
			maximize = group['maximize']
			foreach = group['foreach']
			capturable : bool = group['capturable']
			

			for p in group['params']:
				if p.grad is None:
					continue

				if p in reg_params:

					grad_data = p.grad.data

					#The absolute value of the grad_data that is to be added to omega 
					grad_data_copy = p.grad.data.clone()
					grad_data_copy = grad_data_copy.abs()
					
					param_dict = reg_params[p]

					omega = param_dict['omega']
					omega = omega.to(device)

					current_size = data_position + batch_size
					step_size = 1/float(current_size)

					#Incremental update for the omega
					omega = omega + step_size*(grad_data_copy - batch_size*(omega))

					param_dict['omega'] = omega

					reg_params[p] = param_dict

		return loss

class Local_Adam(torch.optim.Adam):
    def __init__(self, params, reg_lambda ,lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *, foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False):
		
        super(Local_Adam, self).__init__(params, lr)
        self.reg_lambda = reg_lambda

    def __setstate__(self, state):
        super(Local_Adam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, reg_params, args, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()
        device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                d_p = p.grad.data
                if p.grad is None:
                    continue
                if p in reg_params:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    param_dict = reg_params[p]

                    omega = param_dict['omega']
                    init_val = param_dict['init_val']

                    curr_param_value = p.data
                    curr_param_value = curr_param_value.to(device)
 
                    init_val = init_val.to(device)
                    omega = omega.to(device)

					#get the difference
                    param_diff = curr_param_value - init_val

					#get the gradient for the penalty term for change in the weights of the parameters
                    local_grad = torch.mul(param_diff, 2*self.reg_lambda*omega)

                    d_p = d_p + local_grad
                    grads.append(d_p)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                            if self.defaults['capturable'] else torch.tensor(0.)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    state_steps.append(state['step'])

            adam(params_with_grad,
                 grads,
                 exp_avgs,
                 exp_avg_sqs,
                 max_exp_avg_sqs,
                 state_steps,
                 amsgrad=group['amsgrad'],
                 beta1=beta1,
                 beta2=beta2,
                 lr=group['lr'],
                 weight_decay=group['weight_decay'],
                 eps=group['eps'],
                 maximize=group['maximize'],
                 foreach=group['foreach'],
                 capturable=group['capturable'])

        return loss


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[Tensor],
         # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
         # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
         foreach: bool = None,
         capturable: bool = False,
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         maximize: bool):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    if not all([isinstance(t, torch.Tensor) for t in state_steps]):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adam
    else:
        func = _single_tensor_adam

    func(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         amsgrad=amsgrad,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         weight_decay=weight_decay,
         eps=eps,
         maximize=maximize,
         capturable=capturable)


def _single_tensor_adam(params: List[Tensor],
                        grads: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        max_exp_avg_sqs: List[Tensor],
                        state_steps: List[Tensor],
                        *,
                        amsgrad: bool,
                        beta1: float,
                        beta2: float,
                        lr: float,
                        weight_decay: float,
                        eps: float,
                        maximize: bool,
                        capturable: bool):

    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."
        else:
            assert not step_t.is_cuda, "If capturable=False, state_steps should not be CUDA tensors."

        # update step
        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        if capturable:
            step = step_t

            # 1 - beta1 ** step can't be captured in a CUDA graph, even if step is a CUDA tensor
            # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
            bias_correction1 = 1 - torch.pow(beta1, step)
            bias_correction2 = 1 - torch.pow(beta2, step)

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)
        else:
            step = step_t.item()

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = math.sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            param.addcdiv_(exp_avg, denom, value=-step_size)


def _multi_tensor_adam(params: List[Tensor],
                       grads: List[Tensor],
                       exp_avgs: List[Tensor],
                       exp_avg_sqs: List[Tensor],
                       max_exp_avg_sqs: List[Tensor],
                       state_steps: List[Tensor],
                       *,
                       amsgrad: bool,
                       beta1: float,
                       beta2: float,
                       lr: float,
                       weight_decay: float,
                       eps: float,
                       maximize: bool,
                       capturable: bool):
    if len(params) == 0:
        return

    if capturable:
        assert all(p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)), \
            "If capturable=True, params and state_steps must be CUDA tensors."
    else:
        assert all(not step.is_cuda for step in state_steps), \
            "If capturable=False, state_steps should not be CUDA tensors."

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]

    # update steps
    torch._foreach_add_(state_steps, 1)

    if weight_decay != 0:
        torch._foreach_add_(grads, params, alpha=weight_decay)

    # Decay the first and second moment running average coefficient
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)

    if capturable:
        # TODO: use foreach_pow if/when foreach_pow is added
        bias_correction1 = [torch.pow(beta1, step) for step in state_steps]
        bias_correction2 = [torch.pow(beta2, step) for step in state_steps]
        # foreach_sub doesn't allow a scalar as the first arg
        torch._foreach_sub_(bias_correction1, 1)
        torch._foreach_sub_(bias_correction2, 1)
        torch._foreach_neg_(bias_correction1)
        torch._foreach_neg_(bias_correction2)

        # foreach_div doesn't allow a scalar as the first arg
        step_size = torch._foreach_div(bias_correction1, lr)
        torch._foreach_reciprocal_(step_size)
        torch._foreach_neg_(step_size)

        bias_correction2_sqrt = torch._foreach_sqrt(bias_correction2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            max_exp_avg_sqs = torch._foreach_maximum(max_exp_avg_sqs, exp_avg_sqs)  # type: ignore[assignment]

            # Use the max. for normalizing running avg. of gradient
            max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sqs)
            # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
            # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
            torch._foreach_div_(max_exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
            eps_over_step_size = torch._foreach_div(step_size, eps)
            torch._foreach_reciprocal_(eps_over_step_size)
            denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps_over_step_size)
        else:
            exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
            eps_over_step_size = torch._foreach_div(step_size, eps)
            torch._foreach_reciprocal_(eps_over_step_size)
            denom = torch._foreach_add(exp_avg_sq_sqrt, eps_over_step_size)

        torch._foreach_addcdiv_(params, exp_avgs, denom)
    else:
        bias_correction1 = [1 - beta1 ** step.item() for step in state_steps]
        bias_correction2 = [1 - beta2 ** step.item() for step in state_steps]

        step_size = [(lr / bc) * -1 for bc in bias_correction1]

        bias_correction2_sqrt = [math.sqrt(bc) for bc in bias_correction2]

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            max_exp_avg_sqs = torch._foreach_maximum(max_exp_avg_sqs, exp_avg_sqs)  # type: ignore[assignment]

            # Use the max. for normalizing running avg. of gradient
            max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sqs)
            torch._foreach_div_(max_exp_avg_sq_sqrt, bias_correction2_sqrt)
            denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps)
        else:
            exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            denom = torch._foreach_add(exp_avg_sq_sqrt, eps)

        torch._foreach_addcdiv_(params, exp_avgs, denom, step_size)