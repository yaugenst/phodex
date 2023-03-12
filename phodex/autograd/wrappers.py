from functools import partial
from typing import Any, Callable

import autograd.numpy as np
import torch
from autograd import make_vjp
from autograd.extend import defvjp, primitive, vspace


def autograd_to_pytorch(fun: Callable) -> Callable:
    """Creates a PyTorch function from an autograd one."""

    class _Wrapper(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *args: Any) -> torch.Tensor:
            device = torch.device("cpu")
            numpy_args = []
            grads = []
            for arg in args:
                numpy_arg = arg
                requires_grad = None
                if torch.is_tensor(arg):
                    numpy_arg = arg.detach().cpu().numpy()
                    if arg.requires_grad:
                        device = arg.device
                        requires_grad = True
                numpy_args.append(numpy_arg)
                grads.append(requires_grad)

            grad_argnums = [n for n, g in enumerate(grads) if g]
            _vjp = make_vjp(fun, argnum=grad_argnums)
            vjp, ans = _vjp(*numpy_args)
            ctx.vjp = vjp
            ctx.ans = ans
            ctx.grads = grads
            ctx.grad_argnums = grad_argnums
            ctx.device = device
            return torch.as_tensor(ans, device=device)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple:
            _grads = ctx.vjp(vspace(ctx.ans).ones())
            grads = ctx.grads
            for idx, g in zip(ctx.grad_argnums, _grads):
                grads[idx] = torch.as_tensor(g, device=ctx.device) * grad_output
            return tuple(grads)

    return _Wrapper.apply


def pytorch_to_autograd(fun: Callable) -> Callable:
    def _make_vjps(args: Any, out: Any) -> None:
        vjps = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                # see https://stackoverflow.com/a/34021333
                vjps.append(partial(backward, arg=arg, out=out))
            else:
                vjps.append(None)
        defvjp(forward, *vjps)

    def _args_to_pytorch(args: Any) -> list:
        pytorch_args = []
        for arg in args:
            if isinstance(arg, (np.ndarray, float)):
                tensor = torch.as_tensor(arg)
                if tensor.dtype in (torch.float32, torch.float64):
                    tensor.requires_grad_()
                pytorch_args.append(tensor)
            else:
                pytorch_args.append(arg)
        return pytorch_args

    @primitive
    def forward(*args: Any) -> np.ndarray:
        # There is no way to tell w.r.t. which args we are going to need gradients,
        # so we will create a graph for anything that could potentially need one.
        # This can be a problem if the PyTorch function takes non-tensor float
        # arguments - the simple solution is to avoid this where possible or wrap
        # the calls in lambdas.
        # A fix could be to pass the argnums that require gradients explicitly to
        # this decorator, but then the functional wrapping style would get messy.
        pytorch_args = _args_to_pytorch(args)
        out = fun(*pytorch_args)
        _make_vjps(pytorch_args, out)
        return out.detach().numpy()

    def backward(ans: np.ndarray, *x: Any, arg: Any, out: Any) -> Callable:
        def vjp(g: np.ndarray) -> np.ndarray:
            if arg.grad is None:
                out.backward(torch.ones_like(out))
            return arg.grad.numpy() * g

        return vjp

    return forward
