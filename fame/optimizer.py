"""The FAME optimization loop.

The function is ``fame``; the module is ``optimizer`` rather than ``fame`` so
that it cannot shadow the exported function.  A submodule whose name matches an
attribute of its package overwrites that attribute whenever the submodule is
imported, so ``fame.fame`` would resolve to the module or the function
depending on import order.

Iteratively perturbs an input image so that the network output at a chosen
layer moves towards a chosen target, and returns the perturbed image; the
caller turns the accumulated perturbation into an attribution map.  This module
knows nothing about classification, verification or feature maps -- the
task-specific part lives entirely in the loss function that is passed in.

The procedure itself is Layerwise Origin Target Synthesis, published as an
adversarial attack by Rozsa et al., "LOTS about attacking deep features", IJCB
2017.  FAME reuses it unchanged and reinterprets its output as an explanation,
which is why the update rule below is Eq. (3) of the FAME paper verbatim.
"""

from typing import Callable, Optional

import torch

# Per-sample loss: takes the current (batched) image and returns one loss value
# per batch element.  Losses are summed before backward, which is safe because
# batch elements do not interact -- each sample only receives its own gradient.
LossFn = Callable[[torch.Tensor], torch.Tensor]


def fame(
    image: torch.Tensor,
    loss_fn: LossFn,
    step_size: float = 1.0 / 255.0,
    iterations: int = 500,
    epsilon: Optional[float] = None,
    clamp: Optional[tuple] = None,
) -> torch.Tensor:
    """Run the FAME optimization and return the perturbed image.

    Args:
        image: input batch of shape (B, C, H, W).  The step is taken in
            whatever space this tensor lives in: pass a mean/std normalized
            tensor to perturb in normalized units (the setting used for the
            reported results), or a raw [0, 1] image together with a model that
            normalizes internally to perturb in pixel units.  The two are not
            equivalent, because dividing by a per-channel std rescales the step
            differently for each colour channel.
        loss_fn: callable mapping the current image to a per-sample loss of
            shape (B,).  The optimizer descends this loss.
        step_size: eta in Eq. (3) of the paper, 1/255 throughout.
        iterations: number of gradient steps.
        epsilon: optional early-stopping threshold.  Iteration stops once the
            mean loss drops below it.  ``None`` disables early stopping.
        clamp: optional (min, max) range the perturbed image is kept in.
            Defaults to ``None``, matching the original unconstrained updates;
            only set it when perturbing raw pixels.

    Returns:
        The perturbed image, detached, on the same device as the input.
    """
    x_adv = image.clone().detach().requires_grad_(True)

    for _ in range(iterations):
        loss = loss_fn(x_adv)
        if epsilon is not None and loss.mean().item() < epsilon:
            break

        (gradient,) = torch.autograd.grad(loss.sum(), x_adv)

        with torch.no_grad():
            # Normalize the gradient by its largest absolute value so that the
            # strongest pixel moves by exactly ``step_size``.  This is computed
            # per sample, otherwise one sample in the batch would rescale the
            # steps of all the others.
            denominator = gradient.abs().amax(dim=(1, 2, 3), keepdim=True)
            step = gradient * (step_size / denominator.clamp_min(1e-12))
            x_adv = x_adv - step
            if clamp is not None:
                x_adv = x_adv.clamp(*clamp)

        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv.detach()
