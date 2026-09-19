from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import gpytorch
import numpy as np
import torch
from tqdm import tqdm

try:
    from models import SingleTaskDeepKernel
except ImportError as exc:  # pragma: no cover - depends on the user's project layout
    raise ImportError(
        "Could not import SingleTaskDeepKernel from models.py. Run the pipeline "
        "from the project root containing models.py, or add that directory to PYTHONPATH."
    ) from exc


@dataclass(frozen=True)
class DKGPConfig:
    iterations: int = 100
    learning_rate: float = 0.02
    weight_decay: float = 0.10
    dropout: float = 0.20
    activation: str = "relu"
    kernel: str = "RBF"
    mean: str = "Constant"
    uncertainty: str = "epistemic"
    std_floor: float = 1e-6
    prediction_batch_size: int = 2048

    def validate(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if self.uncertainty not in {"epistemic", "predictive"}:
            raise ValueError("uncertainty must be 'epistemic' or 'predictive'.")
        if self.std_floor <= 0:
            raise ValueError("std_floor must be positive.")
        if self.prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_dkgp(
    *,
    train_x_cpu: torch.Tensor,
    train_y_cpu: torch.Tensor,
    device: torch.device,
    config: DKGPConfig,
    progress_description: str | None = None,
) -> Tuple[SingleTaskDeepKernel, gpytorch.likelihoods.GaussianLikelihood, List[float]]:
    """Fit one exact deep-kernel GP for one biomarker."""
    config.validate()

    train_x = train_x_cpu.to(device=device, dtype=torch.float32)
    train_y = train_y_cpu.to(device=device, dtype=torch.float32).reshape(-1)

    if train_x.ndim != 2:
        raise ValueError(f"Expected train_x to be 2D, obtained {tuple(train_x.shape)}.")
    if train_y.ndim != 1:
        raise ValueError(f"Expected train_y to be 1D, obtained {tuple(train_y.shape)}.")
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("train_x and train_y have different numbers of observations.")
    if train_x.shape[0] < 2:
        raise ValueError("At least two training observations are required.")

    latent_dim = max(1, int(train_x.shape[1] / 2))
    depth = [(train_x.shape[1], latent_dim)]

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SingleTaskDeepKernel(
        input_dim=train_x.shape[1],
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        depth=depth,
        dropout=config.dropout,
        activation=config.activation,
        kernel_choice=config.kernel,
        mean=config.mean,
        pretrained=False,
        feature_extractor=None,
        latent_dim=latent_dim,
        gphyper=None,
    ).to(device)

    model.train()
    likelihood.train()
    model.feature_extractor.train()

    optimizer = torch.optim.Adam(
        [
            {"params": model.feature_extractor.parameters(), "lr": config.learning_rate},
            {"params": model.covar_module.parameters(), "lr": config.learning_rate},
            {"params": model.mean_module.parameters(), "lr": config.learning_rate},
            {"params": likelihood.parameters(), "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses: List[float] = []
    iterator = tqdm(
        range(config.iterations),
        leave=False,
        desc=progress_description,
    )
    for _ in iterator:
        optimizer.zero_grad(set_to_none=True)
        output = model(train_x)
        loss = -mll(output, train_y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite DKGP loss: {float(loss.item())}")
        loss.backward()
        optimizer.step()
        value = float(loss.item())
        losses.append(value)
        iterator.set_postfix(loss=f"{value:.4f}")

    model.eval()
    likelihood.eval()
    model.feature_extractor.eval()
    return model, likelihood, losses


@torch.no_grad()
def predict_dkgp(
    *,
    model: SingleTaskDeepKernel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    x_cpu: torch.Tensor,
    device: torch.device,
    config: DKGPConfig,
) -> Dict[str, np.ndarray]:
    """Return mean, selected variance, and standard deviation on CPU."""
    config.validate()
    if x_cpu.ndim != 2:
        raise ValueError(f"Expected x_cpu to be 2D, obtained {tuple(x_cpu.shape)}.")

    model.eval()
    likelihood.eval()

    means: List[np.ndarray] = []
    variances: List[np.ndarray] = []

    for start in range(0, x_cpu.shape[0], config.prediction_batch_size):
        stop = min(start + config.prediction_batch_size, x_cpu.shape[0])
        x_batch = x_cpu[start:stop].to(device=device, dtype=torch.float32)

        with gpytorch.settings.fast_pred_var():
            latent_distribution = model(x_batch)
            if config.uncertainty == "epistemic":
                selected_distribution = latent_distribution
            else:
                selected_distribution = likelihood(latent_distribution)

        means.append(selected_distribution.mean.detach().cpu().numpy())
        variances.append(selected_distribution.variance.detach().cpu().numpy())

    mean = np.concatenate(means).reshape(-1)
    variance = np.maximum(np.concatenate(variances).reshape(-1), 0.0)
    std = np.maximum(np.sqrt(variance), config.std_floor)

    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise FloatingPointError("DKGP prediction produced non-finite mean or standard deviation.")

    return {"mean": mean, "variance": variance, "std": std}


def save_checkpoint(
    *,
    path: Path,
    model: SingleTaskDeepKernel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_x_cpu: torch.Tensor,
    train_y_cpu: torch.Tensor,
    biomarker: str,
    target_index: int,
    config: DKGPConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "biomarker": biomarker,
            "target_index": target_index,
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "train_x": train_x_cpu.detach().cpu(),
            "train_y": train_y_cpu.detach().cpu(),
            "config": asdict(config),
        },
        path,
    )
