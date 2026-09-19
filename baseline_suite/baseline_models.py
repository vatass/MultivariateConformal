from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import gpytorch
except ImportError:  # pragma: no cover - the user's GP environment is external.
    gpytorch = None

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover
    sm = None


ArrayDict = Dict[str, np.ndarray]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def robust_scale(values: np.ndarray, floor: float = 1e-6) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return floor
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < floor:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale < floor:
        scale = floor
    return float(scale)


@dataclass
class NumpyStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, floor: float = 1e-6) -> "NumpyStandardizer":
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std = np.where(np.isfinite(std) & (std >= floor), std, 1.0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x = np.where(np.isfinite(x), x, self.mean)
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x) * self.std + self.mean


@dataclass
class ScalarStandardizer:
    mean: float
    std: float

    @classmethod
    def fit(cls, y: np.ndarray, floor: float = 1e-6) -> "ScalarStandardizer":
        mean = float(np.nanmean(y))
        std = float(np.nanstd(y))
        if not np.isfinite(std) or std < floor:
            std = 1.0
        return cls(mean=mean, std=std)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((np.asarray(y) - self.mean) / self.std).astype(np.float32)

    def inverse_mean(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y) * self.std + self.mean

    def inverse_std(self, std: np.ndarray) -> np.ndarray:
        return np.asarray(std) * self.std


class BaselineRegressor:
    name: str = "base"
    score_type: str = "scaled_residual"

    def fit(self, x: np.ndarray, y: np.ndarray, subject_ids: np.ndarray) -> "BaselineRegressor":
        raise NotImplementedError

    def predict(self, x: np.ndarray, alpha: float) -> ArrayDict:
        raise NotImplementedError


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.0,
        output_dim: int = 1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation == "relu":
            activation_factory = nn.ReLU
        elif activation == "sigmoid":
            activation_factory = nn.Sigmoid
        elif activation == "gelu":
            activation_factory = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: List[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [nn.Linear(previous, hidden), activation_factory(), nn.Dropout(dropout)]
            )
            previous = hidden
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TorchMLPRegressor(BaselineRegressor):
    name = "mlp"

    def __init__(
        self,
        *,
        seed: int,
        device: torch.device,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.0,
        epochs: int = 200,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        activation: str = "relu",
        std_floor: float = 1e-6,
    ) -> None:
        self.seed = seed
        self.device = device
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.activation = activation
        self.std_floor = std_floor

    def _new_network(self, input_dim: int, output_dim: int = 1) -> FeedForwardNetwork:
        return FeedForwardNetwork(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            output_dim=output_dim,
            activation=self.activation,
        ).to(self.device)

    def _fit_network(
        self,
        network: nn.Module,
        x_std: np.ndarray,
        y_std: np.ndarray,
        loss_function,
        seed: int,
    ) -> None:
        set_seed(seed)
        dataset = TensorDataset(
            torch.from_numpy(x_std).float(),
            torch.from_numpy(y_std).float(),
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        network.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                output = network(batch_x)
                loss = loss_function(output, batch_y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite {self.name} loss: {loss.item()}")
                loss.backward()
                optimizer.step()

    def fit(self, x: np.ndarray, y: np.ndarray, subject_ids: np.ndarray) -> "TorchMLPRegressor":
        del subject_ids
        self.x_scaler = NumpyStandardizer.fit(x)
        self.y_scaler = ScalarStandardizer.fit(y)
        x_std = self.x_scaler.transform(x)
        y_std = self.y_scaler.transform(y).reshape(-1, 1)

        self.network = self._new_network(x_std.shape[1])
        self._fit_network(
            network=self.network,
            x_std=x_std,
            y_std=y_std,
            loss_function=lambda prediction, target: torch.mean((prediction - target) ** 2),
            seed=self.seed,
        )

        train_mean = self._predict_mean(x)
        self.residual_scale = robust_scale(y - train_mean, floor=self.std_floor)
        return self

    @torch.no_grad()
    def _predict_mean(self, x: np.ndarray) -> np.ndarray:
        x_std = self.x_scaler.transform(x)
        self.network.eval()
        means: List[np.ndarray] = []
        for start in range(0, len(x_std), self.batch_size):
            batch = torch.from_numpy(x_std[start : start + self.batch_size]).float().to(self.device)
            means.append(self.network(batch).squeeze(-1).cpu().numpy())
        return self.y_scaler.inverse_mean(np.concatenate(means)).reshape(-1)

    def predict(self, x: np.ndarray, alpha: float) -> ArrayDict:
        mean = self._predict_mean(x)
        std = np.full_like(mean, self.residual_scale, dtype=float)
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        return {
            "mean": mean,
            "std": std,
            "native_lower": mean - z * std,
            "native_upper": mean + z * std,
            "conformal_scale": std,
            "score_type": "scaled_residual",
        }


class MCDropoutRegressor(TorchMLPRegressor):
    name = "drmc"

    def __init__(self, *, mc_samples: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.dropout <= 0:
            raise ValueError("DRMC requires dropout > 0.")
        self.mc_samples = mc_samples

    @torch.no_grad()
    def _mc_predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_std = self.x_scaler.transform(x)
        draws: List[np.ndarray] = []
        self.network.train()  # activate dropout at inference
        for _ in range(self.mc_samples):
            draw_parts: List[np.ndarray] = []
            for start in range(0, len(x_std), self.batch_size):
                batch = torch.from_numpy(x_std[start : start + self.batch_size]).float().to(self.device)
                draw_parts.append(self.network(batch).squeeze(-1).cpu().numpy())
            draws.append(np.concatenate(draw_parts))
        stacked = np.stack(draws, axis=0)
        mean_std_units = np.mean(stacked, axis=0)
        std_std_units = np.std(stacked, axis=0, ddof=1)
        mean = self.y_scaler.inverse_mean(mean_std_units)
        std = np.maximum(self.y_scaler.inverse_std(std_std_units), self.std_floor)
        return mean.reshape(-1), std.reshape(-1)

    def predict(self, x: np.ndarray, alpha: float) -> ArrayDict:
        mean, std = self._mc_predict(x)
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        return {
            "mean": mean,
            "std": std,
            "native_lower": mean - z * std,
            "native_upper": mean + z * std,
            "conformal_scale": std,
            "score_type": "scaled_residual",
        }


class BootstrapMLPRegressor(BaselineRegressor):
    name = "bootstrap"

    def __init__(
        self,
        *,
        seed: int,
        device: torch.device,
        ensemble_size: int = 10,
        hidden_dims: Sequence[int] = (128, 64),
        epochs: int = 200,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        std_floor: float = 1e-6,
    ) -> None:
        self.seed = seed
        self.device = device
        self.ensemble_size = ensemble_size
        self.hidden_dims = tuple(hidden_dims)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.std_floor = std_floor

    @staticmethod
    def _cluster_bootstrap_indices(subject_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        unique_ids = np.unique(subject_ids.astype(str))
        sampled_ids = rng.choice(unique_ids, size=len(unique_ids), replace=True)
        indices: List[int] = []
        string_ids = subject_ids.astype(str)
        for sampled_id in sampled_ids:
            indices.extend(np.flatnonzero(string_ids == sampled_id).tolist())
        return np.asarray(indices, dtype=int)

    def fit(self, x: np.ndarray, y: np.ndarray, subject_ids: np.ndarray) -> "BootstrapMLPRegressor":
        self.x_scaler = NumpyStandardizer.fit(x)
        self.y_scaler = ScalarStandardizer.fit(y)
        x_std_all = self.x_scaler.transform(x)
        y_std_all = self.y_scaler.transform(y).reshape(-1, 1)
        self.networks: List[FeedForwardNetwork] = []

        for member in range(self.ensemble_size):
            member_seed = self.seed + member
            rng = np.random.default_rng(member_seed)
            indices = self._cluster_bootstrap_indices(subject_ids, rng)
            trainer = TorchMLPRegressor(
                seed=member_seed,
                device=self.device,
                hidden_dims=self.hidden_dims,
                dropout=0.0,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
                std_floor=self.std_floor,
            )
            network = trainer._new_network(x_std_all.shape[1])
            trainer._fit_network(
                network=network,
                x_std=x_std_all[indices],
                y_std=y_std_all[indices],
                loss_function=lambda prediction, target: torch.mean((prediction - target) ** 2),
                seed=member_seed,
            )
            self.networks.append(network)
        return self

    @torch.no_grad()
    def predict(self, x: np.ndarray, alpha: float) -> ArrayDict:
        x_std = self.x_scaler.transform(x)
        member_predictions: List[np.ndarray] = []
        for network in self.networks:
            network.eval()
            parts: List[np.ndarray] = []
            for start in range(0, len(x_std), self.batch_size):
                batch = torch.from_numpy(x_std[start : start + self.batch_size]).float().to(self.device)
                parts.append(network(batch).squeeze(-1).cpu().numpy())
            member_predictions.append(np.concatenate(parts))

        draws = np.stack(member_predictions, axis=0)
        mean = self.y_scaler.inverse_mean(np.mean(draws, axis=0)).reshape(-1)
        std = np.maximum(
            self.y_scaler.inverse_std(np.std(draws, axis=0, ddof=1)),
            self.std_floor,
        ).reshape(-1)
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        return {
            "mean": mean,
            "std": std,
            "native_lower": mean - z * std,
            "native_upper": mean + z * std,
            "conformal_scale": std,
            "score_type": "scaled_residual",
        }


class DeepQuantileRegressor(TorchMLPRegressor):
    name = "dqr"
    score_type = "cqr"

    def __init__(
        self,
        *,
        quantiles: Sequence[float],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if len(quantiles) != 3:
            raise ValueError("DQR currently expects exactly three quantiles.")
        if list(quantiles) != sorted(quantiles):
            raise ValueError("DQR quantiles must be increasing.")
        if not all(0.0 < q < 1.0 for q in quantiles):
            raise ValueError("Every DQR quantile must be in (0,1).")
        self.quantiles = tuple(float(q) for q in quantiles)

    def fit(self, x: np.ndarray, y: np.ndarray, subject_ids: np.ndarray) -> "DeepQuantileRegressor":
        del subject_ids
        self.x_scaler = NumpyStandardizer.fit(x)
        self.y_scaler = ScalarStandardizer.fit(y)
        x_std = self.x_scaler.transform(x)
        y_std = self.y_scaler.transform(y).reshape(-1, 1)
        self.network = self._new_network(x_std.shape[1], output_dim=3)
        quantiles_tensor = torch.tensor(self.quantiles, dtype=torch.float32, device=self.device).view(1, -1)

        def pinball(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            error = target - prediction
            return torch.maximum(
                quantiles_tensor * error,
                (quantiles_tensor - 1.0) * error,
            ).mean()

        self._fit_network(
            network=self.network,
            x_std=x_std,
            y_std=y_std,
            loss_function=pinball,
            seed=self.seed,
        )

        train_quantiles = self._predict_quantiles(x)
        median_prediction = train_quantiles[:, 1]
        self.cqr_scale = robust_scale(y - median_prediction, floor=self.std_floor)
        return self

    @torch.no_grad()
    def _predict_quantiles(self, x: np.ndarray) -> np.ndarray:
        x_std = self.x_scaler.transform(x)
        self.network.eval()
        parts: List[np.ndarray] = []
        for start in range(0, len(x_std), self.batch_size):
            batch = torch.from_numpy(x_std[start : start + self.batch_size]).float().to(self.device)
            output = self.network(batch)
            output, _ = torch.sort(output, dim=1)
            parts.append(output.cpu().numpy())
        quantiles_std = np.concatenate(parts, axis=0)
        return self.y_scaler.inverse_mean(quantiles_std)

    def predict(self, x: np.ndarray, alpha: float) -> ArrayDict:
        del alpha  # native DQR bounds are determined by the configured quantiles.
        quantiles = self._predict_quantiles(x)
        lower, median, upper = quantiles[:, 0], quantiles[:, 1], quantiles[:, 2]
        scale = np.full_like(median, self.cqr_scale, dtype=float)
        return {
            "mean": median,
            "std": scale,
            "native_lower": lower,
            "native_upper": upper,
            "conformal_scale": scale,
            "score_type": "cqr",
            "q_lower": lower,
            "q_median": median,
            "q_upper": upper,
        }


if gpytorch is not None:
    class _ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )


class ExactGPRegressor(BaselineRegressor):
    name = "exact_gp"

    def __init__(
        self,
        *,
        seed: int,
        device: torch.device,
        iterations: int = 100,
        learning_rate: float = 0.10,
        uncertainty: str = "epistemic",
        std_floor: float = 1e-6,
        prediction_batch_size: int = 2048,
        max_cholesky_size: int = 1000,
    ) -> None:
        if gpytorch is None:
            raise ImportError("ExactGP requires gpytorch.")
        self.seed = seed
        self.device = device
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.uncertainty = uncertainty
        self.std_floor = std_floor
        self.prediction_batch_size = prediction_batch_size
        self.max_cholesky_size = max_cholesky_size

    def fit(self, x: np.ndarray, y: np.ndarray, subject_ids: np.ndarray) -> "ExactGPRegressor":
        del subject_ids
        set_seed(self.seed)
        self.x_scaler = NumpyStandardizer.fit(x)
        self.y_scaler = ScalarStandardizer.fit(y)
        train_x = torch.from_numpy(self.x_scaler.transform(x)).float().to(self.device)
        train_y = torch.from_numpy(self.y_scaler.transform(y)).float().to(self.device)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = _ExactGPModel(train_x, train_y, self.likelihood).to(self.device)
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(self.iterations):
            optimizer.zero_grad(set_to_none=True)
            with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                output = self.model(train_x)
                loss = -mll(output, train_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite exact GP loss: {loss.item()}")
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.likelihood.eval()
        return self

    @torch.no_grad()
    def predict(self, x: np.ndarray, alpha: float) -> ArrayDict:
        x_std = self.x_scaler.transform(x)
        means: List[np.ndarray] = []
        variances: List[np.ndarray] = []
        self.model.eval()
        self.likelihood.eval()

        for start in range(0, len(x_std), self.prediction_batch_size):
            batch = torch.from_numpy(x_std[start : start + self.prediction_batch_size]).float().to(self.device)
            with gpytorch.settings.max_cholesky_size(self.max_cholesky_size), gpytorch.settings.fast_pred_var():
                latent = self.model(batch)
                distribution = latent if self.uncertainty == "epistemic" else self.likelihood(latent)
            means.append(distribution.mean.cpu().numpy())
            variances.append(distribution.variance.cpu().numpy())

        mean = self.y_scaler.inverse_mean(np.concatenate(means)).reshape(-1)
        std = np.maximum(
            self.y_scaler.inverse_std(np.sqrt(np.maximum(np.concatenate(variances), 0.0))),
            self.std_floor,
        ).reshape(-1)
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        return {
            "mean": mean,
            "std": std,
            "native_lower": mean - z * std,
            "native_upper": mean + z * std,
            "conformal_scale": std,
            "score_type": "scaled_residual",
        }


if gpytorch is not None:
    class _ResidualVariationalGP(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points: torch.Tensor):
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            )
            super().__init__(variational_strategy)
            latent_dim = inducing_points.shape[1]
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim)
            )

        def forward(self, z):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(z), self.covar_module(z)
            )


    class _DeepMeanVariationalModel(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int, inducing_x: torch.Tensor):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, latent_dim),
                nn.Sigmoid(),
            )
            self.mean_network = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            with torch.no_grad():
                inducing_z = self.feature_extractor(inducing_x)
            self.residual_gp = _ResidualVariationalGP(inducing_z)

        def forward(self, x):
            z = self.feature_extractor(x)
            residual = self.residual_gp(z)
            deep_mean = self.mean_network(z).squeeze(-1)
            return gpytorch.distributions.MultivariateNormal(
                residual.mean + deep_mean,
                residual.lazy_covariance_matrix,
            )


class DMEGPRegressor(BaselineRegressor):
    name = "dmegp"

    def __init__(
        self,
        *,
        seed: int,
        device: torch.device,
        latent_dim: int = 64,
        inducing_points: int = 256,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 256,
        uncertainty: str = "epistemic",
        std_floor: float = 1e-6,
    ) -> None:
        if gpytorch is None:
            raise ImportError("DMEGP requires gpytorch.")
        self.seed = seed
        self.device = device
        self.latent_dim = latent_dim
        self.inducing_points = inducing_points
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.uncertainty = uncertainty
        self.std_floor = std_floor

    def fit(self, x: np.ndarray, y: np.ndarray, subject_ids: np.ndarray) -> "DMEGPRegressor":
        del subject_ids
        set_seed(self.seed)
        self.x_scaler = NumpyStandardizer.fit(x)
        self.y_scaler = ScalarStandardizer.fit(y)
        x_std = self.x_scaler.transform(x)
        y_std = self.y_scaler.transform(y)

        x_tensor = torch.from_numpy(x_std).float()
        y_tensor = torch.from_numpy(y_std).float()
        dataset = TensorDataset(x_tensor, y_tensor)
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )

        rng = np.random.default_rng(self.seed)
        n_inducing = min(self.inducing_points, len(x_std))
        inducing_indices = rng.choice(len(x_std), size=n_inducing, replace=False)
        inducing_x = x_tensor[inducing_indices].to(self.device)

        self.model = _DeepMeanVariationalModel(
            input_dim=x_std.shape[1],
            latent_dim=self.latent_dim,
            inducing_x=inducing_x,
        ).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        n_total = len(dataset)

        self.model.train()
        self.likelihood.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                output = self.model(batch_x)
                expected_log_prob = self.likelihood.expected_log_prob(batch_y, output).mean()
                kl = self.model.residual_gp.variational_strategy.kl_divergence().sum() / n_total
                loss = -expected_log_prob + kl
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite DMEGP loss: {loss.item()}")
                loss.backward()
                optimizer.step()

        self.model.eval()
        self.likelihood.eval()
        return self

    @torch.no_grad()
    def predict(self, x: np.ndarray, alpha: float) -> ArrayDict:
        x_std = self.x_scaler.transform(x)
        means: List[np.ndarray] = []
        variances: List[np.ndarray] = []
        self.model.eval()
        self.likelihood.eval()

        for start in range(0, len(x_std), self.batch_size):
            batch = torch.from_numpy(x_std[start : start + self.batch_size]).float().to(self.device)
            with gpytorch.settings.fast_pred_var():
                latent = self.model(batch)
                distribution = latent if self.uncertainty == "epistemic" else self.likelihood(latent)
            means.append(distribution.mean.cpu().numpy())
            variances.append(distribution.variance.cpu().numpy())

        mean = self.y_scaler.inverse_mean(np.concatenate(means)).reshape(-1)
        std = np.maximum(
            self.y_scaler.inverse_std(np.sqrt(np.maximum(np.concatenate(variances), 0.0))),
            self.std_floor,
        ).reshape(-1)
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        return {
            "mean": mean,
            "std": std,
            "native_lower": mean - z * std,
            "native_upper": mean + z * std,
            "conformal_scale": std,
            "score_type": "scaled_residual",
        }


class GAMRegressor(BaselineRegressor):
    """Partially linear GAM: spline(time) plus linear non-time covariates."""

    name = "gam"

    def __init__(
        self,
        *,
        n_knots: int = 8,
        degree: int = 3,
        ridge_alpha: float = 1.0,
        max_linear_features: int = 30,
        std_floor: float = 1e-6,
    ) -> None:
        self.n_knots = n_knots
        self.degree = degree
        self.ridge_alpha = ridge_alpha
        self.max_linear_features = max_linear_features
        self.std_floor = std_floor

    @staticmethod
    def _select_linear_features(x: np.ndarray, y: np.ndarray, maximum: int) -> np.ndarray:
        candidates = np.arange(max(0, x.shape[1] - 1))
        if maximum <= 0 or len(candidates) <= maximum:
            return candidates
        scores = []
        for index in candidates:
            feature = x[:, index]
            if np.std(feature) < 1e-12:
                score = 0.0
            else:
                score = abs(np.corrcoef(feature, y)[0, 1])
                if not np.isfinite(score):
                    score = 0.0
            scores.append(score)
        order = np.argsort(scores)[::-1][:maximum]
        return candidates[order]

    def _design(self, x: np.ndarray) -> np.ndarray:
        selected = x[:, self.linear_indices] if len(self.linear_indices) else np.empty((len(x), 0))
        return np.concatenate([selected, x[:, [-1]]], axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray, subject_ids: np.ndarray) -> "GAMRegressor":
        del subject_ids
        self.linear_indices = self._select_linear_features(x, y, self.max_linear_features)
        design = self._design(x)
        time_column = [design.shape[1] - 1]
        linear_columns = list(range(design.shape[1] - 1))

        transformers = []
        if linear_columns:
            transformers.append(("linear", StandardScaler(), linear_columns))
        transformers.append(
            (
                "time_spline",
                SplineTransformer(
                    n_knots=self.n_knots,
                    degree=self.degree,
                    include_bias=False,
                ),
                time_column,
            )
        )
        self.pipeline = Pipeline(
            [
                ("features", ColumnTransformer(transformers, remainder="drop")),
                ("ridge", Ridge(alpha=self.ridge_alpha)),
            ]
        )
        self.pipeline.fit(design, y)
        train_mean = self.pipeline.predict(design)
        self.residual_scale = robust_scale(y - train_mean, floor=self.std_floor)
        return self

    def predict(self, x: np.ndarray, alpha: float) -> ArrayDict:
        mean = self.pipeline.predict(self._design(x)).reshape(-1)
        std = np.full_like(mean, self.residual_scale, dtype=float)
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        return {
            "mean": mean,
            "std": std,
            "native_lower": mean - z * std,
            "native_upper": mean + z * std,
            "conformal_scale": std,
            "score_type": "scaled_residual",
        }


class LMMRegressor(BaselineRegressor):
    """Random-intercept/random-time-slope LMM; predictions for new subjects use fixed effects."""

    name = "lmm"

    def __init__(
        self,
        *,
        max_fixed_features: int = 20,
        maxiter: int = 500,
        std_floor: float = 1e-6,
    ) -> None:
        if sm is None:
            raise ImportError("LMM requires statsmodels.")
        self.max_fixed_features = max_fixed_features
        self.maxiter = maxiter
        self.std_floor = std_floor

    @staticmethod
    def _select_features(x: np.ndarray, y: np.ndarray, maximum: int) -> np.ndarray:
        candidates = np.arange(max(0, x.shape[1] - 1))
        if maximum <= 0 or len(candidates) <= maximum:
            return candidates
        scores = []
        for index in candidates:
            feature = x[:, index]
            if np.std(feature) < 1e-12:
                score = 0.0
            else:
                score = abs(np.corrcoef(feature, y)[0, 1])
                if not np.isfinite(score):
                    score = 0.0
            scores.append(score)
        return candidates[np.argsort(scores)[::-1][:maximum]]

    def _fixed_design(self, x: np.ndarray) -> np.ndarray:
        selected = x[:, self.fixed_indices] if len(self.fixed_indices) else np.empty((len(x), 0))
        combined = np.concatenate([selected, x[:, [-1]]], axis=1)
        standardized = self.fixed_scaler.transform(combined)
        return sm.add_constant(standardized, has_constant="add")

    def _random_design(self, x: np.ndarray) -> np.ndarray:
        time_std = (x[:, -1] - self.time_mean) / self.time_std
        return np.column_stack([np.ones(len(x)), time_std])

    def fit(self, x: np.ndarray, y: np.ndarray, subject_ids: np.ndarray) -> "LMMRegressor":
        self.fixed_indices = self._select_features(x, y, self.max_fixed_features)
        selected = x[:, self.fixed_indices] if len(self.fixed_indices) else np.empty((len(x), 0))
        combined = np.concatenate([selected, x[:, [-1]]], axis=1)
        self.fixed_scaler = StandardScaler().fit(combined)
        self.time_mean = float(np.mean(x[:, -1]))
        self.time_std = float(np.std(x[:, -1]))
        if self.time_std < 1e-6:
            self.time_std = 1.0

        exog = self._fixed_design(x)
        exog_re = self._random_design(x)
        model = sm.MixedLM(
            endog=y,
            exog=exog,
            groups=subject_ids.astype(str),
            exog_re=exog_re,
        )

        result = None
        errors: List[str] = []
        for method in ("lbfgs", "powell"):
            try:
                candidate = model.fit(
                    reml=True,
                    method=method,
                    maxiter=self.maxiter,
                    disp=False,
                )
                result = candidate
                if getattr(candidate, "converged", True):
                    break
            except Exception as exc:  # statsmodels emits several model-specific errors.
                errors.append(f"{method}: {exc}")

        if result is None:
            raise RuntimeError("LMM fitting failed. " + " | ".join(errors))
        if not getattr(result, "converged", True):
            warnings.warn("LMM did not report convergence; results should be inspected.")

        self.result = result
        train_mean = np.asarray(result.predict(exog=exog)).reshape(-1)
        self.residual_scale = robust_scale(y - train_mean, floor=self.std_floor)
        return self

    def predict(self, x: np.ndarray, alpha: float) -> ArrayDict:
        # Test subjects are new clusters, so random effects have conditional mean zero.
        mean = np.asarray(self.result.predict(exog=self._fixed_design(x))).reshape(-1)
        std = np.full_like(mean, self.residual_scale, dtype=float)
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        return {
            "mean": mean,
            "std": std,
            "native_lower": mean - z * std,
            "native_upper": mean + z * std,
            "conformal_scale": std,
            "score_type": "scaled_residual",
        }


def build_model(
    model_name: str,
    *,
    seed: int,
    device: torch.device,
    args,
) -> BaselineRegressor:
    common_mlp = dict(
        seed=seed,
        device=device,
        hidden_dims=(args.hidden_dim_1, args.hidden_dim_2),
        epochs=args.mlp_epochs,
        learning_rate=args.mlp_learning_rate,
        weight_decay=args.mlp_weight_decay,
        batch_size=args.batch_size,
        std_floor=args.std_floor,
    )

    if model_name == "mlp":
        return TorchMLPRegressor(dropout=0.0, **common_mlp)
    if model_name == "drmc":
        return MCDropoutRegressor(
            dropout=args.dropout,
            mc_samples=args.mc_samples,
            **common_mlp,
        )
    if model_name == "bootstrap":
        return BootstrapMLPRegressor(
            seed=seed,
            device=device,
            ensemble_size=args.bootstrap_models,
            hidden_dims=(args.hidden_dim_1, args.hidden_dim_2),
            epochs=args.mlp_epochs,
            learning_rate=args.mlp_learning_rate,
            weight_decay=args.mlp_weight_decay,
            batch_size=args.batch_size,
            std_floor=args.std_floor,
        )
    if model_name == "dqr":
        return DeepQuantileRegressor(
            quantiles=(args.dqr_lower, 0.5, args.dqr_upper),
            dropout=args.dropout,
            **common_mlp,
        )
    if model_name == "exact_gp":
        return ExactGPRegressor(
            seed=seed,
            device=device,
            iterations=args.exact_gp_iterations,
            learning_rate=args.exact_gp_learning_rate,
            uncertainty=args.uncertainty,
            std_floor=args.std_floor,
            prediction_batch_size=args.prediction_batch_size,
            max_cholesky_size=args.max_cholesky_size,
        )
    if model_name == "dmegp":
        return DMEGPRegressor(
            seed=seed,
            device=device,
            latent_dim=args.dmegp_latent_dim,
            inducing_points=args.dmegp_inducing_points,
            epochs=args.dmegp_epochs,
            learning_rate=args.dmegp_learning_rate,
            weight_decay=args.dmegp_weight_decay,
            batch_size=args.batch_size,
            uncertainty=args.uncertainty,
            std_floor=args.std_floor,
        )
    if model_name == "lmm":
        return LMMRegressor(
            max_fixed_features=args.lmm_max_fixed_features,
            maxiter=args.lmm_maxiter,
            std_floor=args.std_floor,
        )
    if model_name == "gam":
        return GAMRegressor(
            n_knots=args.gam_knots,
            degree=args.gam_degree,
            ridge_alpha=args.gam_ridge_alpha,
            max_linear_features=args.gam_max_linear_features,
            std_floor=args.std_floor,
        )
    raise ValueError(f"Unknown baseline model: {model_name}")
