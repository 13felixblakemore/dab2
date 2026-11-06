import os
import random
from collections.abc import Hashable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.cauchy import Cauchy


class RadialBasisFunction2D:
    def __init__(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        dim: int,
        sigma: float,
        device: str = "cpu",
    ):
        self.dim = dim
        self.sigma = sigma
        self.device = device

        xlist = np.linspace(x_bounds[0], x_bounds[1], dim)
        ylist = np.linspace(y_bounds[0], y_bounds[1], dim)
        self.XX, self.YY = np.meshgrid(xlist, ylist)

        self.x_mu = self.XX.flatten()
        self.y_mu = self.YY.flatten()

        self.XX_t = torch.tensor(self.XX, dtype=torch.float32, device=device)
        self.YY_t = torch.tensor(self.YY, dtype=torch.float32, device=device)
        self.x_mu_t = torch.tensor(self.x_mu, dtype=torch.float32, device=device)
        self.y_mu_t = torch.tensor(self.y_mu, dtype=torch.float32, device=device)

    def transform(self, A):
        if isinstance(A, torch.Tensor):
            A = A.to(self.device)

            distance = (A[:, 0:1] - self.x_mu_t.unsqueeze(0)) ** 2 + (
                A[:, 1:2] - self.y_mu_t.unsqueeze(0)
            ) ** 2

            weights = torch.exp(-distance / (2 * (self.sigma**2)))
            return weights / weights.sum(dim=1, keepdim=True)

        else:
            A = np.asarray(A)
            distance = (A[:, 0:1] - self.x_mu[None]) ** 2 + (
                A[:, 1:2] - self.y_mu[None]
            ) ** 2
            weights = np.exp(-distance / (2 * (self.sigma**2)))
            return weights / weights.sum(axis=1, keepdims=True)

    def inverse_transform(self, A):
        if isinstance(A, torch.Tensor):
            index = torch.argmax(A, dim=1)
            i, j = self._1d_index_to_2d_index(index)
            return torch.stack((self.XX_t[i, j], self.YY_t[i, j]), dim=1)
        else:
            A = np.asarray(A)
            index = np.argmax(A, axis=1)
            i, j = self._1d_index_to_2d_index(index)
            return np.column_stack((self.XX[i, j], self.YY[i, j]))

    def _1d_index_to_2d_index(self, index):
        return index // self.dim, index % self.dim


class IdentityFeaturiser:
    """A pass-through featuriser used when none is provided."""

    def transform(self, obs):
        return obs

    def inverse_transform(self, obs):
        return obs


class ReplayBuffer:
    def __init__(self, size: int, obs_dim: int, n_actions: int, device: str = "cuda"):
        self.max_size = size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros((size,), dtype=torch.long, device=device)
        self.obs_next = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.next_actions = torch.zeros(
            (size, n_actions), dtype=torch.bool, device=device
        )

    def add(self, obs, action, obs_next, next_actions):
        self.obs[self.ptr] = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        )
        self.action[self.ptr] = torch.as_tensor(
            action, dtype=torch.long, device=self.device
        )
        self.obs_next[self.ptr] = torch.as_tensor(
            obs_next, dtype=torch.float32, device=self.device
        )
        self.next_actions[self.ptr] = torch.as_tensor(
            next_actions, dtype=torch.bool, device=self.device
        )

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idxes = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "obs": self.obs[idxes],
            "action": self.action[idxes],
            "obs_next": self.obs_next[idxes],
            "next_actions": self.next_actions[idxes],
        }


class BackwardNet(nn.Module):
    def __init__(self, obs_dim: int, embed_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()

        layers = []
        layers.extend([nn.Linear(obs_dim, hidden_dim), nn.ReLU()])
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ForwardNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_actions = n_actions

        layers = []

        layers.extend([nn.Linear(obs_dim + embed_dim, hidden_dim), nn.ReLU()])
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, embed_dim * n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        w = w / torch.sqrt(
            1 + torch.norm(w, dim=-1, keepdim=True) ** 2 / self.embed_dim
        )

        x = torch.cat([obs, w], dim=1)

        forward_value = self.net(x)
        return forward_value.reshape(-1, self.embed_dim, self.n_actions)


class FBLearner:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        embed_dim: int,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        reg_coef: float = 1.0,
        temperature: float = 200.0,
        polyak_coef: float = 0.95,
        batch_size: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        buffer_size: int = 1_000_000,
        device: str = "cuda",
        buffer_device: str = "cpu",
        featuriser: RadialBasisFunction2D | None = None,
        max_grad_norm: None | float = None,
        zero_z_p: float = 0.0,
    ):
        # if CPU, set CUDA_VISIBLE_DEVICES to empty to avoid GPU memory allocation
        if not str(device).startswith("cuda") and not str(buffer_device).startswith(
            "cuda"
        ):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        self.config = {
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "embed_dim": embed_dim,
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "reg_coef": reg_coef,
            "polyak_coef": polyak_coef,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "buffer_size": buffer_size,
            "featuriser": featuriser,
            "zero_z_p": zero_z_p,
            "max_grad_norm": max_grad_norm,
        }
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.embed_dim = embed_dim

        self.discount_factor = discount_factor
        self.reg_coef = reg_coef
        self.temperature = temperature
        self.polyak_coef = polyak_coef

        self.device = device

        self.cauchy = Cauchy(torch.tensor([0.0]), torch.tensor([0.5]))

        self.feature_size = obs_dim if featuriser is None else featuriser.dim**2

        self.forward_network = ForwardNet(
            obs_dim=self.feature_size,
            n_actions=n_actions,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)
        self.backward_network = BackwardNet(
            obs_dim=self.feature_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)

        self.forward_target_network = ForwardNet(
            obs_dim=self.feature_size,
            n_actions=n_actions,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)
        self.backward_target_network = BackwardNet(
            obs_dim=self.feature_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)

        self.forward_target_network.load_state_dict(self.forward_network.state_dict())
        self.backward_target_network.load_state_dict(self.backward_network.state_dict())

        f_params = [param for param in self.forward_network.parameters()]
        b_params = [param for param in self.backward_network.parameters()]
        self.optim = torch.optim.Adam(f_params + b_params, lr=learning_rate)

        self.buffer = ReplayBuffer(
            size=buffer_size,
            obs_dim=obs_dim,
            n_actions=n_actions,
            device=buffer_device,
        )

        self.max_grad_norm = max_grad_norm
        self.featuriser: IdentityFeaturiser | RadialBasisFunction2D = (
            featuriser if featuriser is not None else IdentityFeaturiser()
        )  # type: ignore

        self.zero_z_p = zero_z_p

    def update(
        self,
        obs: Hashable,
        action: int,
        next_obs: Hashable,
        terminal: bool,
        next_actions: list[int],
    ) -> None:
        self.buffer.add(
            obs,
            action,
            next_obs,
            [1 if a in next_actions else 0 for a in range(self.n_actions)],
        )

    def update_network(self) -> float:
        transitions = self.buffer.sample(self.batch_size)
        other_transitions = self.buffer.sample(self.batch_size)

        obs_tensor = transitions["obs"].to(self.device)
        obs_next_tensor = transitions["obs_next"].to(self.device)
        actions_tensor = transitions["action"].to(self.device)
        next_available_actions_tensor = transitions["next_actions"].to(self.device)
        obs_other_tensor = other_transitions["obs"].to(self.device)

        obs_tensor = self.featuriser.transform(obs_tensor)
        obs_next_tensor = self.featuriser.transform(obs_next_tensor)
        obs_other_tensor = self.featuriser.transform(obs_other_tensor)

        w = self.sample_cauchy_ball(self.batch_size)

        zero_z_idxs = torch.rand(self.batch_size) > (1 - self.zero_z_p)
        w[zero_z_idxs] = 0

        with torch.no_grad():
            f = self.forward_target_network(obs_next_tensor, w)
            z = torch.einsum("sda, sd -> sa", f, w)
            z[zero_z_idxs] = 0.0
            logits = z / self.temperature
            logits = logits.masked_fill(~next_available_actions_tensor, -float("inf"))
            pi = F.softmax(logits, dim=-1)

            f_next = torch.einsum("sda, sa -> sd", f, pi)

            b_next = self.backward_target_network(obs_other_tensor)
            z_next = torch.einsum("sd, td -> st", f_next, b_next)

        idxs = actions_tensor[:, None].repeat(1, self.embed_dim)[:, :, None]
        f = self.forward_network(obs_tensor, w).gather(-1, idxs).squeeze(-1)
        b = self.backward_network(obs_tensor)
        b_other = self.backward_network(obs_other_tensor)

        z_diag = torch.einsum("sd, sd -> s", f, b)
        z = torch.einsum("sd, td -> st", f, b_other)
        fb_loss = (
            0.5 * (z - self.discount_factor * z_next).pow(2).mean() - z_diag.mean()
        )

        b_b_other = torch.einsum("sd, xd -> sx", b, b_other)
        b_b_other_detach = torch.einsum("sd, xd -> sx", b, b_other.detach())
        b_b_detach = torch.einsum("sd, sd -> s", b, b.detach())
        reg_loss = (b_b_other_detach * b_b_other.detach()).mean() - b_b_detach.mean()
        # reg_loss = (b_b_detach * b_b_other.detach()).mean() - b_b_other_detach.mean()  # theirs

        fb_loss += self.reg_coef * reg_loss

        self.optim.zero_grad()
        fb_loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.forward_network.parameters(), max_norm=self.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.backward_network.parameters(), max_norm=self.max_grad_norm
            )

        self.optim.step()

        return fb_loss.item()

    def soft_update_target_networks(self) -> None:
        for target_param, param in zip(
            self.forward_target_network.parameters(), self.forward_network.parameters()
        ):
            target_param.data.copy_(
                (1 - self.polyak_coef) * param.data
                + self.polyak_coef * target_param.data
            )

        for target_param, param in zip(
            self.backward_target_network.parameters(),
            self.backward_network.parameters(),
        ):
            target_param.data.copy_(
                (1 - self.polyak_coef) * param.data
                + self.polyak_coef * target_param.data
            )

    def sample_cauchy_ball(self, n, eps: float = 1e-10):
        gaussian_rdv = torch.normal(
            mean=0, std=1, size=(n, self.embed_dim), device=self.device
        )
        gaussian_rdv /= torch.norm(gaussian_rdv, dim=-1, keepdim=True) + eps
        cauchy_rdv = self.cauchy.sample((n,)).to(self.device)
        w = np.sqrt(self.embed_dim) * gaussian_rdv * cauchy_rdv
        return w

    @torch.no_grad()
    def embed_B(self, obs) -> torch.Tensor:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs = self.featuriser.transform(obs)

        return self.backward_network(obs).cpu()

    @torch.no_grad()
    def embed_F(self, obs, z: torch.Tensor | None = None, action=None) -> torch.Tensor:
        obs = torch.atleast_2d(
            torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        )
        obs = self.featuriser.transform(obs)

        if z is None:
            z = torch.zeros((obs.size(0), self.embed_dim), device=self.device)

        z = self.forward_network(obs, z)

        if action is None:
            return z.cpu()

        action = torch.atleast_1d(
            torch.as_tensor(action, dtype=torch.long, device=self.device)
        )

        return z[torch.arange(len(action), device=self.device), :, action].cpu()

    @torch.no_grad()
    def export_sr(self, observations, available_actions) -> torch.Tensor:
        observations = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        assert observations.dim() == 2, (
            "export_sr is batch-only: expects (n, obs_dim) with n >= 2"
        )

        observations = self.featuriser.transform(observations)

        available_actions = torch.as_tensor(
            available_actions, dtype=torch.bool, device=self.device
        )

        # embed F and mean over available actions
        f = self.forward_network(
            observations,
            torch.zeros((observations.size(0), self.embed_dim), device=self.device),
        )

        mask = available_actions.unsqueeze(1)
        f_masked = f * mask
        f_sum = f_masked.sum(dim=2)

        counts = available_actions.sum(dim=1, keepdim=True)
        counts = counts.clamp(min=1)

        f_mean = f_sum / counts

        # embed B
        b = self.backward_network(observations)

        return torch.inner(f_mean, b).clamp(min=0.0).cpu()

    def save(self, path: str | Path) -> None:
        """Saves the networks and optimizer state dictionaries to a file."""
        checkpoint = {
            "config": self.config,
            "forward_network": self.forward_network.state_dict(),
            "backward_network": self.backward_network.state_dict(),
            "forward_target_network": self.forward_target_network.state_dict(),
            "backward_target_network": self.backward_target_network.state_dict(),
            "optimizer": self.optim.state_dict(),
        }
        torch.save(checkpoint, Path(path))

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str = "cpu", buffer_device: str = "cpu"
    ) -> "FBLearner":
        """Loads a learner from a checkpoint file."""
        # if CPU, set CUDA_VISIBLE_DEVICES to empty to avoid GPU memory allocation
        if not str(device).startswith("cuda") and not str(buffer_device).startswith(
            "cuda"
        ):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        learner = cls(
            **checkpoint["config"], device=device, buffer_device=buffer_device
        )
        learner.forward_network.load_state_dict(checkpoint["forward_network"])
        learner.backward_network.load_state_dict(checkpoint["backward_network"])
        learner.forward_target_network.load_state_dict(
            checkpoint["forward_target_network"]
        )
        learner.backward_target_network.load_state_dict(
            checkpoint["backward_target_network"]
        )
        learner.optim.load_state_dict(checkpoint["optimizer"])
        return learner

    def act(self, obs, w, target_network=False):
        obs = self.featuriser.transform(obs)
        if target_network:
            f = self.forward_target_network(obs, w)
        else:
            f = self.forward_network(obs, w)
        z = torch.einsum("sda, sd -> sa", f, w)
        y = z.max(1)[1]
        return y

    def act_e_greedy(self, obs, g, available_actions: list | None = None, epsilon=0.2):
        if available_actions is None:
            return (
                random.randrange(self.n_actions)
                if random.random() < epsilon
                else self.act(obs, g).item()
            )
        else:
            return (
                random.choice(available_actions)
                if random.random() < epsilon
                else self.act(obs, g).item()
            )
