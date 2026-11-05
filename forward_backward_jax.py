"""JAX port of forward_backward.py (requires JAX and NumPy, no Flax/Optax).

The architecture, objectives, detach locations, Adam defaults, and explicit
target-update schedule match the PyTorch implementation. Random draws and
floating-point reductions are not bit-identical across frameworks.

For GPU training use device="cuda", buffer_device="cuda" and
update_networks(40), followed by soft_update_target_networks() at the same
frequency as the original. update_network() still returns a Python float;
update_networks() returns device losses without synchronizing on every step.
Embedding/export helpers also return device arrays; use jax.device_get when
host output is needed. Checkpoints use this module's format, not torch.save.
"""

import pickle
import random
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _device(device):
    if isinstance(device, jax.Device):
        return device
    name, _, index = str(device).partition(":")
    backend = "gpu" if name in ("cuda", "gpu") else name
    return jax.devices(backend)[int(index or 0)]


def _array(value, device, dtype=jnp.float32):
    # Conversion on the selected device avoids an intermediate default-GPU copy
    # when constructing CPU replay storage.
    with jax.default_device(device):
        return jax.device_put(jnp.asarray(value, dtype=dtype), device)


def mask_action_scores(scores, action_mask):
    """Apply a boolean availability mask before argmax or softmax sampling."""
    return jnp.where(action_mask, scores, -jnp.inf)


class RadialBasisFunction2D:
    def __init__(self, x_bounds, y_bounds, dim, sigma, device="cpu"):
        self.dim, self.sigma, self.device = dim, sigma, device
        self.x_bounds, self.y_bounds = x_bounds, y_bounds
        self.XX, self.YY = np.meshgrid(
            np.linspace(*x_bounds, dim), np.linspace(*y_bounds, dim)
        )
        self.x_mu, self.y_mu = self.XX.ravel(), self.YY.ravel()
        target = _device(device)
        self.XX_t, self.YY_t = _array(self.XX, target), _array(self.YY, target)
        self.x_mu_t, self.y_mu_t = self.XX_t.ravel(), self.YY_t.ravel()

    def transform(self, A):
        # Arrays/tracers stay in JAX; NumPy callers retain the original host API.
        if isinstance(A, (jax.Array, jax.core.Tracer)):
            distance = (A[:, :1] - self.x_mu_t[None]) ** 2 + (
                A[:, 1:2] - self.y_mu_t[None]
            ) ** 2
            weights = jnp.exp(-distance / (2 * self.sigma**2))
            return weights / weights.sum(axis=1, keepdims=True)
        A = np.asarray(A)
        distance = (A[:, :1] - self.x_mu[None]) ** 2 + (
            A[:, 1:2] - self.y_mu[None]
        ) ** 2
        weights = np.exp(-distance / (2 * self.sigma**2))
        return weights / weights.sum(axis=1, keepdims=True)

    def inverse_transform(self, A):
        if isinstance(A, (jax.Array, jax.core.Tracer)):
            i, j = self._1d_index_to_2d_index(jnp.argmax(A, axis=1))
            return jnp.stack((self.XX_t[i, j], self.YY_t[i, j]), axis=1)
        i, j = self._1d_index_to_2d_index(np.argmax(np.asarray(A), axis=1))
        return np.column_stack((self.XX[i, j], self.YY[i, j]))

    def _1d_index_to_2d_index(self, index):
        return index // self.dim, index % self.dim


class IdentityFeaturiser:
    def transform(self, obs):
        return obs

    def inverse_transform(self, obs):
        return obs


@partial(jax.jit, donate_argnums=(0,))
def _insert(storage, index, batch):
    return jax.tree_util.tree_map(lambda a, b: a.at[index].set(b), storage, batch)


@partial(jax.jit, static_argnames=("batch_size",))
def _sample(storage, key, size, batch_size):
    idx = jax.random.randint(key, (batch_size,), 0, size)
    return jax.tree_util.tree_map(lambda a: a[idx], storage)


class ReplayBuffer:
    def __init__(self, size, obs_dim, n_actions, device="cuda", seed=0):
        self.max_size, self.device = size, _device(device)
        self.ptr = self.size = 0
        self._host = self.device.platform == "cpu"
        self._rng = np.random.default_rng(seed)
        self._key = jax.device_put(jax.random.PRNGKey(seed), self.device)
        shapes = {"obs": (size, obs_dim), "action": (size,),
                  "obs_next": (size, obs_dim), "next_actions": (size, n_actions)}
        self.storage = {}
        for name, shape in shapes.items():
            dtype = np.int32 if name == "action" else np.bool_ if name == "next_actions" else np.float32
            if self._host:
                self.storage[name] = np.zeros(shape, dtype=dtype)
            else:
                with jax.default_device(self.device):
                    self.storage[name] = jnp.zeros(shape, dtype=dtype)

    def __getattr__(self, name):
        if name in ("obs", "action", "obs_next", "next_actions"):
            return self.storage[name]
        raise AttributeError(name)

    def add(self, obs, action, obs_next, next_actions):
        self.add_batch({"obs": [obs], "action": [action], "obs_next": [obs_next],
                        "next_actions": [next_actions]})

    def add_batch(self, batch):
        """Insert a batch with one dispatch/transfer rather than one per row."""
        count = len(batch["action"])
        if count == 0:
            return
        # If a batch exceeds capacity, only its last capacity entries survive.
        skip = max(0, count - self.max_size)
        index = (self.ptr + np.arange(skip, count)) % self.max_size
        if self._host:
            for name, value in batch.items():
                self.storage[name][index] = np.asarray(value)[skip:]
        else:
            values = {name: _array(value, self.device, self.storage[name].dtype)[skip:]
                      for name, value in batch.items()}
            self.storage = _insert(self.storage, _array(index, self.device, jnp.int32), values)
        self.ptr = (self.ptr + count) % self.max_size
        self.size = min(self.size + count, self.max_size)

    def sample(self, batch_size):
        if not self.size:
            raise ValueError("Cannot sample an empty replay buffer")
        if self._host:
            idx = self._rng.integers(self.size, size=batch_size)
            return {name: value[idx] for name, value in self.storage.items()}
        self._key, key = jax.random.split(self._key)
        return _sample(self.storage, key, self.size, batch_size)


def _init_mlp(key, sizes):
    # torch.nn.Linear initializes both weight and bias U(-1/sqrt(fan_in), ...).
    keys = jax.random.split(key, 2 * (len(sizes) - 1))
    return tuple({
        "weight": jax.random.uniform(keys[2*i], (din, dout), minval=-din**-0.5, maxval=din**-0.5),
        "bias": jax.random.uniform(keys[2*i+1], (dout,), minval=-din**-0.5, maxval=din**-0.5),
    } for i, (din, dout) in enumerate(zip(sizes[:-1], sizes[1:])))


def _mlp(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(jnp.matmul(x, layer["weight"], precision=jax.lax.Precision.HIGHEST) + layer["bias"])
    return jnp.matmul(x, params[-1]["weight"], precision=jax.lax.Precision.HIGHEST) + params[-1]["bias"]


class BackwardNet:
    def __init__(self, obs_dim, embed_dim, hidden_dim, num_layers, key=None):
        self.params = _init_mlp(jax.random.PRNGKey(0) if key is None else key,
                                [obs_dim] + [hidden_dim] * (num_layers + 1) + [embed_dim])

    def apply(self, params, x):
        return _mlp(params, x)

    def __call__(self, x):
        return self.apply(self.params, x)

    forward = __call__


class ForwardNet:
    def __init__(self, obs_dim, n_actions, embed_dim, hidden_dim, num_layers, key=None):
        self.embed_dim, self.n_actions = embed_dim, n_actions
        self.params = _init_mlp(jax.random.PRNGKey(0) if key is None else key,
                                [obs_dim + embed_dim] + [hidden_dim] * (num_layers + 1)
                                + [embed_dim * n_actions])

    def apply(self, params, obs, w):
        w = w / jnp.sqrt(1 + jnp.sum(w**2, axis=-1, keepdims=True) / self.embed_dim)
        return _mlp(params, jnp.concatenate((obs, w), axis=1)).reshape(-1, self.embed_dim, self.n_actions)

    def __call__(self, obs, w):
        return self.apply(self.params, obs, w)

    forward = __call__


def _cauchy_ball(key, n, embed_dim, eps=1e-10):
    gaussian_key, cauchy_key = jax.random.split(key)
    gaussian = jax.random.normal(gaussian_key, (n, embed_dim))
    gaussian /= jnp.linalg.norm(gaussian, axis=-1, keepdims=True) + eps
    return jnp.sqrt(float(embed_dim)) * gaussian * (0.5 * jax.random.cauchy(cauchy_key, (n, 1)))


def _clip(grads, max_norm):
    norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


def _adam(params, grads, optim, learning_rate):
    count, m, v = optim
    count += 1
    m = jax.tree_util.tree_map(lambda m, g: 0.9*m + 0.1*g, m, grads)
    v = jax.tree_util.tree_map(lambda v, g: 0.999*v + 0.001*g**2, v, grads)
    # Epsilon is outside sqrt, matching torch.optim.Adam (not AdamW).
    params = jax.tree_util.tree_map(
        lambda p, m, v: p - learning_rate * (m / (1 - 0.9**count))
        / (jnp.sqrt(v / (1 - 0.999**count)) + 1e-8), params, m, v)
    return params, (count, m, v)


class FBLearner:
    def __init__(self, obs_dim, n_actions, embed_dim, learning_rate=1e-4,
                 discount_factor=0.99, reg_coef=1.0, temperature=200.0,
                 polyak_coef=0.95, batch_size=128, hidden_dim=128, num_layers=1,
                 buffer_size=1_000_000, device="cuda", buffer_device="cpu",
                 featuriser=None, max_grad_norm=None, zero_z_p=0.0, seed=0,
                 z_sampling="cauchy"):
        if z_sampling not in ("cauchy", "replay"):
            raise ValueError("z_sampling must be 'cauchy' or 'replay'")
        self.z_sampling = z_sampling
        self.device = _device(device)
        self.n_actions, self.batch_size, self.embed_dim = n_actions, batch_size, embed_dim
        self.discount_factor, self.reg_coef = discount_factor, reg_coef
        self.temperature, self.polyak_coef = temperature, polyak_coef
        self.max_grad_norm, self.zero_z_p = max_grad_norm, zero_z_p
        self.learning_rate = learning_rate
        self.config = dict(obs_dim=obs_dim, n_actions=n_actions, embed_dim=embed_dim,
                           learning_rate=learning_rate, discount_factor=discount_factor,
                           reg_coef=reg_coef, temperature=temperature, polyak_coef=polyak_coef,
                           batch_size=batch_size, hidden_dim=hidden_dim, num_layers=num_layers,
                           buffer_size=buffer_size, max_grad_norm=max_grad_norm,
                           zero_z_p=zero_z_p, seed=seed, z_sampling=z_sampling)
        self._python_rng = random.Random(seed)
        if isinstance(featuriser, RadialBasisFunction2D):
            featuriser = RadialBasisFunction2D(featuriser.x_bounds, featuriser.y_bounds,
                                             featuriser.dim, featuriser.sigma, self.device)
        self.featuriser = featuriser if featuriser is not None else IdentityFeaturiser()
        self.feature_size = obs_dim if featuriser is None else featuriser.dim**2
        with jax.default_device(self.device):
            key, fkey, bkey = jax.random.split(jax.random.PRNGKey(seed), 3)
            self.forward_network = ForwardNet(self.feature_size, n_actions, embed_dim, hidden_dim, num_layers, fkey)
            self.backward_network = BackwardNet(self.feature_size, embed_dim, hidden_dim, num_layers, bkey)
            self.forward_target_network = ForwardNet(self.feature_size, n_actions, embed_dim, hidden_dim, num_layers, fkey)
            self.backward_target_network = BackwardNet(self.feature_size, embed_dim, hidden_dim, num_layers, bkey)
            params = dict(f=self.forward_network.params, b=self.backward_network.params)
            zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
            self.state = dict(params=params, target=params,
                              optim=(jnp.asarray(0, jnp.int32), zeros, zeros), key=key)
        self.buffer = ReplayBuffer(buffer_size, obs_dim, n_actions, buffer_device, seed)
        # All changing parameters, optimizer state, RNG, and replay are explicit
        # JIT arguments, so compiled functions never capture stale model weights.
        self._train = jax.jit(self._train_impl)
        self._train_many = jax.jit(self._train_many_impl, static_argnums=(3,))
        self._soft_update = jax.jit(lambda target, params: jax.tree_util.tree_map(
            lambda t, p: (1-self.polyak_coef)*p + self.polyak_coef*t, target, params))
        self._embed_b = jax.jit(lambda p, x: self.backward_network.apply(p, self.featuriser.transform(x)))
        self._embed_f = jax.jit(lambda p, x, w: self.forward_network.apply(p, self.featuriser.transform(x), w))
        self._draw_w = jax.jit(partial(_cauchy_ball, embed_dim=embed_dim), static_argnames=("n",))
        self._act = jax.jit(lambda p, x, w, mask: jnp.argmax(mask_action_scores(jnp.einsum(
            "sda,sd->sa", self.forward_network.apply(p, self.featuriser.transform(x), w), w), mask), axis=1))
        self._export = jax.jit(self._export_impl)

    def _sync_networks(self):
        self.forward_network.params, self.backward_network.params = self.state["params"]["f"], self.state["params"]["b"]
        self.forward_target_network.params, self.backward_target_network.params = self.state["target"]["f"], self.state["target"]["b"]

    def update(self, obs, action, next_obs, terminal, next_actions):
        # As in the original, terminal is deliberately unused.
        self.buffer.add(obs, action, next_obs, [a in next_actions for a in range(self.n_actions)])

    def _loss(self, params, target, transitions, other, w, zero):
        obs = self.featuriser.transform(transitions["obs"])
        obs_next = self.featuriser.transform(transitions["obs_next"])
        obs_other = self.featuriser.transform(other["obs"])
        stop = jax.lax.stop_gradient
        f_next_all = self.forward_network.apply(target["f"], obs_next, w)
        z_policy = jnp.einsum("sda,sd->sa", f_next_all, w)
        z_policy = jnp.where(zero[:, None], 0.0, z_policy)
        logits = jnp.where(transitions["next_actions"], z_policy/self.temperature, -jnp.inf)
        pi = jax.nn.softmax(logits, axis=-1)
        f_next = jnp.einsum("sda,sa->sd", f_next_all, pi)
        b_next = self.backward_network.apply(target["b"], obs_other)
        z_next = stop(jnp.matmul(f_next, b_next.T, precision=jax.lax.Precision.HIGHEST))
        f_all = self.forward_network.apply(params["f"], obs, w)
        f = jnp.take_along_axis(f_all, transitions["action"][:, None, None], axis=-1).squeeze(-1)
        b = self.backward_network.apply(params["b"], obs)
        b_other = self.backward_network.apply(params["b"], obs_other)
        z_diag = jnp.sum(f*b, axis=1)
        z = jnp.matmul(f, b_other.T, precision=jax.lax.Precision.HIGHEST)
        fb_loss = 0.5*jnp.mean((z-self.discount_factor*z_next)**2) - z_diag.mean()
        b_b_other = jnp.matmul(b, b_other.T, precision=jax.lax.Precision.HIGHEST)
        b_b_other_detach = jnp.matmul(b, stop(b_other).T, precision=jax.lax.Precision.HIGHEST)
        b_b_detach = jnp.sum(b*stop(b), axis=1)
        reg_loss = jnp.mean(b_b_other_detach*stop(b_b_other)) - b_b_detach.mean()
        return fb_loss + self.reg_coef*reg_loss

    def _training_tasks(self, params, key, goal_obs=None):
        if self.z_sampling == "replay":
            if goal_obs is None:
                raise ValueError("Replay task sampling requires independent goal observations")
            return jax.lax.stop_gradient(self.backward_network.apply(
                params["b"], self.featuriser.transform(goal_obs)))
        return _cauchy_ball(key, self.batch_size, self.embed_dim)

    def _train_impl(self, state, transitions, other, goal_obs=None):
        key, wkey, zkey = jax.random.split(state["key"], 3)
        w = self._training_tasks(state["params"], wkey, goal_obs)
        zero = jax.random.uniform(zkey, (self.batch_size,)) > (1-self.zero_z_p)
        w = jnp.where(zero[:, None], 0.0, w)
        loss, grads = jax.value_and_grad(self._loss)(state["params"], state["target"], transitions, other, w, zero)
        if self.max_grad_norm is not None:
            grads = {name: _clip(value, self.max_grad_norm) for name, value in grads.items()}
        params, optim = _adam(state["params"], grads, state["optim"], self.learning_rate)
        return dict(params=params, target=state["target"], optim=optim, key=key), loss

    def update_network(self, return_device_loss=False):
        transitions = {k: _array(v, self.device, v.dtype) for k, v in self.buffer.sample(self.batch_size).items()}
        other = self.buffer.sample(self.batch_size)
        other = {"obs": _array(other["obs"], self.device)}
        goal_obs = (_array(self.buffer.sample(self.batch_size)["obs"], self.device)
                    if self.z_sampling == "replay" else None)
        self.state, loss = self._train(self.state, transitions, other, goal_obs)
        self._sync_networks()
        return loss if return_device_loss else float(loss)

    def _train_many_impl(self, state, storage, size, num_updates):
        def body(state, _):
            if self.z_sampling == "replay":
                key, k1, k2, kg = jax.random.split(state["key"], 4)
                goal_obs = storage["obs"][jax.random.randint(kg, (self.batch_size,), 0, size)]
            else:
                key, k1, k2 = jax.random.split(state["key"], 3)
                goal_obs = None
            idx = jax.random.randint(k1, (self.batch_size,), 0, size)
            other_idx = jax.random.randint(k2, (self.batch_size,), 0, size)
            transitions = jax.tree_util.tree_map(lambda a: a[idx], storage)
            other = {"obs": storage["obs"][other_idx]}
            return self._train_impl(dict(state, key=key), transitions, other, goal_obs)
        return jax.lax.scan(body, state, None, length=num_updates)

    def update_networks(self, num_updates):
        """Run consecutive updates without changing the target-update schedule.

        GPU replay is sampled inside a single compiled scan. CPU replay keeps
        sampling on the host and transfers only minibatches (never the full
        buffer). Returned losses remain on the learner device.
        """
        if num_updates < 1:
            raise ValueError("num_updates must be positive")
        if not self.buffer.size:
            raise ValueError("Cannot sample an empty replay buffer")
        if self.buffer.device == self.device and not self.buffer._host:
            self.state, losses = self._train_many(self.state, self.buffer.storage, self.buffer.size, num_updates)
            self._sync_networks()
            return losses
        return jnp.stack([self.update_network(return_device_loss=True) for _ in range(num_updates)])

    def soft_update_target_networks(self):
        self.state = dict(self.state, target=self._soft_update(self.state["target"], self.state["params"]))
        self._sync_networks()

    def sample_cauchy_ball(self, n, eps=1e-10):
        key, draw_key = jax.random.split(self.state["key"])
        self.state = dict(self.state, key=key)
        return self._draw_w(draw_key, n=n, eps=eps)

    def sample_task_vectors(self, n):
        """Draw from the configured task distribution; replay mode needs data.

        No renormalisation is applied. zero_z_p is applied separately in training.
        """
        if self.z_sampling == "cauchy":
            return self.sample_cauchy_ball(n)
        obs = _array(self.buffer.sample(n)["obs"], self.device)
        return jax.lax.stop_gradient(self._embed_b(self.state["params"]["b"], obs))

    def embed_B(self, obs):
        return self._embed_b(self.state["params"]["b"], _array(obs, self.device))

    def embed_F(self, obs, z=None, action=None):
        obs = jnp.atleast_2d(_array(obs, self.device))
        z = _array(np.zeros((obs.shape[0], self.embed_dim)), self.device) if z is None else _array(z, self.device)
        f = self._embed_f(self.state["params"]["f"], obs, z)
        if action is None:
            return f
        action = jnp.atleast_1d(_array(action, self.device, jnp.int32))
        return f[jnp.arange(len(action)), :, action]

    def _export_impl(self, params, observations, available_actions):
        obs = self.featuriser.transform(observations)
        f = self.forward_network.apply(params["f"], obs, jnp.zeros((obs.shape[0], self.embed_dim)))
        f_mean = (f*available_actions[:, None, :]).sum(axis=2) / jnp.maximum(available_actions.sum(axis=1, keepdims=True), 1)
        b = self.backward_network.apply(params["b"], obs)
        return jnp.maximum(jnp.matmul(f_mean, b.T, precision=jax.lax.Precision.HIGHEST), 0.0)

    def export_sr(self, observations, available_actions):
        observations = _array(observations, self.device)
        assert observations.ndim == 2, "export_sr is batch-only: expects (n, obs_dim) with n >= 2"
        return self._export(self.state["params"], observations, _array(available_actions, self.device, jnp.bool_))

    def act(self, obs, w, target_network=False, available_actions=None, action_mask=None):
        """Masked greedy actions; absent a mask, all actions are available.

        available_actions is a shared list of action IDs. action_mask accepts
        boolean (n_actions,) or (batch, n_actions) masks. Supply only one.
        """
        if available_actions is not None and action_mask is not None:
            raise ValueError("Supply available_actions or action_mask, not both")
        obs = jnp.atleast_2d(_array(obs, self.device))
        if available_actions is not None:
            ids = np.asarray(available_actions)
            if ids.size == 0 or ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= self.n_actions)):
                raise ValueError("available_actions must contain valid action IDs")
            action_mask = np.zeros(self.n_actions, dtype=bool)
            action_mask[ids] = True
        if action_mask is None:
            action_mask = np.ones(self.n_actions, dtype=bool)
        mask = np.asarray(action_mask, dtype=bool)
        if mask.shape not in ((self.n_actions,), (obs.shape[0], self.n_actions)) or not np.all(mask.any(axis=-1)):
            raise ValueError("Each observation needs at least one available action in a correctly shaped mask")
        params = self.state["target" if target_network else "params"]["f"]
        return self._act(params, obs, _array(w, self.device), _array(mask, self.device, jnp.bool_))

    def act_e_greedy(self, obs, g, available_actions=None, epsilon=0.2, target_network=False):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must lie in [0, 1]")
        if available_actions is not None:
            ids = np.asarray(available_actions)
            if ids.size == 0 or ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= self.n_actions)):
                raise ValueError("available_actions must contain valid action IDs")
        if self._python_rng.random() < epsilon:
            return self._python_rng.choice(range(self.n_actions) if available_actions is None else available_actions)
        return int(self.act(obs, g, target_network=target_network, available_actions=available_actions).item())

    def save(self, path):
        """Save parameters, targets, Adam moments, config and RNG (not replay).

        As with the original checkpoint loader, load only trusted files.
        """
        feature = None
        if isinstance(self.featuriser, RadialBasisFunction2D):
            feature = dict(x_bounds=self.featuriser.x_bounds, y_bounds=self.featuriser.y_bounds,
                           dim=self.featuriser.dim, sigma=self.featuriser.sigma)
        elif not isinstance(self.featuriser, IdentityFeaturiser):
            raise TypeError("Checkpointing supports IdentityFeaturiser and RadialBasisFunction2D")
        checkpoint = dict(format="forward_backward_jax_v1", config=self.config, featuriser=feature,
                          state=jax.device_get(self.state), python_rng=self._python_rng.getstate())
        with Path(path).open("wb") as stream:
            pickle.dump(checkpoint, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_checkpoint(cls, path, device="cpu", buffer_device="cpu"):
        with Path(path).open("rb") as stream:
            checkpoint = pickle.load(stream)
        if checkpoint["format"] != "forward_backward_jax_v1":
            raise ValueError("Expected a forward_backward_jax checkpoint")
        feature = checkpoint["featuriser"]
        learner = cls(**checkpoint["config"], device=device, buffer_device=buffer_device,
                      featuriser=None if feature is None else RadialBasisFunction2D(**feature, device=device))
        learner.state = jax.device_put(checkpoint["state"], learner.device)
        learner._python_rng.setstate(checkpoint["python_rng"])
        learner._sync_networks()
        return learner
