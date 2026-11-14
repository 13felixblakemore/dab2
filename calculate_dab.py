
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dab import visualise_successor_measure, visualise_dab
from fb_navigation.agents.forward_backward_jax import FBLearner
from fb_navigation.environments.adjustable_four_rooms import AdjustableFourRooms
from fb_navigation.environments.continuous_four_rooms_jax import ContinuousFourRooms
from fb_navigation.environments.continuous_four_rooms import visualize_environment


config= {
    "seed": 0,
    "path": "experiments/fourrooms_doorways/runs/width0.8_replay_stopped_new/epoch_0800.pkl",
}

# Instantiate environment and FB
env = AdjustableFourRooms(
    doorway_scale=0.8,
    include_stay_action=False,
    seed=int(config["seed"]+4),
)
learner = FBLearner.from_checkpoint(config["path"])

# Create 50x50 coordinate grid
x = jnp.linspace(0, 1, 50)
y = jnp.linspace(0, 1, 50)
X, Y = jnp.meshgrid(x, y)

# Identify wall locations
wall_mask = jnp.zeros(X.shape, dtype=bool)

for block in env.wall_blocks:
    wall_mask |= (
        (X >= block.xmin) & (X <= block.xmax)
        & (Y >= block.ymin) & (Y <= block.ymax)
    )

# Flatten grid and retain only valid states
all_states = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
valid_mask = ~wall_mask.ravel()
states = all_states[valid_mask]

# embed states
embedded_states = learner.embed_B(states)

# sample n (u,v) pairs

u = env.sample_points(1000)
v = env.sample_points(1000)

z = learner.embed_B(v)
f_num = learner.embed_F(u, z) # 1000,50,4
f_denom = jax.vmap(
    lambda task: learner.embed_F(
        states,
        jnp.broadcast_to(task, (len(states), task.shape[0]))
    )
)(z)

q = jnp.einsum("nda,nd->na", f_num, z)  # (1000, 4)
q_denom = jnp.einsum("nsda,nd->nsa", f_denom, z) # (1000, s, 4)

a = jnp.argmax(q, axis=-1)
a_denom = jnp.argmax(q_denom, axis=-1)

f_pi = jnp.take_along_axis(
    f_num, a[:, None, None], axis=-1
).squeeze(-1)                              # (1000, 50)
f_pi_denom = jnp.take_along_axis(
    f_denom, a_denom[:, :, None, None], axis=-1
).squeeze(-1)                                               # (1000, s, 50)

# 3. Calculate successor densities at every destination state
densities_num = f_pi @ embedded_states.T    # (1000, 2500)
densities_denom = jnp.einsum(
    "nsd,sd->ns", f_pi_denom, embedded_states
)

# Both arrays have shape (1000, 2205)
dab = densities_num / densities_denom

# Diagnostic filtering
valid = (
    jnp.isfinite(dab)
    & (densities_num >= 0)
    & (densities_denom > 50)
    & (dab <= 2)
)

dab_filtered = jnp.where(valid, dab, jnp.nan)
mean_num = jnp.mean(densities_num, axis=0)
median_dab = jnp.nanmedian(dab_filtered, axis=0)

mean_dab = jnp.nanmean(dab_filtered, axis=0)

print("Fraction of valid ratios:", jnp.mean(valid))
print("Mean valid pairs per state:", jnp.mean(jnp.sum(valid, axis=0)))

full_dab = np.full(X.size, np.nan)
full_dab[np.asarray(valid_mask)] = np.asarray(mean_dab)


print("Negative numerator:", jnp.mean(densities_num < 0))
print("Negative denominator:", jnp.mean(densities_denom < 0))

print("Denominator percentiles:",
      jnp.percentile(densities_denom, jnp.array([0, 1, 10, 50, 90, 99, 100])))

print("Ratio percentiles:",
      jnp.percentile(dab, jnp.array([0, 1, 10, 50, 90, 99, 100])))

print("Nonfinite ratios:", jnp.mean(~jnp.isfinite(dab)))


# Plot using your existing function
visualise_dab(
    full_dab.reshape(X.shape),
    X, Y, env,
    title="Mean DAB over sampled pairs"
)

plt.show()


pair_idx = 2  # Choose any start-goal pair

density_grid = np.full(X.size, np.nan)
density_grid[np.asarray(valid_mask)] = np.asarray(densities_num[pair_idx])

visualise_successor_measure(
    density_grid.reshape(X.shape),
    X, Y, env,
    start=u[pair_idx],
    goal=v[pair_idx]
)

plt.show()
