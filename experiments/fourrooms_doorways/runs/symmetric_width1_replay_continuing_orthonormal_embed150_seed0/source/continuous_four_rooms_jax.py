"""JAX port of continuous_four_rooms with the same stateful environment API.

Movement and rejection sampling use compiled, pure JAX kernels. The environment
wrapper owns its PRNG key; use a new instance with the same seed to replay a run.
The wrapper itself is stateful and should not be passed to jit or vmap.
"""

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp

from fb_navigation.environments.continuous_four_rooms import (
    get_classic_four_rooms_blocks as _classic_blocks,
    visualize_environment,
)


@partial(jax.tree_util.register_dataclass, data_fields=["x", "y"], meta_fields=[])
@dataclasses.dataclass(order=True, frozen=True)
class Point:
    x: float
    y: float

    def sum(self):
        return self.x + self.y

    def l2norm(self):
        return jnp.sqrt(self.x * self.x + self.y * self.y)

    def __add__(self, other: "Point"):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point"):
        return Point(self.x - other.x, self.y - other.y)

    def normal_sample_around(self, scale: float, key: jax.Array):
        """Sample using an explicit JAX key (a fresh key is needed per call)."""
        new_coords = (
            jnp.asarray((self.x, self.y))
            + scale * jax.random.normal(key, (2,))
        ).astype(jnp.float32)
        return Point(*new_coords)

    def is_close_to(self, other: "Point", diff: float = 1e-4):
        point_diff = self - other
        return (jnp.abs(point_diff.x) <= diff) & (jnp.abs(point_diff.y) <= diff)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["xmin", "ymin", "xmax", "ymax"],
    meta_fields=[],
)
@dataclasses.dataclass(frozen=True)
class WallBlock:
    """An axis-aligned rectangular thick wall [xmin, xmax] x [ymin, ymax]."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains(self, p: Point, margin: float = 0.0) -> bool:
        """Checks if a point p lies inside the block."""
        return (
            (self.xmin - margin <= p.x)
            & (p.x <= self.xmax + margin)
            & (self.ymin - margin <= p.y)
            & (p.y <= self.ymax + margin)
        )

    def intersects_segment(self, p1: Point, p2: Point) -> bool:
        """The original slab test, expressed without Python data-dependent branches."""
        t_min = jnp.asarray(0.0)
        t_max = jnp.asarray(1.0)
        valid = jnp.asarray(True)
        for start, end, lower, upper in (
            (p1.x, p2.x, self.xmin, self.xmax),
            (p1.y, p2.y, self.ymin, self.ymax),
        ):
            delta = end - start
            parallel = jnp.abs(delta) < 1e-9
            safe_delta = jnp.where(parallel, 1.0, delta)
            t1 = (lower - start) / safe_delta
            t2 = (upper - start) / safe_delta
            t_min = jnp.where(parallel, t_min, jnp.maximum(t_min, jnp.minimum(t1, t2)))
            t_max = jnp.where(parallel, t_max, jnp.minimum(t_max, jnp.maximum(t1, t2)))
            valid = valid & (~parallel | ((start >= lower) & (start <= upper)))
        return self.contains(p2, margin=-1e-6) | (
            valid & (t_max > 1e-6) & ((t_max - t_min) > 1e-6)
        )


def get_classic_four_rooms_blocks(size: float = 11.0) -> list[WallBlock]:
    """Use the common layout while retaining JAX wall dataclasses."""
    return [WallBlock(block.xmin, block.ymin, block.xmax, block.ymax)
            for block in _classic_blocks(size)]


@jax.jit
def _move(position, action, key, noise, size, wall_blocks):
    """Pure movement kernel; wall rejection and boundary clipping match the original."""
    perturbed_action = action.normal_sample_around(noise, key)
    proposed = position + perturbed_action
    proposed = Point(jnp.clip(proposed.x, 0.0, size), jnp.clip(proposed.y, 0.0, size))
    collision = jnp.any(jnp.asarray([
        block.intersects_segment(position, proposed) for block in wall_blocks
    ], dtype=bool))
    result = Point(
        jnp.where(collision, position.x, proposed.x),
        jnp.where(collision, position.y, proposed.y),
    )
    return result, proposed, collision


@jax.jit
def _sample_point(key, size, wall_blocks):
    def draw(draw_key):
        coords = jax.random.uniform(draw_key, (2,), minval=1e-6, maxval=size - 1e-6)
        return Point(*coords)

    def on_wall(state):
        _, point = state
        return jnp.any(jnp.asarray([
            block.contains(point) for block in wall_blocks
        ], dtype=bool))

    def resample(state):
        next_key, _ = state
        next_key, draw_key = jax.random.split(next_key)
        return next_key, draw(draw_key)

    key, draw_key = jax.random.split(key)
    return jax.lax.while_loop(on_wall, resample, (key, draw(draw_key)))[1]


class ContinuousWorld:
    """The ContinuousWorld Environment with support for thick rectangular walls."""

    def __init__(
        self,
        size: float = 11.0,
        wall_blocks: list[WallBlock] | None = None,
        step_size: float = 1.0,
        movement_noise: float = 0.05,
        seed: int = 1,
        threshold_distance: float = 0.5,
        explorable: bool = False,
        include_stay_action: bool = True,
    ):
        self._size = size
        self._wall_blocks = wall_blocks or []
        self._noise = movement_noise
        self.threshold_distance = threshold_distance
        self.explorable = explorable
        self.include_stay_action = include_stay_action

        self.actions = [
            Point(-step_size, 0.0),  # LEFT
            Point(step_size, 0.0),  # RIGHT
            Point(0.0, step_size),  # UP
            Point(0.0, -step_size),  # DOWN
        ]
        self.actions_str = ["LEFT", "RIGHT", "UP", "DOWN"]

        if self.include_stay_action:
            self.actions.append(Point(0.0, 0.0))
            self.actions_str.append("STAY")

        self._rng = jax.random.PRNGKey(seed)

        self._current_position = Point(0.5, 0.5)
        self._goal: Point | None = (
            None if self.explorable else Point(size - 0.5, size - 0.5)
        )

    def _past_edge(self, x: float) -> tuple[bool, float]:
        return (x >= self._size) | (x <= 0.0), jnp.clip(x, 0.0, self._size)

    def _wrap_coordinate(self, point: Point) -> Point:
        return Point(self._past_edge(point.x)[1], self._past_edge(point.y)[1])

    def _check_on_wall(self, p: Point) -> bool:
        return jnp.any(jnp.asarray([block.contains(p) for block in self._wall_blocks], dtype=bool))

    def _check_goes_through_wall(self, start: Point, end: Point) -> bool:
        return jnp.any(jnp.asarray([
            block.intersects_segment(start, end) for block in self._wall_blocks
        ], dtype=bool))

    def sample_point(self) -> Point:
        self._rng, key = jax.random.split(self._rng)
        return _sample_point(key, self._size, tuple(self._wall_blocks))

    def sample_points(self, n: int) -> jax.Array:
        self._rng, key = jax.random.split(self._rng)
        keys = jax.random.split(key, n)

        points = jax.vmap(
            _sample_point,
            in_axes=(0, None, None)
        )(keys, self._size, tuple(self._wall_blocks))

        return jnp.stack([points.x, points.y], axis=-1)

    def reset(self, state: Point | None = None, goal: Point | None = None):
        self._current_position = state if state is not None else self.sample_point()
        if self.explorable:
            self._goal = None
        else:
            self._goal = goal if goal is not None else self.sample_point()
        return self.current_position

    def step(self, id_action: int):
        self._rng, key = jax.random.split(self._rng)
        self._current_position, proposed_position, goes_through_wall = _move(
            self._current_position, self.actions[id_action], key,
            self._noise, self._size, tuple(self._wall_blocks),
        )

        if self.explorable or self._goal is None:
            reward = 0.0
            done = False
        else:
            reward = jnp.where(
                self._current_position.is_close_to(
                    self._goal, diff=self.threshold_distance
                ),
                1.0, 0.0,
            )
            done = reward > 0.0

        return (
            self.current_position,
            reward,
            done,
            {
                "goes_through_wall": goes_through_wall,
                "proposed_position": proposed_position,
            },
        )

    @property
    def current_position(self) -> jax.Array:
        return jnp.asarray((self._current_position.x, self._current_position.y))

    @property
    def goal(self) -> jax.Array | None:
        if self._goal is None:
            return None
        return jnp.asarray((self._goal.x, self._goal.y))

    @property
    def size(self):
        return self._size

    @property
    def wall_blocks(self):
        return self._wall_blocks

    def get_action_space(self):
        return list(range(len(self.actions)))

    def get_available_actions(self, state):
        return list(range(len(self.actions)))


class ContinuousFourRooms(ContinuousWorld):
    def __init__(
        self,
        size: float = 1.0,
        step_size: float = 0.05,
        movement_noise: float = 0.005,
        seed: int = 1,
        explorable: bool = False,
        include_stay_action: bool = True,
    ):
        super().__init__(
            size=size,
            wall_blocks=get_classic_four_rooms_blocks(size=size),
            step_size=step_size,
            movement_noise=movement_noise,
            seed=seed,
            threshold_distance=0.1,
            explorable=explorable,
            include_stay_action=include_stay_action,
        )
