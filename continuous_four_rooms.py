import dataclasses
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


@dataclasses.dataclass(order=True, frozen=True)
class Point:
    x: float
    y: float

    def sum(self):
        return self.x + self.y

    def l2norm(self):
        return np.sqrt(self.x * self.x + self.y * self.y)

    def __add__(self, other: "Point"):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point"):
        return Point(self.x - other.x, self.y - other.y)

    def normal_sample_around(self, scale: float):
        new_coords = np.random.normal(dataclasses.astuple(self), scale).astype(
            np.float32
        )
        return Point(*new_coords)

    def is_close_to(self, other: "Point", diff: float = 1e-4):
        point_diff = self - other
        return abs(point_diff.x) <= diff and abs(point_diff.y) <= diff


@dataclasses.dataclass(frozen=True)
class WallBlock:
    """An axis-aligned rectangular thick wall [xmin, xmax] x [ymin, ymax]."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains(self, p: Point, margin: float = 0.0) -> bool:
        """Checks if a point p lies inside the block."""
        return (self.xmin - margin) <= p.x <= (self.xmax + margin) and (
            self.ymin - margin
        ) <= p.y <= (self.ymax + margin)

    def intersects_segment(self, p1: Point, p2: Point) -> bool:
        """Checks if the trajectory segment p1 -> p2 penetrates or enters the block (Slab method)."""
        if self.contains(p2, margin=-1e-6):
            return True

        dx = p2.x - p1.x
        dy = p2.y - p1.y

        t_min = 0.0
        t_max = 1.0

        # X-slab
        if abs(dx) < 1e-9:
            if p1.x < self.xmin or p1.x > self.xmax:
                return False
        else:
            t1 = (self.xmin - p1.x) / dx
            t2 = (self.xmax - p1.x) / dx
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
            if t_min > t_max:
                return False

        # Y-slab
        if abs(dy) < 1e-9:
            if p1.y < self.ymin or p1.y > self.ymax:
                return False
        else:
            t1 = (self.ymin - p1.y) / dy
            t2 = (self.ymax - p1.y) / dy
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
            if t_min > t_max:
                return False

        # Collision occurs if there is a positive penetration into the wall
        return t_max > 1e-6 and (t_max - t_min) > 1e-6


def get_classic_four_rooms_blocks(size: float = 11.0) -> list[WallBlock]:
    """Returns the 7 wall blocks forming the classic 11x11 Four Rooms layout.

    Layout breakdown (11x11 grid):
      - Rooms: Top-Left (6x6), Top-Right (4x6), Bottom-Left (6x4), Bottom-Right (4x4)
      - Vertical wall: x in [6, 7] with doors at y in [1, 2] and [8, 9]
      - Horizontal wall: y in [4, 5] with doors at x in [2, 3] and [9, 10]
    """
    scale = size / 11.0

    raw_blocks = [
        # --- Vertical Wall Segments (x in [6, 7]) ---
        (6, 0, 7, 1),  # Below bottom doorway
        (6, 2, 7, 8),  # Middle pillar connecting intersection, between doorways
        (6, 9, 7, 11),  # Above top doorway
        # --- Horizontal Wall Segments (y in [4, 5]) ---
        (0, 4, 2, 5),  # Left outer segment
        (3, 4, 6, 5),  # Left inner segment
        (7, 4, 9, 5),  # Right inner segment
        (10, 4, 11, 5),  # Right outer segment
    ]

    return [
        WallBlock(
            xmin=b[0] * scale,
            ymin=b[1] * scale,
            xmax=b[2] * scale,
            ymax=b[3] * scale,
        )
        for b in raw_blocks
    ]


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

        self._rng = np.random.RandomState(seed)
        random.seed(seed)

        self._current_position = Point(0.5, 0.5)
        self._goal: Point | None = (
            None if self.explorable else Point(size - 0.5, size - 0.5)
        )

    def _past_edge(self, x: float) -> tuple[bool, float]:
        if x >= self._size:
            return True, self._size
        elif x <= 0.0:
            return True, 0.0
        return False, x

    def _wrap_coordinate(self, point: Point) -> Point:
        return Point(self._past_edge(point.x)[1], self._past_edge(point.y)[1])

    def _check_on_wall(self, p: Point) -> bool:
        return any(block.contains(p) for block in self._wall_blocks)

    def _check_goes_through_wall(self, start: Point, end: Point) -> bool:
        return any(block.intersects_segment(start, end) for block in self._wall_blocks)

    def sample_point(self) -> Point:
        while True:
            p = Point(*np.random.uniform(1e-6, self._size - 1e-6, 2))
            if not self._check_on_wall(p):
                return p

    def reset(self, state: Point | None = None, goal: Point | None = None):
        self._current_position = state if state is not None else self.sample_point()
        if self.explorable:
            self._goal = None
        else:
            self._goal = goal if goal is not None else self.sample_point()
        return self.current_position

    def step(self, id_action: int):
        perturbed_action = self.actions[id_action].normal_sample_around(self._noise)
        proposed_position = self._wrap_coordinate(
            self._current_position + perturbed_action
        )
        goes_through_wall = self._check_goes_through_wall(
            self._current_position, proposed_position
        )

        if not goes_through_wall:
            self._current_position = proposed_position

        if self.explorable or self._goal is None:
            reward = 0.0
            done = False
        else:
            reward = (
                1.0
                if self._current_position.is_close_to(
                    self._goal, diff=self.threshold_distance
                )
                else 0.0
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
    def current_position(self) -> np.ndarray:
        return np.array(dataclasses.astuple(self._current_position))

    @property
    def goal(self) -> np.ndarray | None:
        if self._goal is None:
            return None
        return np.array(dataclasses.astuple(self._goal))

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


def visualize_environment(
    world: ContinuousWorld,
    ax: plt.Axes,
    draw_grid: bool = True,
    draw_initial_position: bool = True,
    draw_goal: bool = True,
):
    size = world.size

    # 1. Draw light cell grid lines
    if draw_grid:
        n_cells = int(round(size))
        for i in range(n_cells + 1):
            ax.axhline(i, color="#dcdcdc", linewidth=1, zorder=1)
            ax.axvline(i, color="#dcdcdc", linewidth=1, zorder=1)

    # 2. Draw solid black rectangular walls
    for block in world.wall_blocks:
        rect = patches.Rectangle(
            (block.xmin, block.ymin),
            block.xmax - block.xmin,
            block.ymax - block.ymin,
            facecolor="black",
            edgecolor="black",
            zorder=2,
        )
        ax.add_patch(rect)

    # 3. Outer boundary border
    border = patches.Rectangle(
        (0, 0),
        size,
        size,
        fill=False,
        edgecolor="black",
        linewidth=10,
        zorder=3,
    )
    ax.add_patch(border)

    # 4. Agent and goal markers
    carrot = (235 / 255, 137 / 255, 33 / 255, 0.9)
    marker_style = dict(color=carrot, markersize=12)

    if draw_initial_position:
        x, y = world.current_position
        ax.plot(x, y, marker="o", **marker_style, zorder=4)

    if draw_goal and world.goal is not None:
        x, y = world.goal
        ax.plot(x, y, marker="*", **marker_style, zorder=4)

    ax.set_xlim(-0.3, size + 0.3)
    ax.set_ylim(-0.3, size + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax
