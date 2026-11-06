"""Touati & Ollivier Continuous Maze, including released geometry edge cases.

Reference: ahmed-touati/controllable_agent, cea625688c7e9680bb98d82339d702758d812b3e.
Local RandomState replaces the reference's global numpy RNG; draws are identical
when initialized with the same state. Noise is cast to float32 before addition.
"""
import numpy as np

ACTIONS = np.array([[-.1, 0], [.1, 0], [0, .1], [0, -.1], [0, 0]])
ACTION_NAMES = ('LEFT', 'RIGHT', 'UP', 'DOWN', 'STAY')
WALLS = np.array([[[.25, 0], [.25, .4]], [[.75, 1], [.75, .6]]])


def on_segment(a, b, c):
    if a[0] == b[0]:
        # Preserve the reference's ordered-y test, including reversed walls.
        return c[0] == b[0] and a[1] <= c[1] <= b[1]
    return ((c[1]-a[1]) == (b[1]-a[1])/(b[0]-a[0])*(c[0]-a[0])
            and min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))


def intersects(a, b, c, d):
    if (on_segment(a, b, c) or on_segment(a, b, d)
            or on_segment(c, d, a) or on_segment(c, d, b)):
        return True
    def ccw(p, q, r):
        return (r[1]-p[1])*(q[0]-p[0]) >= (q[1]-p[1])*(r[0]-p[0])
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def goal_reached(position, goal, threshold=.05):
    return bool(np.all(np.abs(np.asarray(position)-goal) <= threshold))


class ContinuousMaze:
    def __init__(self, seed=0, movement_noise=.01):
        self.rng = np.random.RandomState(seed)
        self.movement_noise = movement_noise

    @property
    def position(self):
        return np.array(self._position)

    @position.setter
    def position(self, value):
        self._position = list(value)

    def sample_point(self):
        while True:
            p = self.rng.uniform(0, 1, 2)
            if not any(on_segment(a, b, p) for a, b in WALLS):
                return p

    def reset(self):
        self.position, self.goal = self.sample_point(), self.sample_point()
        return self.position.copy()

    def step(self, action):
        if action not in range(5):
            raise ValueError('Action must be an integer in [0, 4]')
        delta = self.rng.normal(ACTIONS[action], self.movement_noise).astype(np.float32)
        # Preserve scalar types: boundary clipping in the original returns Python
        # scalars, whose subsequent addition to np.float32 rounds in float32 on
        # NumPy 2. An array-wide clip/add would subtly change later transitions.
        proposed = []
        for coordinate, displacement in zip(self._position, delta):
            value = coordinate + displacement
            proposed.append(1 if value >= 1 else 0.0 if value <= 0 else value)
        collision = any(intersects(self.position, proposed, a, b) for a, b in WALLS)
        if not collision:
            self.position = proposed
        return (self.position.copy(), float(goal_reached(self.position, self.goal)), False,
                dict(goes_through_wall=collision, proposed_position=np.array(proposed)))


def plot_maze(ax, states=None, trajectories=()):
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-')
    for wall in WALLS:
        ax.plot(*wall.T, 'k-', linewidth=3)
    if states is not None:
        ax.scatter(*np.asarray(states).T, s=3, alpha=.25, label='Sampled states')
    for trajectory in trajectories:
        path = np.asarray(trajectory['path'])
        line, = ax.plot(*path.T, alpha=.7)
        ax.scatter(*path[0], marker='o', color=line.get_color())
        ax.scatter(*trajectory['goal'], marker='*', color=line.get_color())
    ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel='x', ylabel='y', aspect='equal')
    return ax
