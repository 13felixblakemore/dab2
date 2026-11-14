"""Four Rooms with centered, uniformly scaled doorways; base environment unchanged."""
import math
import numpy as np
from fb_navigation.environments.continuous_four_rooms_jax import ContinuousFourRooms, WallBlock


class AdjustableFourRooms(ContinuousFourRooms):
    def __init__(self, doorway_scale=1.0, threshold_distance=0.1, **kwargs):
        if not math.isfinite(doorway_scale) or not 0 < doorway_scale < 3:
            raise ValueError('doorway_scale must be finite and strictly between 0 and 3')
        if not math.isfinite(threshold_distance) or threshold_distance <= 0:
            raise ValueError('threshold_distance must be finite and positive')
        super().__init__(**kwargs)
        self.threshold_distance = float(threshold_distance)
        self.doorway_scale = float(doorway_scale)
        half = self.doorway_scale/2
        bottom = (1.5-half, 1.5+half)
        top = (8.5-half, 8.5+half)
        left = (2.5-half, 2.5+half)
        right = (9.5-half, 9.5+half)
        blocks = [(6, 0, 7, bottom[0]), (6, bottom[1], 7, top[0]), (6, top[1], 7, 11),
                  (0, 4, left[0], 5), (left[1], 4, 6, 5),
                  (7, 4, right[0], 5), (right[1], 4, 11, 5)]
        scale = self.size/11
        self._wall_blocks = [WallBlock(*(value*scale for value in b)) for b in blocks]


def fixed_pairs(env, per_category, seed):
    """Keep baseline endpoints identical across widths; recompute geometry-specific BFS."""
    # Evaluation helpers select a headless plotting backend; keep that side
    # effect out of imports used only to construct an interactive environment.
    from fb_navigation.diagnostics.control_bfs_bottlenecks import build_graph, bfs_distances
    from fb_navigation.evaluation.fourrooms import make_pairs

    baseline = ContinuousFourRooms(size=env.size, include_stay_action=False)
    pairs = make_pairs(baseline, per_category, seed)
    nodes, graph = build_graph(env, 20)
    starts = np.argmin(np.sum((pairs['starts'][:, None]-nodes[None])**2, axis=-1), axis=1)
    goals = np.argmin(np.sum((pairs['goals'][:, None]-nodes[None])**2, axis=-1), axis=1)
    if (np.max(abs(nodes[starts]-pairs['starts'])) > 1e-6
            or np.max(abs(nodes[goals]-pairs['goals'])) > 1e-6):
        raise ValueError('Baseline endpoints unavailable in this geometry')
    cache, lengths = {}, []
    for start, goal in zip(starts, goals):
        if goal not in cache:
            cache[goal] = bfs_distances(graph, np.max(abs(nodes-nodes[goal]), axis=1) <= env.threshold_distance)
        lengths.append(cache[goal][start])
    if min(lengths) < 1:
        raise ValueError('Disconnected or initially successful reference pair')
    pairs['bfs_lengths'] = np.array(lengths)
    return pairs
