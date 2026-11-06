"""Project paths and canonical source locations for reproducible snapshots."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = {'forward_backward.py': 'fb_navigation/agents/forward_backward.py',
 'forward_backward_jax.py': 'fb_navigation/agents/forward_backward_jax.py',
 'continuous_four_rooms.py': 'fb_navigation/environments/continuous_four_rooms.py',
 'continuous_four_rooms_jax.py': 'fb_navigation/environments/continuous_four_rooms_jax.py',
 'continuous_maze.py': 'fb_navigation/environments/continuous_maze.py',
 'adjustable_four_rooms.py': 'fb_navigation/environments/adjustable_four_rooms.py',
 'run_fb.py': 'fb_navigation/training/run_fb.py',
 'run_fb_jax.py': 'fb_navigation/training/run_fb_jax.py',
 'run_continuous_maze.py': 'fb_navigation/training/run_continuous_maze.py',
 'run_fourrooms_reference.py': 'fb_navigation/training/run_fourrooms_reference.py',
 'run_fourrooms_doorways.py': 'fb_navigation/training/run_fourrooms_doorways.py',
 'smoke_goal_terminated_fb_jax.py': 'fb_navigation/training/smoke_goal_terminated_fb_jax.py',
 'evaluate_width2_goals.py': 'fb_navigation/evaluation/evaluate_width2_goals.py',
 'evaluate_original_goals_jax.py': 'fb_navigation/evaluation/evaluate_original_goals_jax.py',
 'evaluate_torch_goals.py': 'fb_navigation/evaluation/evaluate_torch_goals.py',
 'evaluate_epsilon_bottlenecks.py': 'fb_navigation/evaluation/evaluate_epsilon_bottlenecks.py',
 'evaluate_fb_action_values.py': 'fb_navigation/evaluation/evaluate_fb_action_values.py',
 'fourrooms_reference_diagnostics.py': 'fb_navigation/evaluation/fourrooms_reference_diagnostics.py',
 'analyse_fb_rewards_failures.py': 'fb_navigation/diagnostics/analyse_fb_rewards_failures.py',
 'audit_doorway_goals.py': 'fb_navigation/diagnostics/audit_doorway_goals.py',
 'compare_bottleneck_policies.py': 'fb_navigation/diagnostics/compare_bottleneck_policies.py',
 'compare_navigation_stages.py': 'fb_navigation/diagnostics/compare_navigation_stages.py',
 'control_bfs_bottlenecks.py': 'fb_navigation/diagnostics/control_bfs_bottlenecks.py',
 'diagnose_continuous_maze.py': 'fb_navigation/diagnostics/diagnose_continuous_maze.py',
 'diagnose_fb_walls.py': 'fb_navigation/diagnostics/diagnose_fb_walls.py',
 'estimate_fb_bottlenecks_jax.py': 'fb_navigation/diagnostics/estimate_fb_bottlenecks_jax.py',
 'investigate_doorway_decline.py': 'fb_navigation/diagnostics/investigate_doorway_decline.py',
 'sweep_fb_temperature.py': 'fb_navigation/diagnostics/sweep_fb_temperature.py',
 'track_doorway_advantages.py': 'fb_navigation/diagnostics/track_doorway_advantages.py',
 'record_replay_failures_20261106.py': 'fb_navigation/diagnostics/record_replay_failures_20261106.py',
 'analyse_threshold025_results_20261106.py': 'fb_navigation/diagnostics/analyse_threshold025_results_20261106.py',
 'diagnose_implicit_reward_traps_20261106.py': 'fb_navigation/diagnostics/diagnose_implicit_reward_traps_20261106.py',
 'geometry_detour_implicit_reward_20261106.py': 'fb_navigation/diagnostics/geometry_detour_implicit_reward_20261106.py',
 'summarize_implicit_reward_traps_20261106.py': 'fb_navigation/diagnostics/summarize_implicit_reward_traps_20261106.py',
 'fb_width2_lr_polyak_sweep_20261106.py': 'fb_navigation/training/fb_width2_lr_polyak_sweep_20261106.py',
 'paths.py': 'fb_navigation/paths.py'}

def source_path(name):
    """Resolve historical source basenames to active implementations."""
    return PROJECT_ROOT / SOURCE_FILES.get(str(name), str(name))
