"""Safe goal reaching through a field of obstacles with control barrier functions.

A drone flies from a start to a goal behind a field of cylindrical obstacles, commanding attitude and
collective thrust. Control barrier functions keep it away from the obstacles in two ways: as a
safety filter that corrects the policy's commands before they reach the drone, and as reward terms
that teach the policy to respect the barrier conditions itself. `config.toml` describes a run, and
variants of it are overlay files in `configs/`. Runs in the `safe-reach` Pixi environment.
"""

from crazyflow_experiments.utils.paths import RESULTS_DIR, SAVES_DIR

MODEL_DIR = SAVES_DIR / "safe_reach"
RECORDINGS_DIR = RESULTS_DIR / "safe_reach"
