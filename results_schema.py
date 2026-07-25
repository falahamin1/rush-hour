"""
Unified result-file schema shared across the three benchmarks in this paper
(Tangram, Rush Hour, Navigation) so a single downstream statistics module can
consume all three without per-benchmark parsing. This file is intentionally
self-contained (no imports from the rest of this repo) since the same source
is pasted into each benchmark's own codebase.

Schema:
{
  "benchmark": "rush_hour", "tier": "12_moves", "encoder": "hrep", "seed": 3,
  "config": {...every hyperparameter used, plus git_commit + timestamp...},
  "curve": [{"iter": 100, "entropy": 1.82, "solve_greedy": 0.42, "solve_stochastic": 0.51}, ...],
  "final": {
    "per_instance": [{"instance_id": "d12_p3", "solved_greedy": true,
                       "solved_stochastic": true, "steps": 14, "optimal_steps": 12}, ...],
    "solve_rate_greedy": 0.60, "solve_rate_stochastic": 0.67
  }
}
"""
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone


def git_commit_hash(repo_dir=None):
    """Best-effort commit hash of the checked-out code, or None outside a git repo."""
    repo_dir = repo_dir or os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def build_result(*, benchmark, tier, encoder, seed, config, curve, per_instance):
    """Assemble one result dict matching the shared cross-benchmark schema.

    Aggregate solve rates are the mean of solved_greedy / solved_stochastic
    across `per_instance`. `config` is copied and augmented with git_commit
    and timestamp if not already present (callers may pre-set timestamp to
    the training start time instead of write time).
    """
    n = len(per_instance)
    solve_rate_greedy = sum(1 for r in per_instance if r["solved_greedy"]) / n if n else 0.0
    solve_rate_stochastic = sum(1 for r in per_instance if r["solved_stochastic"]) / n if n else 0.0
    full_config = dict(config)
    full_config.setdefault("git_commit", git_commit_hash())
    full_config.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    return {
        "benchmark": benchmark,
        "tier": tier,
        "encoder": encoder,
        "seed": seed,
        "config": full_config,
        "curve": curve,
        "final": {
            "per_instance": per_instance,
            "solve_rate_greedy": solve_rate_greedy,
            "solve_rate_stochastic": solve_rate_stochastic,
        },
    }


def write_result(result, output_dir):
    """Atomically write `result` to {output_dir}/{benchmark}_{tier}_{encoder}_seed{N}.json.

    Uses a temp-file + os.replace (same convention as checkpoint_utils.py's
    save_checkpoint) so a job killed mid-write never leaves a truncated file
    behind, and so this can safely be called repeatedly at every training
    checkpoint -- each call fully overwrites the previous file, so partial
    results from the latest completed checkpoint survive a crash.
    """
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{result['benchmark']}_{result['tier']}_{result['encoder']}_seed{result['seed']}.json"
    path = os.path.join(output_dir, fname)
    fd, tmp_path = tempfile.mkstemp(dir=output_dir, prefix=fname + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path
