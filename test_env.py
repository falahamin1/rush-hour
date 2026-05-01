"""
Test script for RushHourGym.

Loads rush.txt, picks a puzzle with < 15 moves, builds the gym environment,
and runs a sequence of checks:
  1. Parsing and rendering the initial board
  2. Observation shapes
  3. Action mask
  4. Manual step-through of random valid moves
  5. Reward and done signals
"""

import random
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from rush_hour_env import RushHourGym, parse_board, load_puzzles, GRID, MAX_VEHICLES


PUZZLE_FILE = os.path.join(os.path.dirname(__file__), 'rush.txt')
MAX_MOVES_THRESHOLD = 15
RANDOM_SEED = 42


# ── helpers ────────────────────────────────────────────────────────────────

def load_easy_puzzles(path, max_moves=MAX_MOVES_THRESHOLD, limit=500):
    """Return up to `limit` puzzles that require <= max_moves moves."""
    puzzles = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            if int(parts[0]) <= max_moves:
                puzzles.append((int(parts[0]), parts[1]))
            if len(puzzles) >= limit:
                break
    return puzzles


def board_to_grid(board_str):
    """Pretty-print a 36-char board string as a 6x6 grid."""
    rows = [board_str[r * 6:(r + 1) * 6] for r in range(6)]
    return '\n'.join('  ' + ' '.join(row) for row in rows)


# ── test functions ─────────────────────────────────────────────────────────

def test_parsing(board_str, moves_required):
    print('=' * 60)
    print(f'TEST 1 — Board parsing  (requires {moves_required} moves)')
    print('=' * 60)
    print('Board string:', board_str)
    print('Grid layout:')
    print(board_to_grid(board_str))
    print()

    vehicles = parse_board(board_str)
    print(f'Parsed {len(vehicles)} vehicles:')
    for v in vehicles:
        print(f'  {v}')

    a = vehicles[0]
    assert a.vid == 'A', 'First vehicle must be A (red car)'
    assert a.horizontal, 'Red car must be horizontal'
    assert a.row == 2, f'Red car must be on row 2 (EXIT_ROW), got row {a.row}'
    print(f'\nRed car (A): row={a.row}, col={a.col}, size={a.size}  ✓')
    print()


def test_observations(env):
    print('=' * 60)
    print('TEST 2 — Observation shapes')
    print('=' * 60)
    obs, info = env.reset()

    h = obs['h_rep']
    v = obs['v_rep']

    print(f"h_rep shape : {h.shape}  expected ({MAX_VEHICLES}, 4, 3)")
    print(f"v_rep shape : {v.shape}  expected ({MAX_VEHICLES}, 4, 2)")

    assert h.shape == (MAX_VEHICLES, 4, 3), f'h_rep shape mismatch: {h.shape}'
    assert v.shape == (MAX_VEHICLES, 4, 2), f'v_rep shape mismatch: {v.shape}'

    print(f'\nFirst vehicle h_rep (red car constraints):')
    for j, row in enumerate(h[0]):
        print(f'  constraint {j}: {row}')

    print(f'\nFirst vehicle v_rep (red car corners):')
    for j, corner in enumerate(v[0]):
        if any(corner != 0):
            print(f'  corner {j}: {corner} → col={corner[0]*GRID:.1f}, row={corner[1]*GRID:.1f}')

    # Values should be in expected ranges
    assert np.all(np.isfinite(h)), 'h_rep contains NaN or Inf'
    assert np.all(np.isfinite(v)), 'v_rep contains NaN or Inf'
    print('\nAll observation values finite  ✓')
    print()


def test_action_mask(env):
    print('=' * 60)
    print('TEST 3 — Action mask')
    print('=' * 60)
    obs, _ = env.reset()
    mask = env.get_action_mask()

    print(f'Action space size : {env.action_space.n}  (MAX_VEHICLES * 2 = {MAX_VEHICLES * 2})')
    print(f'Valid actions     : {mask.sum()} / {len(mask)}')

    vehicles = env.inner.vehicles
    print('\nValid moves:')
    for action in np.where(mask)[0]:
        vi = action // 2
        direction = '+1' if action % 2 == 0 else '-1'
        vid = vehicles[vi].vid if vi < len(vehicles) else '?'
        orient = 'right/down' if vehicles[vi].horizontal else 'down/up' if not vehicles[vi].horizontal else '?'
        label = 'right' if (vehicles[vi].horizontal and action % 2 == 0) else \
                'left'  if (vehicles[vi].horizontal and action % 2 == 1) else \
                'down'  if (not vehicles[vi].horizontal and action % 2 == 0) else 'up'
        print(f'  action {action:3d}: vehicle {vid} ({label})')

    assert mask.sum() > 0, 'No valid actions — board must be stuck'
    print(f'\nAt least one valid action exists  ✓')
    print()


def test_step(env, n_steps=30):
    print('=' * 60)
    print(f'TEST 4 — Random rollout ({n_steps} steps max)')
    print('=' * 60)
    random.seed(RANDOM_SEED)
    obs, _ = env.reset()

    total_reward = 0.0
    invalid_count = 0

    for step in range(1, n_steps + 1):
        mask = env.get_action_mask()
        valid_actions = np.where(mask)[0]

        if len(valid_actions) == 0:
            print('  No valid actions left — stopping.')
            break

        action = int(random.choice(valid_actions))
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        vi = action // 2
        direction = '+1' if action % 2 == 0 else '-1'
        vid = env.inner.vehicles[vi].vid if vi < len(env.inner.vehicles) else '?'
        print(f'  step {step:3d}: action={action:3d} (vehicle {vid} {direction})'
              f'  reward={reward:+.2f}  cumulative={total_reward:+.2f}'
              f'  msg={info["msg"]}')

        if done:
            if info['solved']:
                print(f'\n  *** SOLVED in {step} steps! ***')
            else:
                print(f'\n  Max steps reached.')
            break

    print(f'\nTotal reward: {total_reward:.2f}')
    print(f'Invalid moves taken: {invalid_count}')
    print()


def test_rewards(env):
    print('=' * 60)
    print('TEST 5 — Reward and done signal sanity checks')
    print('=' * 60)

    obs, _ = env.reset()
    mask = env.get_action_mask()

    # Test that an invalid action (first masked-out one) returns -1 reward
    invalid_actions = np.where(~mask)[0]
    if len(invalid_actions) > 0:
        bad_action = int(invalid_actions[0])
        _, reward, done, _, info = env.step(bad_action)
        print(f'Invalid action reward: {reward}  (expected -1.0)')
        assert reward == -1.0, f'Expected -1.0 for invalid action, got {reward}'
        print('  ✓')
    else:
        print('  (all actions valid — skipping invalid-action test)')

    env.reset()
    print()


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print(f'Loading easy puzzles (≤{MAX_MOVES_THRESHOLD} moves) from {PUZZLE_FILE} ...')
    easy = load_easy_puzzles(PUZZLE_FILE, max_moves=MAX_MOVES_THRESHOLD, limit=500)
    print(f'Found {len(easy)} puzzles in first scan.\n')

    # Pick a puzzle in the 8–12 move range for a nice difficulty
    candidates = [(m, b) for m, b in easy if 8 <= m <= 12]
    if not candidates:
        candidates = easy
    random.seed(RANDOM_SEED)
    moves_required, board_str = random.choice(candidates)

    # ── run tests ───────────────────────────────────────────────────────

    test_parsing(board_str, moves_required)

    env = RushHourGym(board_str)

    test_observations(env)
    test_action_mask(env)
    test_step(env, n_steps=50)
    test_rewards(env)

    # ── final render ─────────────────────────────────────────────────────

    print('=' * 60)
    print('Rendering initial board state ...')
    print('=' * 60)
    env.reset()
    env.inner.render(title=f'Rush Hour — {moves_required}-move puzzle')

    print('\nAll tests passed.')


if __name__ == '__main__':
    main()
