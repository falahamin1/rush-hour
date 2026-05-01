"""
Rush Hour gym environment.

Game logic uses a plain 6×6 occupancy grid (integer arithmetic).
PPL polytopes are built once per step, only when constructing neural-network
observations (h_rep / v_rep / adj).

Puzzle format (michaelfogleman.com/rush): 36-char string, 6x6 grid, row-major.
  'o' = empty  |  'x' = wall  |  'A' = red car (target)  |  'B'-'Z' = vehicles

Coordinates: x = column (0 = left), y = row (0 = top).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ppl import Variable, C_Polyhedron, Constraint_System
import gymnasium as gym
from gymnasium import spaces

GRID = 6
EXIT_ROW = 2       # Red car always exits through row 2 (0-indexed from top)
MAX_VEHICLES = 16  # Pad observations to this fixed size
MAX_CONSTRAINTS = 4  # Rectangles have exactly 4 half-space constraints
NUM_ACTIONS = MAX_VEHICLES * 2


# ── Vehicle ────────────────────────────────────────────────────────────────

class Vehicle:
    __slots__ = ('vid', 'row', 'col', 'size', 'horizontal')

    def __init__(self, vid, row, col, size, horizontal):
        self.vid = vid
        self.row = row
        self.col = col
        self.size = size
        self.horizontal = horizontal

    def __repr__(self):
        d = 'H' if self.horizontal else 'V'
        return f'Vehicle({self.vid} {d} row={self.row} col={self.col} size={self.size})'


# ── Board parser ───────────────────────────────────────────────────────────

def parse_board(s):
    """Return ordered list of Vehicles from a 36-char board string.

    Vehicles are ordered: A first (red car), then B-Z alphabetically.
    Orientation is inferred from repeated letters in the same row (horizontal)
    or same column (vertical).
    """
    if len(s) != 36:
        raise ValueError(f'Board string must be 36 chars, got {len(s)}')
    grid = [[s[r * 6 + c] for c in range(GRID)] for r in range(GRID)]
    seen, raw = set(), {}
    for r in range(GRID):
        for c in range(GRID):
            ch = grid[r][c]
            if ch in ('o', 'x') or ch in seen:
                continue
            seen.add(ch)
            if c + 1 < GRID and grid[r][c + 1] == ch:
                sz = 3 if c + 2 < GRID and grid[r][c + 2] == ch else 2
                raw[ch] = Vehicle(ch, r, c, sz, horizontal=True)
            elif r + 1 < GRID and grid[r + 1][c] == ch:
                sz = 3 if r + 2 < GRID and grid[r + 2][c] == ch else 2
                raw[ch] = Vehicle(ch, r, c, sz, horizontal=False)
    order = ['A'] + sorted(k for k in raw if k != 'A')
    return [raw[k] for k in order]


def load_puzzles(path, max_puzzles=None):
    """Load puzzles from a Fogleman-format file.

    Each line: <moves> <36-char-board> <cluster_size>
    Returns list of board strings.
    """
    puzzles = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                puzzles.append(parts[1])
            if max_puzzles and len(puzzles) >= max_puzzles:
                break
    return puzzles


# ── Raw environment ────────────────────────────────────────────────────────

class RushHourEnv:
    """Low-level Rush Hour environment backed by an integer occupancy grid.

    No PPL calls here — all collision detection is plain array arithmetic.
    PPL polytopes are built by RushHourGym only when assembling observations.
    """

    def __init__(self, board_str):
        self.board_str = board_str
        self.reset()

    # ── lifecycle ──────────────────────────────────────────────────────────

    def reset(self):
        self.vehicles = parse_board(self.board_str)
        self.grid = self._build_grid()

    def _build_grid(self):
        """6×6 int8 array; cell value = vehicle_index+1, or 0 if empty."""
        g = np.zeros((GRID, GRID), dtype=np.int8)
        for vi, v in enumerate(self.vehicles):
            if v.horizontal:
                g[v.row, v.col:v.col + v.size] = vi + 1
            else:
                g[v.row:v.row + v.size, v.col] = vi + 1
        return g

    # ── validity check (non-mutating) ──────────────────────────────────────

    def can_move(self, vi, delta):
        """Return True if vehicle vi can move delta steps without collision.

        Only the single leading-edge cell needs to be checked since delta is
        always +1 or -1.
        """
        v = self.vehicles[vi]
        if v.horizontal:
            edge = v.col + v.size if delta > 0 else v.col - 1
            if edge < 0 or edge >= GRID:
                return False
            return self.grid[v.row, edge] == 0
        else:
            edge = v.row + v.size if delta > 0 else v.row - 1
            if edge < 0 or edge >= GRID:
                return False
            return self.grid[edge, v.col] == 0

    # ── action ────────────────────────────────────────────────────────────

    def move(self, vi, delta):
        """Move vehicle vi by delta (+1 or -1). Returns (success, msg)."""
        if not self.can_move(vi, delta):
            return False, 'invalid'
        v = self.vehicles[vi]
        if v.horizontal:
            self.grid[v.row, v.col:v.col + v.size] = 0
            v.col += delta
            self.grid[v.row, v.col:v.col + v.size] = vi + 1
        else:
            self.grid[v.row:v.row + v.size, v.col] = 0
            v.row += delta
            self.grid[v.row:v.row + v.size, v.col] = vi + 1
        return True, 'ok'

    def is_solved(self):
        a = self.vehicles[0]  # A is always index 0
        return a.col + a.size >= GRID

    # ── render ────────────────────────────────────────────────────────────

    def render(self, title='Rush Hour', inline=True, save_path=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-0.1, GRID + 0.5)
        ax.set_ylim(-0.1, GRID + 0.1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

        for i in range(GRID + 1):
            ax.axhline(i, color='#bbb', lw=0.6)
            ax.axvline(i, color='#bbb', lw=0.6)

        # Exit arrow on the right side at EXIT_ROW
        ey = GRID - EXIT_ROW - 1  # flip: row 0 (top) → display y = GRID-1
        ax.annotate('', xy=(GRID + 0.45, ey + 0.5), xytext=(GRID, ey + 0.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

        palette = plt.cm.tab20.colors
        cmap, cidx = {}, 0

        for v in self.vehicles:
            if v.vid == 'A':
                fc = '#e74c3c'
            else:
                if v.vid not in cmap:
                    cmap[v.vid] = palette[cidx % len(palette)]
                    cidx += 1
                fc = cmap[v.vid]

            w = v.size if v.horizontal else 1
            h = 1 if v.horizontal else v.size
            dy = GRID - v.row - h  # flip y axis for display

            ax.add_patch(patches.FancyBboxPatch(
                (v.col + 0.07, dy + 0.07), w - 0.14, h - 0.14,
                boxstyle='round,pad=0.04',
                facecolor=fc, edgecolor='#333', lw=1.5, alpha=0.88,
            ))
            ax.text(v.col + w / 2, dy + h / 2, v.vid,
                    ha='center', va='center', fontsize=13,
                    fontweight='bold', color='white')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
        if inline:
            plt.show()
        else:
            plt.close()


# ── Gymnasium wrapper ──────────────────────────────────────────────────────

class RushHourGym(gym.Env):
    """
    Gymnasium wrapper for RushHourEnv.

    Action space : Discrete(MAX_VEHICLES * 2)
        action = vehicle_index * 2 + direction
        direction 0 → move +1 step  |  direction 1 → move -1 step

    Observations (both padded to MAX_VEHICLES):
        h_rep : (MAX_VEHICLES, MAX_CONSTRAINTS, 3)   half-space features
        v_rep : (MAX_VEHICLES, 4, 2)                 corner vertices
        adj   : (MAX_VEHICLES, MAX_CONSTRAINTS, MAX_CONSTRAINTS)

    PPL polytopes are built once per step inside _get_obs(); constraints and
    generators are cached and shared across all three observation builders.

    Reward (potential-based shaping):
        -1.0           invalid move (blocked or out of bounds)
        -0.01          step penalty for every valid move
        γ·f(s')−f(s)  potential shaping term; f(s) ∈ [0,1] blends
                       red-car progress (0.7) and path clearance (0.3)
        +10.0          bonus on solve
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, board_str, gamma=0.99):
        super().__init__()
        self.board_str = board_str
        self.inner = RushHourEnv(board_str)
        self.gamma = gamma
        self.max_steps = 200
        self.step_count = 0

        # PPL variables created once and reused across all observation calls
        self._px = Variable(0)
        self._py = Variable(1)

        self.action_space = spaces.Discrete(MAX_VEHICLES * 2)
        self.observation_space = spaces.Dict({
            'h_rep': spaces.Box(-1.0, 1.0,
                                shape=(MAX_VEHICLES, MAX_CONSTRAINTS, 3),
                                dtype=np.float32),
            'v_rep': spaces.Box(0.0, 1.0,
                                shape=(MAX_VEHICLES, 4, 2),
                                dtype=np.float32),
            'adj':   spaces.Box(0.0, 1.0,
                                shape=(MAX_VEHICLES, MAX_CONSTRAINTS, MAX_CONSTRAINTS),
                                dtype=np.float32),
        })

    # ── polytope construction (obs-time only) ──────────────────────────────

    def _make_poly(self, v):
        """Build a PPL C_Polyhedron for vehicle v from its current position."""
        px, py = self._px, self._py
        cs = Constraint_System()
        if v.horizontal:
            cs.insert(px >= v.col)
            cs.insert(px <= v.col + v.size)
            cs.insert(py >= v.row)
            cs.insert(py <= v.row + 1)
        else:
            cs.insert(px >= v.col)
            cs.insert(px <= v.col + 1)
            cs.insert(py >= v.row)
            cs.insert(py <= v.row + v.size)
        return C_Polyhedron(cs)

    # ── observation builders ───────────────────────────────────────────────

    def _get_obs(self):
        """Build the observation dict.

        Polytopes are constructed once; their constraints and generators are
        cached and passed to each sub-builder so PPL is not called redundantly.
        """
        px, py = self._px, self._py
        vehicles = self.inner.vehicles
        n = min(len(vehicles), MAX_VEHICLES)

        polys = [self._make_poly(vehicles[i]) for i in range(n)]
        constraints = [list(p.minimized_constraints()) for p in polys]
        generators  = [[g for g in p.generators() if g.is_point()] for p in polys]

        return {
            'h_rep': self._extract_h_rep(constraints, px, py),
            'v_rep': self._extract_v_rep(generators, px, py),
            'adj':   self._build_graph_adj(constraints, generators, px, py),
        }

    def _extract_h_rep(self, constraints, px, py):
        out = np.zeros((MAX_VEHICLES, MAX_CONSTRAINTS, 3), dtype=np.float32)
        for i, cs in enumerate(constraints):
            for j, c in enumerate(cs[:MAX_CONSTRAINTS]):
                a1 = -float(c.coefficient(px))
                a2 = -float(c.coefficient(py))
                b  =  float(c.inhomogeneous_term())
                norm = max(np.sqrt(a1 ** 2 + a2 ** 2), 1e-9)
                out[i, j] = [a1 / norm, a2 / norm, (b / norm) / GRID]
        return out

    def _extract_v_rep(self, generators, px, py):
        out = np.zeros((MAX_VEHICLES, 4, 2), dtype=np.float32)
        for i, verts in enumerate(generators):
            for j, g in enumerate(verts[:4]):
                out[i, j] = [float(g.coefficient(px)) / GRID,
                             float(g.coefficient(py)) / GRID]
        return out

    def _build_graph_adj(self, constraints, generators, px, py):
        """
        Per-vehicle 4×4 adjacency: edge between two constraints if they
        share a vertex (meet at an active boundary).  For axis-aligned
        rectangles this always produces the K_{2,2} bipartite structure
        between the two x-bounds and the two y-bounds.
        """
        epsilon = 1e-5
        all_adj = np.zeros((MAX_VEHICLES, MAX_CONSTRAINTS, MAX_CONSTRAINTS),
                           dtype=np.float32)
        for idx, (cs, verts) in enumerate(zip(constraints, generators)):
            nc = min(len(cs), MAX_CONSTRAINTS)
            for i in range(nc):
                for j in range(i + 1, nc):
                    for v in verts:
                        vi = self._eval_constraint(cs[i], v, px, py)
                        vj = self._eval_constraint(cs[j], v, px, py)
                        if abs(vi) < epsilon and abs(vj) < epsilon:
                            all_adj[idx, i, j] = all_adj[idx, j, i] = 1.0
                            break
        return all_adj

    @staticmethod
    def _eval_constraint(constraint, vertex, px, py):
        # PPL stores c as: coeff(px)*x + coeff(py)*y + inhomogeneous_term >= 0
        # A constraint is active at a vertex when this expression equals 0.
        d = vertex.divisor()
        x = float(vertex.coefficient(px)) / d
        y = float(vertex.coefficient(py)) / d
        return (float(constraint.coefficient(px)) * x
                + float(constraint.coefficient(py)) * y
                + float(constraint.inhomogeneous_term()))

    # ── action mask ───────────────────────────────────────────────────────

    def get_action_mask(self):
        """Boolean mask over the full action space (MAX_VEHICLES * 2)."""
        mask = np.zeros(MAX_VEHICLES * 2, dtype=bool)
        n = min(len(self.inner.vehicles), MAX_VEHICLES)
        for vi in range(n):
            for di, delta in enumerate([+1, -1]):
                if self.inner.can_move(vi, delta):
                    mask[vi * 2 + di] = True
        return mask

    # ── gym API ───────────────────────────────────────────────────────────

    def _potential(self):
        """Potential f(s) ∈ [0, 1]; higher = closer to solved.

        Blends two signals:
          - red-car progress: right-edge column / GRID  (weight 0.7)
          - path clearance: fraction of exit-path cells unblocked  (weight 0.3)

        A vehicle blocks the path if it occupies any cell (col, EXIT_ROW)
        in the columns strictly between the red car's right edge and the exit.
        """
        a = self.inner.vehicles[0]
        red_progress = (a.col + a.size) / GRID

        path_cols = set(range(a.col + a.size, GRID))
        if path_cols:
            occupied = set()
            for v in self.inner.vehicles[1:]:
                if v.horizontal and v.row == EXIT_ROW:
                    occupied.update(c for c in range(v.col, v.col + v.size)
                                    if c in path_cols)
                elif not v.horizontal and v.col in path_cols:
                    if v.row <= EXIT_ROW < v.row + v.size:
                        occupied.add(v.col)
            clearance = 1.0 - len(occupied) / len(path_cols)
        else:
            clearance = 1.0  # red car already at exit

        return 0.7 * red_progress + 0.3 * clearance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.inner.reset()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        vi = int(action) // 2
        delta = +1 if int(action) % 2 == 0 else -1

        f_before = self._potential()

        if vi >= len(self.inner.vehicles):
            reward = -1.0
            msg = 'no such vehicle'
        else:
            ok, msg = self.inner.move(vi, delta)
            if not ok:
                reward = -1.0
            else:
                f_after = self._potential()
                reward = self.gamma * f_after - f_before - 0.01

        solved = self.inner.is_solved()
        if solved:
            reward += 10.0
        done = solved or self.step_count >= self.max_steps

        return self._get_obs(), reward, done, False, {'msg': msg, 'solved': solved}

    def render(self, mode='human', save_path=None):
        step_info = f'step {self.step_count}'
        self.inner.render(title=f'Rush Hour — {step_info}',
                          inline=(mode == 'human'), save_path=save_path)


# ── Quick demo ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Example puzzle from the Fogleman database
    example = 'IBBxooIooLDDJAALooJoKEEMFFKooMGGHHHM'
    print('Board string:', example)

    vehicles = parse_board(example)
    print(f'Parsed {len(vehicles)} vehicles:')
    for v in vehicles:
        print(' ', v)

    env = RushHourGym(example)
    obs, _ = env.reset()
    print('\nh_rep shape:', obs['h_rep'].shape)
    print('v_rep shape:', obs['v_rep'].shape)
    print('Action mask (first 10):', env.get_action_mask()[:10])

    env.inner.render(title='Initial state')

    # Take a few random valid moves
    import random
    for _ in range(10):
        mask = env.get_action_mask()
        valid = np.where(mask)[0]
        if len(valid) == 0:
            break
        action = random.choice(valid)
        obs, reward, done, _, info = env.step(action)
        vi, delta = action // 2, '+1' if action % 2 == 0 else '-1'
        print(f'  action={action} (vehicle {vehicles[vi].vid if vi < len(vehicles) else "?"} {delta})'
              f' reward={reward:.2f}  msg={info["msg"]}')
        if done:
            print('  Solved!' if info['solved'] else '  Max steps reached.')
            break

    env.inner.render(title='After moves')
