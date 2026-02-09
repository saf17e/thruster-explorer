"""
AUV Thruster Configuration Engine
===================================
Core computation module for thruster analysis and optimization.

Coordinate Frame (NED-like):
  +X = Forward   (Surge)
  +Y = Starboard (Sway)
  +Z = Downward  (Heave)

6 DOFs:
  Linear:  Surge (X), Sway (Y), Heave (Z)
  Angular: Roll (K), Pitch (M), Yaw (N)

Allocation Model:
  τ = B · u
  where τ ∈ ℝ⁶ = [Fx, Fy, Fz, Mx, My, Mz]ᵀ
        u ∈ ℝⁿ = thruster command vector
        B ∈ ℝ⁶ˣⁿ = allocation matrix
"""

import numpy as np
from itertools import combinations
import time
import copy
import math

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

DOF_LABELS = ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw']
DOF_SYMBOLS = ['X', 'Y', 'Z', 'K', 'M', 'N']
DOF_FULL = [
    'Surge (X)', 'Sway (Y)', 'Heave (Z)',
    'Roll (K)', 'Pitch (M)', 'Yaw (N)'
]


# ═══════════════════════════════════════════════════════════════
# Vehicle Model
# ═══════════════════════════════════════════════════════════════

class Vehicle:
    """
    Uniform-density rectangular AUV body model.
    
    For a uniform rectangular parallelepiped:
      COM = geometric center
      Inertia tensor is diagonal with:
        Ixx = (1/12) m (B² + H²)
        Iyy = (1/12) m (L² + H²)
        Izz = (1/12) m (L² + B²)
    """

    def __init__(self, length=60.0, breadth=40.0, height=20.0, mass=10.0):
        self.L = length    # cm, along X-axis (Surge)
        self.B = breadth   # cm, along Y-axis (Sway)
        self.H = height    # cm, along Z-axis (Heave)
        self.mass = mass   # kg (normalized)

        # Uniform body → COM at geometric center
        self.com = np.array([0.0, 0.0, 0.0])

        # Inertia tensor (diagonal for uniform rect box)
        self.inertia = np.diag([
            (1 / 12) * self.mass * (self.B ** 2 + self.H ** 2),
            (1 / 12) * self.mass * (self.L ** 2 + self.H ** 2),
            (1 / 12) * self.mass * (self.L ** 2 + self.B ** 2),
        ])

    def bounds(self):
        """Return half-extents along each axis."""
        return self.L / 2, self.B / 2, self.H / 2

    def __repr__(self):
        return f"Vehicle(L={self.L}, B={self.B}, H={self.H}, COM={self.com})"


# ═══════════════════════════════════════════════════════════════
# Thruster Model
# ═══════════════════════════════════════════════════════════════

class Thruster:
    """
    A single thruster defined by position and thrust direction.
    
    Attributes
    ----------
    position : ndarray (3,)
        Location [x, y, z] relative to vehicle COM (cm).
    orientation : ndarray (3,)
        Unit vector in the direction of force applied to the vehicle
        when the thruster fires at positive power.
    max_thrust : float
        Maximum thrust magnitude (N), normalized to 1.0 by default.
    name : str
        Human-readable label.
    """

    def __init__(self, position, orientation, max_thrust=1.0, name="T?"):
        self.position = np.array(position, dtype=float)
        self.orientation = np.array(orientation, dtype=float)
        norm = np.linalg.norm(self.orientation)
        if norm > 1e-12:
            self.orientation /= norm
        self.max_thrust = max_thrust
        self.name = name

    def force_vector(self, power=1.0):
        """Force applied to vehicle at given power fraction [-1, 1]."""
        return self.orientation * self.max_thrust * power

    def torque_vector(self, com=None, power=1.0):
        """Torque about COM: τ = r × F."""
        if com is None:
            com = np.zeros(3)
        r = self.position - com
        return np.cross(r, self.force_vector(power))

    def copy(self):
        return Thruster(
            self.position.copy(), self.orientation.copy(),
            self.max_thrust, self.name
        )

    def __repr__(self):
        p = self.position
        o = self.orientation
        return (f"{self.name}: pos=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f}) "
                f"dir=({o[0]:.3f},{o[1]:.3f},{o[2]:.3f})")


# ═══════════════════════════════════════════════════════════════
# Default Thruster Configurations
# ═══════════════════════════════════════════════════════════════

def default_thruster_config(n, vehicle):
    """
    Generate sensible default thruster positions for n thrusters.
    
    Layout strategy:
      - First 4: Horizontal vectored at 45° in XY plane (corners)
                 → provides Surge, Sway, Yaw
      - Next 2-4: Vertical thrusters
                 → provides Heave, Roll, Pitch
      - Remaining: Supplementary thrusters for redundancy
    """
    Lh, Bh, Hh = vehicle.L / 2, vehicle.B / 2, vehicle.H / 2

    # Corner offsets (slightly inboard)
    cx = Lh * 0.70   # 21 cm from center
    cy = Bh * 0.75   # 15 cm from center
    hz = -Hh * 0.5   # horizontal thrusters above center-plane

    c45 = math.cos(math.radians(45))

    thrusters = []

    # ── Horizontal vectored thrusters (Surge / Sway / Yaw) ──────
    # T1: Front-Port → thrust direction Forward + Starboard
    thrusters.append(Thruster([cx, -cy, hz], [c45, c45, 0], name="T1"))
    # T2: Front-Starboard → thrust direction Forward + Port
    thrusters.append(Thruster([cx, cy, hz], [c45, -c45, 0], name="T2"))
    # T3: Rear-Port → thrust direction Aft + Port
    thrusters.append(Thruster([-cx, -cy, hz], [-c45, -c45, 0], name="T3"))
    # T4: Rear-Starboard → thrust direction Aft + Starboard
    thrusters.append(Thruster([-cx, cy, hz], [-c45, c45, 0], name="T4"))

    if n == 6:
        # ── 2 vertical thrusters (center-line, fore/aft) ────────
        thrusters.append(Thruster([cx * 0.6, 0, 0], [0, 0, 1], name="T5"))
        thrusters.append(Thruster([-cx * 0.6, 0, 0], [0, 0, 1], name="T6"))

    elif n == 7:
        # 2 vertical on center + 1 vertical offset for roll
        thrusters.append(Thruster([cx * 0.6, 0, 0], [0, 0, 1], name="T5"))
        thrusters.append(Thruster([-cx * 0.6, 0, 0], [0, 0, 1], name="T6"))
        thrusters.append(Thruster([0, -cy, 0], [0, 0, 1], name="T7"))

    elif n >= 8:
        # ── 4 vertical thrusters at corners (full attitude) ─────
        vy = cy * 0.6
        vx = cx * 0.5
        thrusters.append(Thruster([vx, -vy, 0], [0, 0, 1], name="T5"))
        thrusters.append(Thruster([vx, vy, 0], [0, 0, 1], name="T6"))
        thrusters.append(Thruster([-vx, -vy, 0], [0, 0, 1], name="T7"))
        thrusters.append(Thruster([-vx, vy, 0], [0, 0, 1], name="T8"))

    if n >= 9:
        # Pure surge thruster at stern
        thrusters.append(Thruster([-Lh, 0, 0], [1, 0, 0], name="T9"))

    if n >= 10:
        # Lateral sway thruster port-side
        thrusters.append(Thruster([0, -Bh, 0], [0, 1, 0], name="T10"))

    if n >= 11:
        # Lateral sway thruster starboard-side
        thrusters.append(Thruster([0, Bh, 0], [0, -1, 0], name="T11"))

    if n >= 12:
        # Center vertical for redundant heave
        thrusters.append(Thruster([0, 0, 0], [0, 0, 1], name="T12"))

    return thrusters[:n]


# ═══════════════════════════════════════════════════════════════
# Allocation Matrix & Analysis
# ═══════════════════════════════════════════════════════════════

class AllocationMatrix:
    """
    Constructs and analyzes the 6×n thruster allocation matrix B.
    
    The mapping is:
      τ = B · u
    
    where row ordering is [Surge, Sway, Heave, Roll, Pitch, Yaw].
    """

    def __init__(self, vehicle, thrusters):
        self.vehicle = vehicle
        self.thrusters = thrusters
        self.n = len(thrusters)
        self.B = self._build()

    def _build(self):
        B = np.zeros((6, self.n))
        for i, t in enumerate(self.thrusters):
            # Force contributions
            B[0, i] = t.orientation[0]   # Surge
            B[1, i] = t.orientation[1]   # Sway
            B[2, i] = t.orientation[2]   # Heave

            # Torque contributions: τ = r × F_direction
            r = t.position - self.vehicle.com
            moment = np.cross(r, t.orientation)
            B[3, i] = moment[0]   # Roll  (about X)
            B[4, i] = moment[1]   # Pitch (about Y)
            B[5, i] = moment[2]   # Yaw   (about Z)
        return B

    def rank(self):
        return np.linalg.matrix_rank(self.B)

    def is_fully_actuated(self):
        return self.rank() >= 6

    # ── Efficiency ──────────────────────────────────────────

    def efficiency_scores(self):
        """
        Per-DOF efficiency: ratio of achievable force/torque
        to total available thrust if all thrusters were aligned.
        Returns dict {DOF_name: percentage}.
        """
        total = sum(t.max_thrust for t in self.thrusters)
        scores = {}
        for i, name in enumerate(DOF_LABELS):
            achievable = np.sum(np.abs(self.B[i, :]))
            scores[name] = round((achievable / total) * 100, 1) if total > 0 else 0.0
        return scores

    def cosine_losses(self):
        """
        Per-thruster projection efficiency onto each body axis.
        cos(angle) = |dot(orientation, axis)|
        """
        axes = {
            'X (Surge)': np.array([1, 0, 0]),
            'Y (Sway)':  np.array([0, 1, 0]),
            'Z (Heave)': np.array([0, 0, 1]),
        }
        losses = {}
        for t in self.thrusters:
            t_eff = {}
            for axis_name, axis_vec in axes.items():
                t_eff[axis_name] = round(abs(np.dot(t.orientation, axis_vec)) * 100, 1)
            losses[t.name] = t_eff
        return losses

    # ── Coupling ────────────────────────────────────────────

    def coupling_matrix(self):
        """
        6×6 coupling matrix: C = B · B⁺
        Ideal = Identity (perfectly decoupled).
        Off-diagonal entries indicate cross-coupling.
        """
        B_pinv = np.linalg.pinv(self.B)
        return self.B @ B_pinv

    def coupling_report(self):
        """Identify significant cross-coupling pairs."""
        C = self.coupling_matrix()
        issues = []
        for i in range(6):
            for j in range(6):
                if i != j and abs(C[i, j]) > 0.05:
                    issues.append({
                        'from': DOF_LABELS[i],
                        'to': DOF_LABELS[j],
                        'magnitude': round(C[i, j], 4),
                        'severity': 'HIGH' if abs(C[i, j]) > 0.2 else 'MODERATE'
                    })
        return issues

    # ── Subset / Set Analysis ───────────────────────────────

    def subset_analysis(self, thruster_indices, target_dof_indices):
        """
        Analyze whether a subset of thrusters can control target DOFs
        without coupling to non-target DOFs.
        
        Parameters
        ----------
        thruster_indices : list[int]
            Indices of thrusters in this set.
        target_dof_indices : list[int]
            DOF indices (0-5) this set should exclusively control.
        
        Returns
        -------
        dict with controllability, authority, and cross-coupling info.
        """
        sub_B = self.B[:, thruster_indices]
        non_target = [d for d in range(6) if d not in target_dof_indices]

        # Controllability of target DOFs
        target_rows = sub_B[target_dof_indices, :]
        target_rank = np.linalg.matrix_rank(target_rows, tol=1e-6)

        # Cross-coupling magnitude to non-target DOFs
        non_target_rows = sub_B[non_target, :]
        max_cc = float(np.max(np.abs(non_target_rows))) if len(non_target) > 0 else 0.0

        return {
            'controllable': target_rank == len(target_dof_indices),
            'rank': int(target_rank),
            'required_rank': len(target_dof_indices),
            'max_cross_coupling': round(max_cc, 4),
            'cross_coupling_detail': {
                DOF_LABELS[d]: round(float(np.sum(np.abs(sub_B[d, :]))), 4)
                for d in non_target
            },
            'target_authority': {
                DOF_LABELS[d]: round(float(np.sum(np.abs(sub_B[d, :]))), 4)
                for d in target_dof_indices
            },
        }

    # ── Failure Analysis ────────────────────────────────────

    def failure_analysis(self, failed_indices):
        """
        Analyze remaining capability after thruster failure(s).
        
        Returns per-DOF retention percentage and status.
        """
        remaining = [i for i in range(self.n) if i not in failed_indices]
        sub_B = self.B[:, remaining]
        r = int(np.linalg.matrix_rank(sub_B))

        dof_status = {}
        for i, name in enumerate(DOF_LABELS):
            original = float(np.sum(np.abs(self.B[i, :])))
            reduced = float(np.sum(np.abs(sub_B[i, :])))
            retention = round((reduced / original) * 100, 1) if original > 0 else 0.0

            if retention == 0:
                status = 'LOST'
            elif retention < 30:
                status = 'CRITICAL'
            elif retention < 60:
                status = 'DEGRADED'
            else:
                status = 'OK'

            dof_status[name] = {'retention': retention, 'status': status}

        return {
            'remaining_count': len(remaining),
            'matrix_rank': r,
            'fully_actuated': r >= 6,
            'dof_status': dof_status,
        }


# ═══════════════════════════════════════════════════════════════
# DOF Partition Generator
# ═══════════════════════════════════════════════════════════════

def generate_dof_partitions(num_sets):
    """
    Generate all unique partitions of 6 DOFs into `num_sets` non-empty groups.
    
    Uses restricted growth strings to enumerate set partitions
    without duplicates (Stirling numbers of the second kind).
    
    S(6,2) = 31  |  S(6,3) = 90
    
    Returns list of tuples-of-tuples, e.g.:
      ((0, 5), (1, 2, 3, 4))  →  Set A: Surge+Yaw | Set B: Sway+Heave+Roll+Pitch
    """
    n = 6
    if num_sets < 1 or num_sets > n:
        return []

    results = []

    def _backtrack(idx, assignment, max_label):
        if idx == n:
            if max_label == num_sets - 1:
                partition = [[] for _ in range(num_sets)]
                for i, label in enumerate(assignment):
                    partition[label].append(i)
                results.append(tuple(tuple(s) for s in partition))
            return

        upper = min(max_label + 2, num_sets)
        for label in range(upper):
            assignment.append(label)
            _backtrack(idx + 1, assignment, max(max_label, label))
            assignment.pop()

    _backtrack(0, [], -1)
    return results


def partition_to_string(partition):
    """Human-readable string for a DOF partition."""
    parts = []
    for i, group in enumerate(partition):
        names = [DOF_LABELS[d] for d in group]
        parts.append(f"Set {chr(65 + i)}: {' + '.join(names)}")
    return "  |  ".join(parts)


def partition_short_string(partition):
    """Compact string using DOF symbols."""
    parts = []
    for i, group in enumerate(partition):
        syms = [DOF_SYMBOLS[d] for d in group]
        parts.append(f"{chr(65 + i)}:{','.join(syms)}")
    return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════
# Thruster Optimizer (Time-Bounded)
# ═══════════════════════════════════════════════════════════════

class ThrusterOptimizer:
    """
    Stochastic optimizer that searches for the best thruster
    positions, orientations, and set assignments given:
      - Fixed vehicle geometry
      - Fixed thruster count
      - A specific DOF partition
    
    Uses simulated-annealing-style search within a time budget.
    """

    def __init__(self, vehicle, n_thrusters, dof_partition, runtime_seconds=900):
        self.vehicle = vehicle
        self.n_thrusters = n_thrusters
        self.dof_partition = dof_partition
        self.n_sets = len(dof_partition)
        self.runtime = runtime_seconds

        # State
        self.best_score = -np.inf
        self.best_thrusters = None
        self.best_assignment = None
        self.iterations = 0
        self.running = False
        self.progress = 0.0
        self.status_text = "Idle"

    # ── Scoring ─────────────────────────────────────────────

    def _score(self, thrusters, assignment):
        """
        Evaluate a configuration. Higher is better.
        
        Criteria:
          +50  per set that is fully controllable (rank == required)
          +10 × authority in each target DOF
          -20 × cross-coupling to each non-target DOF
          -1000 if any set has zero thrusters
        """
        am = AllocationMatrix(self.vehicle, thrusters)
        score = 0.0

        for set_idx, dof_group in enumerate(self.dof_partition):
            t_indices = [i for i, a in enumerate(assignment) if a == set_idx]
            if len(t_indices) == 0:
                return -1000.0

            analysis = am.subset_analysis(t_indices, list(dof_group))

            if analysis['controllable']:
                score += 50.0
            else:
                score -= 50.0 * (analysis['required_rank'] - analysis['rank'])

            for auth in analysis['target_authority'].values():
                score += auth * 10.0

            for cc in analysis['cross_coupling_detail'].values():
                score -= cc * 20.0

        # Bonus for overall full-actuation
        if am.is_fully_actuated():
            score += 30.0

        return score

    # ── Random generators ───────────────────────────────────

    def _random_thrusters(self):
        Lh, Bh, Hh = self.vehicle.bounds()
        thrusters = []
        for i in range(self.n_thrusters):
            pos = np.array([
                np.random.uniform(-Lh * 0.9, Lh * 0.9),
                np.random.uniform(-Bh * 0.9, Bh * 0.9),
                np.random.uniform(-Hh * 0.9, Hh * 0.9),
            ])
            orient = np.random.randn(3)
            orient /= np.linalg.norm(orient)
            thrusters.append(Thruster(pos, orient, name=f"T{i + 1}"))
        return thrusters

    def _random_assignment(self):
        """Ensure every set has ≥ 1 thruster."""
        # Guarantee one per set
        base = list(range(self.n_sets))
        extra = [np.random.randint(0, self.n_sets)
                 for _ in range(self.n_thrusters - self.n_sets)]
        assignment = base + extra
        np.random.shuffle(assignment)
        return list(assignment)

    # ── Mutation ────────────────────────────────────────────

    def _mutate(self, thrusters, assignment):
        Lh, Bh, Hh = self.vehicle.bounds()
        new_t = []
        for t in thrusters:
            pos = t.position.copy()
            orient = t.orientation.copy()

            if np.random.random() < 0.35:
                pos += np.random.randn(3) * 2.5
                pos[0] = np.clip(pos[0], -Lh * 0.95, Lh * 0.95)
                pos[1] = np.clip(pos[1], -Bh * 0.95, Bh * 0.95)
                pos[2] = np.clip(pos[2], -Hh * 0.95, Hh * 0.95)

            if np.random.random() < 0.25:
                orient += np.random.randn(3) * 0.15
                norm = np.linalg.norm(orient)
                if norm > 1e-12:
                    orient /= norm

            new_t.append(Thruster(pos, orient, name=t.name))

        new_a = list(assignment)
        if np.random.random() < 0.2:
            idx = np.random.randint(len(new_a))
            new_a[idx] = np.random.randint(0, self.n_sets)
            # Repair: ensure all sets represented
            for s in range(self.n_sets):
                if s not in new_a:
                    repair_idx = np.random.randint(len(new_a))
                    new_a[repair_idx] = s

        return new_t, new_a

    # ── Main loop ───────────────────────────────────────────

    def optimize(self, callback=None):
        """
        Run the optimizer for self.runtime seconds.
        
        callback(progress, iterations, best_score) is called periodically.
        Returns dict with best configuration found.
        """
        self.running = True
        self.status_text = "Initializing..."
        start = time.time()

        # Seed with default config
        self.best_thrusters = default_thruster_config(self.n_thrusters, self.vehicle)
        self.best_assignment = self._random_assignment()
        self.best_score = self._score(self.best_thrusters, self.best_assignment)

        # Try several random seeds
        for _ in range(30):
            t = self._random_thrusters()
            a = self._random_assignment()
            s = self._score(t, a)
            if s > self.best_score:
                self.best_score = s
                self.best_thrusters = t
                self.best_assignment = a

        # Also try default with different assignments
        default_t = default_thruster_config(self.n_thrusters, self.vehicle)
        for _ in range(20):
            a = self._random_assignment()
            s = self._score(default_t, a)
            if s > self.best_score:
                self.best_score = s
                self.best_thrusters = [th.copy() for th in default_t]
                self.best_assignment = a

        self.status_text = "Optimizing..."

        while self.running:
            elapsed = time.time() - start
            if elapsed >= self.runtime:
                break

            self.progress = min(elapsed / self.runtime, 1.0)
            self.iterations += 1

            new_t, new_a = self._mutate(self.best_thrusters, self.best_assignment)
            new_s = self._score(new_t, new_a)

            # Acceptance: greedy + temperature-based exploration
            temperature = max(0.01, 1.0 - self.progress)
            if new_s > self.best_score:
                self.best_score = new_s
                self.best_thrusters = new_t
                self.best_assignment = new_a
            elif np.random.random() < temperature * 0.05:
                # Accept worse solution occasionally for exploration
                self.best_thrusters = new_t
                self.best_assignment = new_a

            # Periodic random restart
            if self.iterations % 800 == 0:
                t = self._random_thrusters()
                a = self._random_assignment()
                s = self._score(t, a)
                if s > self.best_score:
                    self.best_score = s
                    self.best_thrusters = t
                    self.best_assignment = a

            if callback and self.iterations % 200 == 0:
                callback(self.progress, self.iterations, self.best_score)

        self.running = False
        self.progress = 1.0
        self.status_text = "Complete"
        elapsed = time.time() - start

        return {
            'thrusters': self.best_thrusters,
            'assignment': self.best_assignment,
            'score': self.best_score,
            'iterations': self.iterations,
            'elapsed': elapsed,
        }

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════
# ASCII Visualization
# ═══════════════════════════════════════════════════════════════

def ascii_top_view(vehicle, thrusters, assignment=None, width=60, height=28):
    """
    Render ASCII art top view.
    X = Forward (up on screen), Y = Starboard (right on screen).
    """
    L, B = vehicle.L, vehicle.B
    lines = []
    lines.append("=" * width)
    lines.append("  TOP VIEW  (+X Forward ↑,  +Y Starboard →)")
    lines.append("=" * width)

    grid = [[' '] * width for _ in range(height)]

    # Vehicle outline
    x1, x2 = int(width * 0.15), int(width * 0.85)
    y1, y2 = int(height * 0.15), int(height * 0.85)

    for x in range(x1, x2 + 1):
        grid[y1][x] = '─'
        grid[y2][x] = '─'
    for y in range(y1, y2 + 1):
        grid[y][x1] = '│'
        grid[y][x2] = '│'
    grid[y1][x1] = '┌'
    grid[y1][x2] = '┐'
    grid[y2][x1] = '└'
    grid[y2][x2] = '┘'

    # COM
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    grid[cy][cx] = '+'

    # Forward label
    mid_x = (x1 + x2) // 2
    if y1 >= 2:
        grid[y1 - 1][mid_x] = '^'
        fwd = "FWD"
        for k, ch in enumerate(fwd):
            col = mid_x - 1 + k
            if 0 <= col < width and y1 - 2 >= 0:
                grid[y1 - 2][col] = ch

    # Place thrusters
    for idx, t in enumerate(thrusters):
        # Map position to grid coordinates
        # Y → horizontal (starboard = right), X → vertical (forward = up)
        gx = int(cx + (t.position[1] / (B / 2)) * (x2 - cx) * 0.75)
        gy = int(cy - (t.position[0] / (L / 2)) * (cy - y1) * 0.75)
        gx = max(1, min(width - 3, gx))
        gy = max(1, min(height - 2, gy))

        # Symbol based on orientation
        if abs(t.orientation[2]) > 0.7:
            sym = 'o'   # vertical
        else:
            ang = math.degrees(math.atan2(t.orientation[1], t.orientation[0]))
            if -22.5 <= ang < 22.5:
                sym = '>'
            elif 22.5 <= ang < 67.5:
                sym = '\\'
            elif 67.5 <= ang < 112.5:
                sym = 'v'
            elif ang >= 112.5 or ang < -112.5:
                sym = '<'
            elif -112.5 <= ang < -67.5:
                sym = '^'
            elif -67.5 <= ang < -22.5:
                sym = '/'
            else:
                sym = '*'

        # Set label
        set_tag = ""
        if assignment is not None:
            set_tag = chr(65 + assignment[idx])

        grid[gy][gx] = sym
        label = f"{t.name}{set_tag}"
        for k, ch in enumerate(label):
            col = gx + 1 + k
            if col < width:
                grid[gy][col] = ch

    for row in grid:
        lines.append(''.join(row))
    return '\n'.join(lines)


def ascii_side_view(vehicle, thrusters, width=60, height=18):
    """
    Render ASCII art side view.
    X = Forward (right), Z = Down (down on screen).
    """
    L, H = vehicle.L, vehicle.H
    lines = []
    lines.append("=" * width)
    lines.append("  SIDE VIEW  (+X Forward →,  +Z Down ↓)")
    lines.append("=" * width)

    grid = [[' '] * width for _ in range(height)]

    x1, x2 = int(width * 0.15), int(width * 0.85)
    y1, y2 = int(height * 0.2), int(height * 0.8)

    for x in range(x1, x2 + 1):
        grid[y1][x] = '─'
        grid[y2][x] = '─'
    for y in range(y1, y2 + 1):
        grid[y][x1] = '│'
        grid[y][x2] = '│'
    grid[y1][x1] = '┌'
    grid[y1][x2] = '┐'
    grid[y2][x1] = '└'
    grid[y2][x2] = '┘'

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    grid[cy][cx] = '+'

    for t in thrusters:
        gx = int(cx + (t.position[0] / (L / 2)) * (x2 - cx) * 0.75)
        gy = int(cy + (t.position[2] / (H / 2)) * (y2 - cy) * 0.75)
        gx = max(1, min(width - 4, gx))
        gy = max(1, min(height - 2, gy))

        if abs(t.orientation[2]) > 0.7:
            sym = '|'
        elif abs(t.orientation[0]) > 0.7:
            sym = '-'
        else:
            sym = '*'

        grid[gy][gx] = sym
        label = t.name
        for k, ch in enumerate(label):
            col = gx + 1 + k
            if col < width:
                grid[gy][col] = ch

    for row in grid:
        lines.append(''.join(row))
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# Report Formatting
# ═══════════════════════════════════════════════════════════════

def format_matrix_table(matrix, row_labels, col_labels):
    """Pretty-print a matrix with labels."""
    col_w = max(9, max(len(c) for c in col_labels) + 1)
    header = f"{'':>10s} | " + " | ".join(f"{c:>{col_w}s}" for c in col_labels)
    sep = "-" * len(header)
    lines = [header, sep]
    for i, rl in enumerate(row_labels):
        vals = " | ".join(f"{matrix[i, j]:>{col_w}.4f}" for j in range(matrix.shape[1]))
        lines.append(f"{rl:>10s} | {vals}")
    return '\n'.join(lines)


def full_analysis_report(vehicle, thrusters, dof_partition=None,
                         thruster_assignment=None):
    """
    Generate comprehensive analysis report as a formatted string.
    """
    am = AllocationMatrix(vehicle, thrusters)
    R = []

    R.append("+" + "=" * 62 + "+")
    R.append("|     AUV THRUSTER CONFIGURATION — ANALYSIS REPORT            |")
    R.append("+" + "=" * 62 + "+")
    R.append("")

    # ── Vehicle ─────────────────────────────────────────────
    R.append(">>> VEHICLE PARAMETERS")
    R.append(f"    Length  (X / Surge): {vehicle.L:6.1f} cm")
    R.append(f"    Breadth (Y / Sway):  {vehicle.B:6.1f} cm")
    R.append(f"    Height  (Z / Heave): {vehicle.H:6.1f} cm")
    R.append(f"    COM:  ({vehicle.com[0]:.1f}, {vehicle.com[1]:.1f}, {vehicle.com[2]:.1f})")
    R.append(f"    Body:  Uniform density → COM at geometric center")
    R.append("")

    # ── Thruster Table ──────────────────────────────────────
    R.append(f">>> THRUSTER CONFIGURATION  ({len(thrusters)} thrusters)")
    R.append(f"    {'Name':>6s}  {'Position (x, y, z)':>28s}  {'Orientation (x, y, z)':>28s}")
    R.append("    " + "-" * 66)
    for t in thrusters:
        p = t.position
        o = t.orientation
        ps = f"({p[0]:7.2f}, {p[1]:7.2f}, {p[2]:7.2f})"
        os = f"({o[0]:7.4f}, {o[1]:7.4f}, {o[2]:7.4f})"
        R.append(f"    {t.name:>6s}  {ps:>28s}  {os:>28s}")
    R.append("")

    # ── ASCII Views ─────────────────────────────────────────
    R.append(ascii_top_view(vehicle, thrusters, thruster_assignment))
    R.append("")
    R.append(ascii_side_view(vehicle, thrusters))
    R.append("")

    # ── Allocation Matrix ───────────────────────────────────
    R.append(">>> ALLOCATION MATRIX  B  (tau = B * u)")
    R.append("")
    col_labels = [t.name for t in thrusters]
    R.append(format_matrix_table(am.B, DOF_LABELS, col_labels))
    R.append("")

    # ── Rank ────────────────────────────────────────────────
    r = am.rank()
    R.append(f">>> MATRIX RANK: {r} / 6")
    if r >= 6:
        R.append("    [OK] FULLY ACTUATED — all 6 DOFs independently controllable")
    else:
        R.append(f"    [!!] UNDER-ACTUATED — only {r} DOFs independently controllable")
    R.append("")

    # ── Efficiency ──────────────────────────────────────────
    R.append(">>> EFFICIENCY SCORES  (per DOF)")
    scores = am.efficiency_scores()
    for name, score in scores.items():
        bar_len = int(score / 5)
        bar = '#' * bar_len + '.' * (20 - bar_len)
        flag = ""
        if score == 0:
            flag = "  << UNACTUATED"
        elif score < 30:
            flag = "  << WEAKLY ACTUATED"
        R.append(f"    {name:>8s}: [{bar}] {score:5.1f}%{flag}")
    R.append("")

    # ── Cosine Losses ───────────────────────────────────────
    R.append(">>> COSINE LOSSES  (direct force efficiency per axis)")
    losses = am.cosine_losses()
    for tname, tloss in losses.items():
        parts = " | ".join(f"{k}={v:5.1f}%" for k, v in tloss.items())
        R.append(f"    {tname:>6s}: {parts}")
    R.append("")

    # ── Coupling ────────────────────────────────────────────
    R.append(">>> COUPLING ANALYSIS  (C = B * pinv(B), ideal = I)")
    C = am.coupling_matrix()
    R.append(format_matrix_table(C, DOF_LABELS, DOF_LABELS))
    R.append("")

    issues = am.coupling_report()
    if issues:
        R.append("    Cross-coupling warnings:")
        for iss in issues:
            R.append(f"      [{iss['severity']:>8s}] {iss['from']:>6s} -> {iss['to']:>6s}: "
                     f"{iss['magnitude']:+.4f}")
    else:
        R.append("    [OK] No significant cross-coupling detected.")
    R.append("")

    # ── DOF Set Analysis ────────────────────────────────────
    if dof_partition and thruster_assignment:
        R.append(">>> DOF SET ANALYSIS")
        R.append("")

        for set_idx, dof_group in enumerate(dof_partition):
            set_name = f"Set {chr(65 + set_idx)}"
            t_indices = [i for i, a in enumerate(thruster_assignment) if a == set_idx]
            dof_names = [DOF_LABELS[d] for d in dof_group]
            t_names = [thrusters[i].name for i in t_indices]

            R.append(f"    {set_name}")
            R.append(f"      Target DOFs:  {dof_names}")
            R.append(f"      Thrusters:    {t_names}")

            if t_indices:
                analysis = am.subset_analysis(t_indices, list(dof_group))
                ctrl = "YES" if analysis['controllable'] else "NO"
                R.append(f"      Controllable: [{ctrl}]  "
                         f"(rank {analysis['rank']}/{analysis['required_rank']})")

                R.append(f"      Target Authority:")
                for dn, auth in analysis['target_authority'].items():
                    R.append(f"        {dn:>8s}: {auth:.4f}")

                if analysis['cross_coupling_detail']:
                    R.append(f"      Cross-Coupling to non-target DOFs:")
                    feasible = True
                    for dn, cc in analysis['cross_coupling_detail'].items():
                        flag = "  << WARNING" if cc > 0.1 else ""
                        if cc > 0.1:
                            feasible = False
                        R.append(f"        {dn:>8s}: {cc:.4f}{flag}")
                    if feasible:
                        R.append(f"      ==> PURE CONTROL: feasible (low cross-coupling)")
                    else:
                        R.append(f"      ==> PURE CONTROL: NOT feasible — counter-mixing required")
            else:
                R.append(f"      [!!] NO THRUSTERS ASSIGNED TO THIS SET")

            R.append("")

    # ── Counter-mixing Suggestions ──────────────────────────
    if dof_partition and thruster_assignment:
        R.append(">>> COUNTER-MIXING SUGGESTIONS")
        any_suggestions = False
        for set_idx, dof_group in enumerate(dof_partition):
            t_indices = [i for i, a in enumerate(thruster_assignment) if a == set_idx]
            if not t_indices:
                continue
            analysis = am.subset_analysis(t_indices, list(dof_group))
            high_cc = {k: v for k, v in analysis['cross_coupling_detail'].items() if v > 0.1}
            if high_cc:
                any_suggestions = True
                set_name = f"Set {chr(65 + set_idx)}"
                R.append(f"    {set_name} has cross-coupling issues:")
                for dn, cc in high_cc.items():
                    R.append(f"      - {dn} coupling ({cc:.4f}): consider repositioning")
                    R.append(f"        thrusters further from COM along the {dn} axis,")
                    R.append(f"        or add differential thrust mixing.")
        if not any_suggestions:
            R.append("    No counter-mixing needed — configuration is clean.")
        R.append("")

    return '\n'.join(R)


# ═══════════════════════════════════════════════════════════════
# Python Code Generator (for export)
# ═══════════════════════════════════════════════════════════════

def generate_python_code(vehicle, thrusters, dof_partition=None,
                         thruster_assignment=None):
    """
    Generate a self-contained Python class snippet
    defining the mixing matrix for the current setup.
    """
    lines = []
    lines.append('"""')
    lines.append("Auto-generated AUV Thruster Mixing Matrix")
    lines.append(f"Vehicle: {vehicle.L}x{vehicle.B}x{vehicle.H} cm")
    lines.append(f"Thrusters: {len(thrusters)}")
    lines.append('"""')
    lines.append("")
    lines.append("import numpy as np")
    lines.append("")
    lines.append("")
    lines.append("class ThrusterMixer:")
    lines.append(f'    """Mixing matrix for {len(thrusters)}-thruster AUV configuration."""')
    lines.append("")
    lines.append("    DOF_NAMES = ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw']")
    lines.append("")
    lines.append("    def __init__(self):")

    # Build B matrix
    am = AllocationMatrix(vehicle, thrusters)
    lines.append("        # 6×n Allocation Matrix (tau = B @ u)")
    lines.append("        self.B = np.array([")
    for i in range(6):
        row = ", ".join(f"{am.B[i, j]:8.4f}" for j in range(am.n))
        comma = "," if i < 5 else ""
        lines.append(f"            [{row}]{comma}  # {DOF_LABELS[i]}")
    lines.append("        ])")
    lines.append("")
    lines.append("        # Pseudo-inverse for control allocation")
    lines.append("        self.B_pinv = np.linalg.pinv(self.B)")
    lines.append("")

    if dof_partition and thruster_assignment:
        lines.append("        # Thruster set assignments")
        for set_idx, dof_group in enumerate(dof_partition):
            t_idx = [i for i, a in enumerate(thruster_assignment) if a == set_idx]
            dof_names = [DOF_LABELS[d] for d in dof_group]
            lines.append(f"        self.set_{chr(65 + set_idx)}_thrusters = {t_idx}"
                         f"  # {dof_names}")
        lines.append("")

    lines.append("    def allocate(self, tau_desired):")
    lines.append('        """')
    lines.append("        Compute thruster commands for desired force/torque vector.")
    lines.append("        tau_desired: [Surge, Sway, Heave, Roll, Pitch, Yaw]")
    lines.append("        Returns: thruster command vector u")
    lines.append('        """')
    lines.append("        tau = np.array(tau_desired, dtype=float)")
    lines.append("        u = self.B_pinv @ tau")
    lines.append("        return np.clip(u, -1.0, 1.0)")
    lines.append("")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    mixer = ThrusterMixer()")
    lines.append("    # Example: pure forward surge")
    lines.append("    u = mixer.allocate([1, 0, 0, 0, 0, 0])")
    lines.append("    print('Surge command:', u)")

    return '\n'.join(lines)
