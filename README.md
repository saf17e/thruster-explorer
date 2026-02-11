# AUV Thruster Explorer

An interactive GUI tool for designing, analyzing, and optimizing thruster configurations for Autonomous Underwater Vehicles (AUVs).

---

## Overview

This tool helps you explore how different numbers and arrangements of thrusters affect an AUV's ability to move and rotate in all six degrees of freedom (DOFs):

| DOF | Symbol | Direction |
|-----|--------|-----------|
| Surge | X | Forward / Backward |
| Sway | Y | Left / Right (Port / Starboard) |
| Heave | Z | Up / Down |
| Roll | K | Rotation about X-axis |
| Pitch | M | Rotation about Y-axis |
| Yaw | N | Rotation about Z-axis |

---

## How It Works

### Coordinate Frame (NED-like)

The vehicle uses a body-fixed coordinate frame:
- **+X** = Forward (Surge)
- **+Y** = Starboard (Sway)
- **+Z** = Downward (Heave)

The vehicle is modeled as a uniform-density rectangular box with the center of mass at the geometric center.

### Allocation Matrix (τ = B · u)

The core math is the **thruster allocation matrix** `B`, a 6×n matrix where n is the number of thrusters:

```
τ = B · u
```

- **τ** ∈ ℝ⁶ = desired force/torque vector `[Fx, Fy, Fz, Mx, My, Mz]`
- **u** ∈ ℝⁿ = thruster command vector (each thruster's power from -1 to +1)
- **B** ∈ ℝ⁶ˣⁿ = allocation matrix

Each column of B encodes what force and torque a single thruster contributes:
- **Rows 0–2**: Force direction (from thruster orientation)
- **Rows 3–5**: Torque contribution, computed as `τ = r × F` where `r` is the thruster's position relative to the center of mass

### DOF Partitioning

DOF partitions split the 6 DOFs into separate **thruster sets** — groups of thrusters that are each responsible for controlling specific DOFs. This is useful for:

- **Decoupled control**: e.g., horizontal thrusters handle Surge+Sway+Yaw while vertical thrusters handle Heave+Roll+Pitch
- **Fault tolerance analysis**: understanding what capability remains if a set fails
- **Simplified control software**: each set can be controlled independently

The tool generates **all** mathematically possible partitions using Stirling numbers of the second kind:

| Sets | Count | Description |
|------|-------|-------------|
| 2 | 31 | All ways to split 6 DOFs into 2 non-empty groups |
| 3 | 90 | All ways to split 6 DOFs into 3 non-empty groups |
| 4 | 65 | All ways to split 6 DOFs into 4 non-empty groups |
| 5 | 15 | All ways to split 6 DOFs into 5 non-empty groups |
| 6 | 1 | Each DOF in its own group |

**Every possible combination is included** — for example, Heave+Yaw together, Surge+Pitch together, etc.

### Common Preset Configurations

Quick-access presets are provided for popular AUV designs:

| Preset | Set A | Set B | Use Case |
|--------|-------|-------|----------|
| **Horizontal / Vertical** | Surge+Sway+Yaw | Heave+Roll+Pitch | Most common AUV layout |
| **Translation / Rotation** | Surge+Sway+Heave | Roll+Pitch+Yaw | Separate linear and angular motion |
| **Planar / Lateral** | Surge+Heave+Pitch | Sway+Roll+Yaw | Longitudinal plane vs lateral plane |
| **Heave+Yaw / Rest** | Heave+Yaw | Surge+Sway+Roll+Pitch | Depth+heading control separated |
| **Surge+Yaw / Rest** | Surge+Yaw | Sway+Heave+Roll+Pitch | Forward+heading control separated |

### Analysis Features

1. **Matrix Rank**: Determines if the configuration is fully actuated (rank 6 = can independently control all 6 DOFs)
2. **Efficiency Scores**: Per-DOF measure of how much thrust authority is available
3. **Cosine Losses**: How much thrust is "wasted" due to angled thrusters (a 45° thruster loses ~29% efficiency per axis)
4. **Coupling Analysis**: Cross-coupling matrix `C = B · B⁺` — ideal is identity matrix (no coupling between DOFs)
5. **Subset Analysis**: Whether each thruster set can independently control its target DOFs without disturbing others
6. **Failure Analysis**: What happens when one or more thrusters fail

### Optimizer

The stochastic optimizer uses simulated annealing to search for optimal thruster positions and orientations:

- **Scoring**: Rewards full controllability (+50 per set), target DOF authority (+10× each), and penalizes cross-coupling (-20× each)
- **Mutation**: Randomly adjusts thruster positions (±2.5 cm), orientations (±0.15 rad), and set assignments
- **Time-bounded**: Runs for 1–60 minutes with periodic random restarts for exploration
- **Seeds**: Starts from both the default configuration and random seeds for diversity

---

## Default Thruster Layouts

The tool provides sensible default placements based on thruster count:

| Count | Layout |
|-------|--------|
| **6** | 4 horizontal vectored at corners (45° in XY) + 2 vertical on centerline |
| **7** | 6-thruster layout + 1 offset vertical for roll |
| **8** | 4 horizontal vectored + 4 vertical at corners (full 6-DOF) |
| **9** | 8-thruster + 1 pure surge at stern |
| **10** | 9-thruster + 1 lateral sway (port) |
| **11** | 10-thruster + 1 lateral sway (starboard) |
| **12** | 11-thruster + 1 center vertical for redundant heave |

---

## Usage

### Requirements

- Python 3.8+
- NumPy
- Tkinter (included with Python on most systems)

### Install & Run

```bash
pip install numpy
python app.py
```

### GUI Workflow

1. **Set thruster count** (slider, 6–12)
2. **Choose DOF set count** (2–6 groups) or click a **preset button**
3. **Browse/filter** the DOF combination list
   - Type DOF names (e.g., `yaw heave`) to find all partitions where those DOFs are in the **same group**
   - Supports `+` separator: `yaw+heave` works too
4. Click **Quick Analysis** for instant results with default thruster placement
5. Click **Run Optimizer** to search for the best thruster configuration (1–60 min)
6. Use **Failure Test** to simulate thruster failures
7. **Export** results as Python code or text report

### Exported Python Code

The export generates a self-contained `ThrusterMixer` class with:
- The full allocation matrix `B`
- Pseudo-inverse `B_pinv` for control allocation
- An `allocate(tau_desired)` method that converts a desired 6-DOF force/torque vector into thruster commands

```python
mixer = ThrusterMixer()
u = mixer.allocate([1, 0, 0, 0, 0, 0])  # Pure forward surge
```

---

## Project Structure

```
├── app.py       # GUI application (Tkinter)
├── engine.py    # Core computation engine (allocation, optimization, analysis)
├── test.py      # Test suite for engine module
└── README.md    # This file
```

### engine.py — Core Classes

| Class / Function | Purpose |
|-----------------|---------|
| `Vehicle` | Rectangular AUV body model with inertia tensor |
| `Thruster` | Single thruster with position, orientation, force/torque |
| `AllocationMatrix` | Builds and analyzes the 6×n matrix B |
| `ThrusterOptimizer` | Time-bounded simulated annealing optimizer |
| `generate_dof_partitions(n)` | Enumerates all partitions of 6 DOFs into n sets |
| `full_analysis_report(...)` | Generates formatted text report |
| `generate_python_code(...)` | Exports mixing matrix as Python class |

### app.py — GUI

| Component | Description |
|-----------|-------------|
| Left panel | Vehicle params, thruster count, DOF sets, presets, filter, actions |
| Right panel | Scrollable analysis report with syntax highlighting |
| Menu bar | File export, analysis commands, failure test |

---

## Running Tests

```bash
python test.py
```

Covers: vehicle model, thruster model, allocation matrix, rank checks, efficiency scores, DOF partitions, subset analysis, failure analysis, coupling, report generation, code export, and a quick optimizer run.
