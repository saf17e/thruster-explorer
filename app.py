"""
AUV Thruster Explorer — Interactive GUI Application
=====================================================
A comprehensive tool for designing, analyzing, and optimizing
thruster configurations for Autonomous Underwater Vehicles.

Features:
  - Adjustable thruster count (6–12)
  - DOF set partitioning (2 or 3 sets) with all possible combinations
  - Time-bounded stochastic optimization (1 min – 1 hour)
  - Full 6-DOF allocation matrix, efficiency, and coupling analysis
  - ASCII visualization of thruster layout
  - Python mixing-matrix code export

Dependencies: numpy, tkinter (stdlib)
Usage: python app.py
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import math
import sys
import os

# Ensure engine module is importable from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import (
    Vehicle, Thruster, AllocationMatrix,
    default_thruster_config, generate_dof_partitions,
    partition_to_string, partition_short_string,
    ThrusterOptimizer, full_analysis_report,
    generate_python_code, DOF_LABELS, DOF_SYMBOLS,
)
from visualize import visualize_config


# ═══════════════════════════════════════════════════════════════
# Color / Style Constants
# ═══════════════════════════════════════════════════════════════

BG_DARK = "#1e1e2e"
BG_PANEL = "#2b2b3d"
BG_INPUT = "#363650"
FG_TEXT = "#cdd6f4"
FG_DIM = "#7f849c"
FG_ACCENT = "#89b4fa"
FG_GREEN = "#a6e3a1"
FG_YELLOW = "#f9e2af"
FG_RED = "#f38ba8"
FONT_MONO = ("Menlo", 11)
FONT_MONO_SM = ("Menlo", 10)
FONT_HEADER = ("Helvetica", 13, "bold")
FONT_LABEL = ("Helvetica", 11)


# ═══════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════

class ThrusterExplorerApp:
    """Main GUI application for AUV thruster exploration."""

    def __init__(self, root):
        self.root = root
        self.root.title("AUV Thruster Explorer v1.0")
        self.root.geometry("1500x950")
        self.root.minsize(1200, 700)

        # ── State variables ─────────────────────────────────
        self.vehicle = Vehicle(60, 40, 20)
        self.n_thrusters_var = tk.IntVar(value=8)
        self.n_sets_var = tk.IntVar(value=2)
        self.runtime_var = tk.StringVar(value="15")
        self.filter_var = tk.StringVar(value="")
        self.partitions = []
        self.optimizer = None
        self.optimizer_thread = None
        self.current_thrusters = None
        self.current_assignment = None

        # ── Build UI ────────────────────────────────────────
        self._build_menu()
        self._build_layout()
        self._populate_partitions()

        # Initial quick analysis with defaults
        self.root.after(300, self._quick_analysis)

    # ═══════════════════════════════════════════════════════
    # Menu Bar
    # ═══════════════════════════════════════════════════════

    def _build_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Export Python Code...", command=self._export_python)
        file_menu.add_command(label="Export Report...", command=self._export_report)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        analysis_menu = tk.Menu(menubar, tearoff=0)
        analysis_menu.add_command(label="Quick Analysis", command=self._quick_analysis)
        analysis_menu.add_command(label="Run Optimizer", command=self._run_optimizer)
        analysis_menu.add_command(label="Failure Test...", command=self._failure_test_dialog)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)

        self.root.config(menu=menubar)

    # ═══════════════════════════════════════════════════════
    # Layout
    # ═══════════════════════════════════════════════════════

    def _build_layout(self):
        # Top-level horizontal split
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ── LEFT PANEL ──────────────────────────────────────
        left_outer = ttk.Frame(main_pane, width=420)
        main_pane.add(left_outer, weight=0)

        canvas = tk.Canvas(left_outer, width=400)
        scrollbar = ttk.Scrollbar(left_outer, orient="vertical", command=canvas.yview)
        left_scroll = ttk.Frame(canvas)

        left_scroll.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=left_scroll, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        left = left_scroll

        # ── Vehicle Info ────────────────────────────────────
        vf = ttk.LabelFrame(left, text=" Vehicle Parameters ", padding=8)
        vf.pack(fill=tk.X, padx=4, pady=(4, 2))

        ttk.Label(vf, text=f"Dimensions (L × B × H):  "
                       f"{self.vehicle.L:.0f} × {self.vehicle.B:.0f} × {self.vehicle.H:.0f} cm",
                  font=FONT_LABEL).pack(anchor=tk.W)
        ttk.Label(vf, text=f"Center of Mass:  (0, 0, 0)  [geometric center]",
                  font=FONT_LABEL).pack(anchor=tk.W)
        ttk.Label(vf, text=f"Body:  Uniform density", font=FONT_LABEL).pack(anchor=tk.W)

        # ── Thruster Count ──────────────────────────────────
        tf = ttk.LabelFrame(left, text=" Number of Thrusters ", padding=8)
        tf.pack(fill=tk.X, padx=4, pady=2)

        self.thruster_lbl = ttk.Label(tf, text=f"Thrusters: {self.n_thrusters_var.get()}",
                                      font=FONT_HEADER)
        self.thruster_lbl.pack(anchor=tk.W)

        thruster_scale = ttk.Scale(
            tf, from_=6, to=12, variable=self.n_thrusters_var,
            orient=tk.HORIZONTAL, command=self._on_thruster_change
        )
        thruster_scale.pack(fill=tk.X, pady=4)

        tick_frame = ttk.Frame(tf)
        tick_frame.pack(fill=tk.X)
        for v in range(6, 13):
            ttk.Label(tick_frame, text=str(v), font=("Helvetica", 9)).pack(
                side=tk.LEFT, expand=True
            )

        # ── DOF Set Count ───────────────────────────────────
        sf = ttk.LabelFrame(left, text=" DOF Set Configuration ", padding=8)
        sf.pack(fill=tk.X, padx=4, pady=2)

        ttk.Label(sf, text="Number of thruster sets:", font=FONT_LABEL).pack(anchor=tk.W)
        radio_frame = ttk.Frame(sf)
        radio_frame.pack(fill=tk.X, pady=4)

        set_options = [
            (2, "2 Sets (31)"), (3, "3 Sets (90)"),
            (4, "4 Sets (65)"), (5, "5 Sets (15)"), (6, "6 Sets (1)")
        ]
        for val, label in set_options:
            ttk.Radiobutton(
                radio_frame, text=label, variable=self.n_sets_var,
                value=val, command=self._populate_partitions
            ).pack(side=tk.LEFT, padx=4)

        # ── Common Presets ──────────────────────────────────
        pf2 = ttk.LabelFrame(left, text=" Quick Presets (common AUV configs) ", padding=8)
        pf2.pack(fill=tk.X, padx=4, pady=2)

        self.presets = [
            ("Horizontal / Vertical",
             ((0, 1, 5), (2, 3, 4)),
             "Surge+Sway+Yaw | Heave+Roll+Pitch"),
            ("Translation / Rotation",
             ((0, 1, 2), (3, 4, 5)),
             "Surge+Sway+Heave | Roll+Pitch+Yaw"),
            ("Planar / Lateral",
             ((0, 2, 4), (1, 3, 5)),
             "Surge+Heave+Pitch | Sway+Roll+Yaw"),
            ("Heave+Yaw / Rest",
             ((2, 5), (0, 1, 3, 4)),
             "Heave+Yaw | Surge+Sway+Roll+Pitch"),
            ("Surge+Yaw / Rest",
             ((0, 5), (1, 2, 3, 4)),
             "Surge+Yaw | Sway+Heave+Roll+Pitch"),
            ("Heave+Roll+Pitch / Surge+Sway+Yaw",
             ((2, 3, 4), (0, 1, 5)),
             "Depth control | Planar motion"),
            ("Each DOF separate",
             ((0,), (1,), (2,), (3,), (4,), (5,)),
             "X | Y | Z | K | M | N"),
        ]

        for name, partition, desc in self.presets:
            btn = ttk.Button(
                pf2, text=f"{name}",
                command=lambda p=partition, d=desc: self._apply_preset(p, d)
            )
            btn.pack(fill=tk.X, pady=1)

        # ── DOF Combinations ───────────────────────────────
        cf = ttk.LabelFrame(left, text=" DOF Combinations ", padding=8)
        cf.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        # Filter
        filter_frame = ttk.Frame(cf)
        filter_frame.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(filter_frame, text="Filter:", font=FONT_LABEL).pack(side=tk.LEFT)
        filter_entry = ttk.Entry(
            filter_frame, textvariable=self.filter_var, font=FONT_MONO_SM
        )
        filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.filter_var.trace_add('write', lambda *a: self._apply_filter())

        # Listbox
        list_frame = ttk.Frame(cf)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.partition_list = tk.Listbox(
            list_frame, selectmode=tk.SINGLE, font=FONT_MONO_SM,
            height=12, activestyle='none'
        )
        p_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                 command=self.partition_list.yview)
        self.partition_list.configure(yscrollcommand=p_scroll.set)
        self.partition_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        p_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.combo_count_lbl = ttk.Label(cf, text="", font=FONT_LABEL)
        self.combo_count_lbl.pack(anchor=tk.W, pady=(4, 0))

        # Highlight commonly requested partitions
        ttk.Label(cf, text="Tip: 'yaw heave' finds both in same set",
                  font=("Helvetica", 9, "italic")).pack(anchor=tk.W)

        # ── Runtime Settings ────────────────────────────────
        rf = ttk.LabelFrame(left, text=" Optimization Runtime ", padding=8)
        rf.pack(fill=tk.X, padx=4, pady=2)

        ttk.Label(rf, text="Computation time budget:", font=FONT_LABEL).pack(anchor=tk.W)

        runtime_frame = ttk.Frame(rf)
        runtime_frame.pack(fill=tk.X, pady=4)

        runtime_values = [
            ("1 min", "1"), ("5 min", "5"), ("15 min", "15"),
            ("30 min", "30"), ("1 hour", "60")
        ]
        for text, val in runtime_values:
            ttk.Radiobutton(
                runtime_frame, text=text, variable=self.runtime_var, value=val
            ).pack(side=tk.LEFT, padx=6)

        # ── Action Buttons ──────────────────────────────────
        bf = ttk.LabelFrame(left, text=" Actions ", padding=8)
        bf.pack(fill=tk.X, padx=4, pady=2)

        self.quick_btn = ttk.Button(
            bf, text=">> QUICK ANALYSIS", command=self._quick_analysis
        )
        self.quick_btn.pack(fill=tk.X, pady=2)

        self.opt_btn = ttk.Button(
            bf, text=">> RUN OPTIMIZER", command=self._run_optimizer
        )
        self.opt_btn.pack(fill=tk.X, pady=2)

        self.stop_btn = ttk.Button(
            bf, text=">> STOP", command=self._stop_optimizer, state='disabled'
        )
        self.stop_btn.pack(fill=tk.X, pady=2)

        self.failure_btn = ttk.Button(
            bf, text=">> FAILURE TEST", command=self._failure_test_dialog
        )
        self.failure_btn.pack(fill=tk.X, pady=2)

        self.export_btn = ttk.Button(
            bf, text=">> EXPORT PYTHON CODE", command=self._export_python
        )
        self.export_btn.pack(fill=tk.X, pady=2)

        self.viz_btn = ttk.Button(
            bf, text=">> VISUALIZE 3D", command=self._visualize
        )
        self.viz_btn.pack(fill=tk.X, pady=2)

        # ── Progress ────────────────────────────────────────
        pf = ttk.Frame(left, padding=4)
        pf.pack(fill=tk.X, padx=4, pady=4)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            pf, variable=self.progress_var, maximum=100, length=380
        )
        self.progress_bar.pack(fill=tk.X, pady=2)

        self.status_lbl = ttk.Label(pf, text="Status: Ready", font=FONT_LABEL)
        self.status_lbl.pack(anchor=tk.W)

        # ── RIGHT PANEL (Results) ───────────────────────────
        right = ttk.LabelFrame(main_pane, text=" Analysis Results ", padding=8)
        main_pane.add(right, weight=1)

        # Results text area with both scrollbars
        txt_frame = ttk.Frame(right)
        txt_frame.pack(fill=tk.BOTH, expand=True)

        self.results_txt = tk.Text(
            txt_frame, wrap=tk.NONE, font=FONT_MONO,
            bg="#1e1e2e", fg="#cdd6f4", insertbackground="#cdd6f4",
            selectbackground="#45475a", selectforeground="#cdd6f4",
            relief=tk.FLAT, padx=10, pady=10
        )
        y_scroll = ttk.Scrollbar(txt_frame, orient=tk.VERTICAL,
                                 command=self.results_txt.yview)
        x_scroll = ttk.Scrollbar(txt_frame, orient=tk.HORIZONTAL,
                                 command=self.results_txt.xview)
        self.results_txt.configure(
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set
        )

        self.results_txt.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        txt_frame.grid_rowconfigure(0, weight=1)
        txt_frame.grid_columnconfigure(0, weight=1)

    # ═══════════════════════════════════════════════════════
    # Partition Management
    # ═══════════════════════════════════════════════════════

    def _on_thruster_change(self, *_):
        val = int(round(self.n_thrusters_var.get()))
        self.n_thrusters_var.set(val)
        self.thruster_lbl.config(text=f"Thrusters: {val}")

    def _populate_partitions(self):
        """Regenerate the partition list for current set count."""
        n_sets = self.n_sets_var.get()
        self.partitions = generate_dof_partitions(n_sets)
        self._apply_filter()

    def _apply_filter(self):
        """Filter the partition list by text. Multi-word queries match DOFs in the same group."""
        flt = self.filter_var.get().strip().lower()
        self.partition_list.delete(0, tk.END)

        # Parse filter words
        filter_words = [w.strip().strip('+') for w in flt.replace('+', ' ').split() if w.strip()]

        visible = 0
        for i, p in enumerate(self.partitions):
            desc = partition_to_string(p)
            if filter_words:
                # Check if ALL filter words appear in the SAME group
                matched = False
                for group in p:
                    group_names = [DOF_LABELS[d].lower() for d in group]
                    group_syms = [DOF_SYMBOLS[d].lower() for d in group]
                    group_text = ' '.join(group_names + group_syms)
                    if all(w in group_text for w in filter_words):
                        matched = True
                        break
                if not matched:
                    continue
            self.partition_list.insert(tk.END, f"{i + 1:3d}. {desc}")
            visible += 1

        self.combo_count_lbl.config(
            text=f"Showing {visible} / {len(self.partitions)} combinations"
        )

        if visible > 0:
            self.partition_list.selection_set(0)

    def _apply_preset(self, partition, description):
        """Apply a preset DOF partition directly and run quick analysis."""
        n_sets = len(partition)
        self.n_sets_var.set(n_sets)
        self.partitions = generate_dof_partitions(n_sets)

        # Find matching partition in the generated list
        target_sets = [set(g) for g in partition]
        matched_idx = None
        for i, p in enumerate(self.partitions):
            p_sets = [set(g) for g in p]
            if len(p_sets) == len(target_sets):
                # Check if same sets (order-independent)
                if all(any(ps == ts for ps in p_sets) for ts in target_sets):
                    matched_idx = i
                    break

        self.filter_var.set("")
        self._apply_filter()

        if matched_idx is not None:
            # Find the item in the listbox and select it
            for j in range(self.partition_list.size()):
                text = self.partition_list.get(j)
                idx = int(text.split('.')[0].strip()) - 1
                if idx == matched_idx:
                    self.partition_list.selection_clear(0, tk.END)
                    self.partition_list.selection_set(j)
                    self.partition_list.see(j)
                    break

        self.status_lbl.config(text=f"Status: Preset loaded — {description}")
        self._quick_analysis()

    def _get_selected_partition(self):
        """Return the currently selected DOF partition, or None."""
        sel = self.partition_list.curselection()
        if not sel:
            messagebox.showinfo("Selection Required",
                                "Please select a DOF combination from the list.")
            return None
        # Parse index from the display string  "  3. Set A: Surge + ..."
        text = self.partition_list.get(sel[0])
        idx = int(text.split('.')[0].strip()) - 1
        return self.partitions[idx]

    # ═══════════════════════════════════════════════════════
    # Quick Analysis
    # ═══════════════════════════════════════════════════════

    def _quick_analysis(self):
        """Run instant analysis with default thruster placement."""
        partition = self._get_selected_partition()
        n = int(round(self.n_thrusters_var.get()))

        thrusters = default_thruster_config(n, self.vehicle)

        # Distribute thrusters across sets (simple round-robin)
        assignment = None
        if partition:
            n_sets = len(partition)
            assignment = []
            per_set = n // n_sets
            rem = n % n_sets
            for s in range(n_sets):
                count = per_set + (1 if s < rem else 0)
                assignment.extend([s] * count)

        self.current_thrusters = thrusters
        self.current_assignment = assignment

        report = full_analysis_report(
            self.vehicle, thrusters, partition, assignment
        )
        self._display_report(report)
        self.status_lbl.config(text="Status: Quick analysis complete")

    # ═══════════════════════════════════════════════════════
    # Optimizer
    # ═══════════════════════════════════════════════════════

    def _run_optimizer(self):
        partition = self._get_selected_partition()
        if not partition:
            return

        n = int(round(self.n_thrusters_var.get()))
        runtime_sec = int(self.runtime_var.get()) * 60

        self.optimizer = ThrusterOptimizer(
            self.vehicle, n, partition, runtime_sec
        )

        # Disable controls
        self.quick_btn.config(state='disabled')
        self.opt_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.failure_btn.config(state='disabled')
        self.progress_var.set(0)
        self.status_lbl.config(text="Status: Optimizing...")

        def _run():
            def _callback(progress, iterations, best_score):
                self.root.after(0, lambda: self._update_progress(
                    progress, iterations, best_score
                ))

            result = self.optimizer.optimize(callback=_callback)

            report = full_analysis_report(
                self.vehicle, result['thrusters'],
                partition, result['assignment']
            )
            report += "\n"
            report += "+" + "=" * 62 + "+\n"
            report += "|           OPTIMIZATION RESULTS                               |\n"
            report += "+" + "=" * 62 + "+\n"
            report += f"    Iterations:  {result['iterations']:,}\n"
            report += f"    Elapsed:     {result['elapsed']:.1f} seconds\n"
            report += f"    Best Score:  {result['score']:.2f}\n"
            report += f"    Runtime:     {int(self.runtime_var.get())} min\n"

            self.current_thrusters = result['thrusters']
            self.current_assignment = result['assignment']

            self.root.after(0, lambda: self._display_report(report))
            self.root.after(0, self._optimizer_done)

        self.optimizer_thread = threading.Thread(target=_run, daemon=True)
        self.optimizer_thread.start()

    def _update_progress(self, progress, iterations, best_score):
        self.progress_var.set(progress * 100)
        self.status_lbl.config(
            text=f"Status: {iterations:,} iters | score: {best_score:.1f} | "
                 f"{progress * 100:.0f}%"
        )

    def _optimizer_done(self):
        self.quick_btn.config(state='normal')
        self.opt_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.failure_btn.config(state='normal')
        self.progress_var.set(100)
        self.status_lbl.config(text="Status: Optimization complete")

    def _stop_optimizer(self):
        if self.optimizer:
            self.optimizer.stop()
        self.status_lbl.config(text="Status: Stopping...")

    # ═══════════════════════════════════════════════════════
    # Visualize
    # ═══════════════════════════════════════════════════════

    def _visualize(self):
        """Open a matplotlib 3D dashboard for the current configuration."""
        if self.current_thrusters is None:
            # No analysis run yet — generate from current settings
            self._quick_analysis()

        if self.current_thrusters is None:
            return

        n = len(self.current_thrusters)
        title = f'{n}-Thruster Configuration'
        if self.current_assignment is not None:
            n_sets = len(set(self.current_assignment))
            title += f' ({n_sets} sets)'

        self.status_lbl.config(text="Status: Visualization opened")

        # Must run on main thread — macOS requires NSWindow on main thread
        visualize_config(
            self.vehicle,
            self.current_thrusters,
            self.current_assignment,
            title
        )

    # ═══════════════════════════════════════════════════════
    # Failure Test
    # ═══════════════════════════════════════════════════════

    def _failure_test_dialog(self):
        """Dialog to select which thrusters have failed."""
        n = int(round(self.n_thrusters_var.get()))
        thrusters = default_thruster_config(n, self.vehicle)
        am = AllocationMatrix(self.vehicle, thrusters)

        dialog = tk.Toplevel(self.root)
        dialog.title("Thruster Failure Simulation")
        dialog.geometry("450x400")
        dialog.transient(self.root)

        ttk.Label(dialog, text="Select failed thrusters:", font=FONT_HEADER).pack(
            anchor=tk.W, padx=10, pady=10
        )

        check_vars = []
        for i, t in enumerate(thrusters):
            var = tk.BooleanVar(value=False)
            check_vars.append(var)
            ttk.Checkbutton(
                dialog, text=f"{t.name}  pos=({t.position[0]:.1f}, "
                             f"{t.position[1]:.1f}, {t.position[2]:.1f})",
                variable=var
            ).pack(anchor=tk.W, padx=20)

        def _run_test():
            failed = [i for i, v in enumerate(check_vars) if v.get()]
            if not failed:
                messagebox.showinfo("No Selection",
                                    "Select at least one thruster to fail.")
                return

            result = am.failure_analysis(failed)
            failed_names = [thrusters[i].name for i in failed]

            report = []
            report.append("+" + "=" * 62 + "+")
            report.append("|           THRUSTER FAILURE ANALYSIS                          |")
            report.append("+" + "=" * 62 + "+")
            report.append("")
            report.append(f"    Failed thrusters: {failed_names}")
            report.append(f"    Remaining:        {result['remaining_count']} / {n}")
            report.append(f"    Matrix rank:      {result['matrix_rank']} / 6")
            fa = "YES" if result['fully_actuated'] else "NO"
            report.append(f"    Fully actuated:   [{fa}]")
            report.append("")
            report.append("    Per-DOF Status:")
            report.append(f"    {'DOF':>8s}  {'Retention':>10s}  {'Status':>10s}")
            report.append("    " + "-" * 35)

            for dof, info in result['dof_status'].items():
                bar_len = int(info['retention'] / 5)
                bar = '#' * bar_len + '.' * (20 - bar_len)
                report.append(
                    f"    {dof:>8s}  [{bar}] {info['retention']:5.1f}%  "
                    f"{info['status']}"
                )

            report.append("")
            report.append("    Interpretation:")
            for dof, info in result['dof_status'].items():
                if info['status'] == 'LOST':
                    report.append(f"      [!!] {dof}: COMPLETELY LOST — no authority")
                elif info['status'] == 'CRITICAL':
                    report.append(f"      [!!] {dof}: CRITICAL — <30% authority remaining")
                elif info['status'] == 'DEGRADED':
                    report.append(f"      [!]  {dof}: DEGRADED — reduced but functional")

            self._display_report('\n'.join(report))
            dialog.destroy()

        ttk.Button(dialog, text="Run Failure Test", command=_run_test).pack(
            pady=15
        )

    # ═══════════════════════════════════════════════════════
    # Export
    # ═══════════════════════════════════════════════════════

    def _export_python(self):
        """Export mixing matrix as Python code."""
        partition = self._get_selected_partition()
        n = int(round(self.n_thrusters_var.get()))
        thrusters = default_thruster_config(n, self.vehicle)

        assignment = None
        if partition:
            n_sets = len(partition)
            assignment = []
            per_set = n // n_sets
            rem = n % n_sets
            for s in range(n_sets):
                count = per_set + (1 if s < rem else 0)
                assignment.extend([s] * count)

        code = generate_python_code(self.vehicle, thrusters, partition, assignment)

        filepath = filedialog.asksaveasfilename(
            title="Save Python Code",
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            initialfile="thruster_mixer.py"
        )
        if filepath:
            with open(filepath, 'w') as f:
                f.write(code)
            self.status_lbl.config(text=f"Status: Code exported to {filepath}")

    def _export_report(self):
        """Export the current report to a text file."""
        content = self.results_txt.get('1.0', tk.END)
        filepath = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile="thruster_report.txt"
        )
        if filepath:
            with open(filepath, 'w') as f:
                f.write(content)
            self.status_lbl.config(text=f"Status: Report exported to {filepath}")

    # ═══════════════════════════════════════════════════════
    # Display
    # ═══════════════════════════════════════════════════════

    def _display_report(self, text):
        """Update the results text area."""
        self.results_txt.config(state=tk.NORMAL)
        self.results_txt.delete('1.0', tk.END)
        self.results_txt.insert('1.0', text)

        # Syntax highlighting for headers
        self._highlight_report()

    def _highlight_report(self):
        """Apply simple color tags to the report text."""
        self.results_txt.tag_configure("header", foreground=FG_ACCENT,
                                       font=("Menlo", 12, "bold"))
        self.results_txt.tag_configure("ok", foreground=FG_GREEN)
        self.results_txt.tag_configure("warn", foreground=FG_YELLOW)
        self.results_txt.tag_configure("error", foreground=FG_RED)
        self.results_txt.tag_configure("dim", foreground=FG_DIM)

        content = self.results_txt.get('1.0', tk.END)
        lines = content.split('\n')

        for i, line in enumerate(lines):
            line_start = f"{i + 1}.0"
            line_end = f"{i + 1}.end"

            if line.startswith(">>>"):
                self.results_txt.tag_add("header", line_start, line_end)
            elif "[OK]" in line or "[YES]" in line or "YES" in line:
                self.results_txt.tag_add("ok", line_start, line_end)
            elif "[!!]" in line or "WARNING" in line or "LOST" in line or "CRITICAL" in line:
                self.results_txt.tag_add("error", line_start, line_end)
            elif "WEAKLY" in line or "DEGRADED" in line or "[!]" in line:
                self.results_txt.tag_add("warn", line_start, line_end)
            elif line.startswith("=") or line.startswith("+") or line.startswith("|"):
                self.results_txt.tag_add("dim", line_start, line_end)


# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    # Set default theme
    style = ttk.Style()
    available_themes = style.theme_names()
    if 'aqua' in available_themes:
        style.theme_use('aqua')
    elif 'clam' in available_themes:
        style.theme_use('clam')

    app = ThrusterExplorerApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
