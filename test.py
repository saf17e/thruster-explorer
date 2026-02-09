"""
AUV Thruster Explorer — Test Suite
====================================
Run all engine tests to verify correctness.

Usage:
    python test.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import (
    Vehicle, Thruster, AllocationMatrix,
    default_thruster_config, generate_dof_partitions,
    partition_to_string, ThrusterOptimizer,
    full_analysis_report, generate_python_code,
    DOF_LABELS,
)


def test_vehicle():
    v = Vehicle(60, 40, 20)
    assert v.L == 60
    assert v.B == 40
    assert v.H == 20
    assert np.allclose(v.com, [0, 0, 0])
    assert v.bounds() == (30.0, 20.0, 10.0)
    print("  [PASS] Vehicle model")


def test_thruster():
    t = Thruster([10, 5, 0], [1, 0, 0], name="T1")
    assert np.allclose(t.orientation, [1, 0, 0])
    f = t.force_vector(0.5)
    assert np.allclose(f, [0.5, 0, 0])

    # Torque check: r=(10,5,0), F=(1,0,0) → τ = r×F = (0*0-0*0, 0*1-10*0, 10*0-5*1) = (0,0,-5)
    tau = t.torque_vector()
    assert np.allclose(tau, [0, 0, -5])
    print("  [PASS] Thruster model")


def test_allocation_matrix():
    v = Vehicle(60, 40, 20)
    for n in range(6, 13):
        thrusters = default_thruster_config(n, v)
        am = AllocationMatrix(v, thrusters)
        assert am.B.shape == (6, n), f"Wrong shape for n={n}"
        assert am.rank() >= 5, f"Rank too low for n={n}: {am.rank()}"
    print("  [PASS] Allocation matrix for all thruster counts (6-12)")


def test_rank_fullactuated():
    v = Vehicle(60, 40, 20)
    # 8-thruster config should be fully actuated
    t8 = default_thruster_config(8, v)
    am = AllocationMatrix(v, t8)
    assert am.rank() == 6, f"8-thruster should be rank 6, got {am.rank()}"
    assert am.is_fully_actuated()
    print("  [PASS] 8-thruster fully actuated (rank 6)")


def test_efficiency_scores():
    v = Vehicle(60, 40, 20)
    t = default_thruster_config(8, v)
    am = AllocationMatrix(v, t)
    scores = am.efficiency_scores()
    assert len(scores) == 6
    for name in DOF_LABELS:
        assert name in scores
        assert 0 <= scores[name] <= 100
    print("  [PASS] Efficiency scores")


def test_partition_generation():
    p2 = generate_dof_partitions(2)
    p3 = generate_dof_partitions(3)
    assert len(p2) == 31, f"Expected 31 two-set partitions, got {len(p2)}"
    assert len(p3) == 90, f"Expected 90 three-set partitions, got {len(p3)}"

    # Verify all partitions cover all 6 DOFs
    for p in p2:
        all_dofs = sorted([d for group in p for d in group])
        assert all_dofs == [0, 1, 2, 3, 4, 5], f"Incomplete partition: {p}"

    for p in p3:
        all_dofs = sorted([d for group in p for d in group])
        assert all_dofs == [0, 1, 2, 3, 4, 5], f"Incomplete partition: {p}"

    # Verify Surge+Yaw partition exists
    target = ({0, 5}, {1, 2, 3, 4})
    found = False
    for p in p2:
        sets = [set(g) for g in p]
        if sets[0] == target[0] and sets[1] == target[1]:
            found = True
        elif sets[1] == target[0] and sets[0] == target[1]:
            found = True
    assert found, "Surge+Yaw | rest partition not found"

    print("  [PASS] DOF partitions (2-set=31, 3-set=90)")


def test_subset_analysis():
    v = Vehicle(60, 40, 20)
    t = default_thruster_config(8, v)
    am = AllocationMatrix(v, t)

    # Set A: thrusters 0-3 (horizontal), DOFs: Surge(0), Sway(1), Yaw(5)
    analysis = am.subset_analysis([0, 1, 2, 3], [0, 1, 5])
    assert analysis['rank'] == analysis['required_rank'] == 3
    assert analysis['controllable']
    print("  [PASS] Subset analysis")


def test_failure_analysis():
    v = Vehicle(60, 40, 20)
    t = default_thruster_config(8, v)
    am = AllocationMatrix(v, t)

    # Kill one thruster
    result = am.failure_analysis([0])
    assert result['remaining_count'] == 7
    assert result['matrix_rank'] >= 5
    for name in DOF_LABELS:
        assert name in result['dof_status']

    print("  [PASS] Failure analysis")


def test_coupling_matrix():
    v = Vehicle(60, 40, 20)
    t = default_thruster_config(8, v)
    am = AllocationMatrix(v, t)

    C = am.coupling_matrix()
    assert C.shape == (6, 6)
    # Diagonal should be close to 1 for fully actuated system
    for i in range(6):
        assert abs(C[i, i] - 1.0) < 0.5, f"Diagonal C[{i},{i}]={C[i,i]:.3f}"

    print("  [PASS] Coupling matrix")


def test_full_report():
    v = Vehicle(60, 40, 20)
    t = default_thruster_config(8, v)
    partition = ((0, 5), (1, 2, 3, 4))
    assignment = [0, 0, 0, 0, 1, 1, 1, 1]

    report = full_analysis_report(v, t, partition, assignment)
    assert len(report) > 1000
    assert "ALLOCATION MATRIX" in report
    assert "EFFICIENCY" in report
    assert "COUPLING" in report
    assert "DOF SET ANALYSIS" in report

    print("  [PASS] Full analysis report generation")


def test_python_export():
    v = Vehicle(60, 40, 20)
    t = default_thruster_config(8, v)
    code = generate_python_code(v, t)
    assert "class ThrusterMixer" in code
    assert "self.B = np.array" in code
    assert "def allocate" in code

    print("  [PASS] Python code export")


def test_optimizer_quick():
    """Quick optimizer test (2 seconds)."""
    v = Vehicle(60, 40, 20)
    partition = ((0, 5), (1, 2, 3, 4))

    opt = ThrusterOptimizer(v, 8, partition, runtime_seconds=2)
    result = opt.optimize()

    assert result['iterations'] > 0
    assert result['elapsed'] >= 1.5
    assert len(result['thrusters']) == 8
    assert len(result['assignment']) == 8

    print(f"  [PASS] Optimizer ({result['iterations']} iters in {result['elapsed']:.1f}s)")


if __name__ == '__main__':
    print("=" * 56)
    print("  AUV Thruster Explorer — Engine Test Suite")
    print("=" * 56)
    print()

    tests = [
        test_vehicle,
        test_thruster,
        test_allocation_matrix,
        test_rank_fullactuated,
        test_efficiency_scores,
        test_partition_generation,
        test_subset_analysis,
        test_failure_analysis,
        test_coupling_matrix,
        test_full_report,
        test_python_export,
        test_optimizer_quick,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1

    print()
    print("-" * 56)
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("-" * 56)

    if failed == 0:
        print("  ALL TESTS PASSED")
    else:
        print(f"  {failed} TEST(S) FAILED")
        sys.exit(1)
