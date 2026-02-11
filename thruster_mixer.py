"""
Auto-generated AUV Thruster Mixing Matrix
Vehicle: 60x40x20 cm
Thrusters: 12
"""

import numpy as np


class ThrusterMixer:
    """Mixing matrix for 12-thruster AUV configuration."""

    DOF_NAMES = ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw']

    def __init__(self):
        # 6×n Allocation Matrix (tau = B @ u)
        self.B = np.array([
            [  0.7071,   0.7071,  -0.7071,  -0.7071,   0.0000,   0.0000,   0.0000,   0.0000,   1.0000,   0.0000,   0.0000,   0.0000],  # Surge
            [  0.7071,  -0.7071,  -0.7071,   0.7071,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   1.0000,  -1.0000,   0.0000],  # Sway
            [  0.0000,   0.0000,   0.0000,   0.0000,   1.0000,   1.0000,   1.0000,   1.0000,   0.0000,   0.0000,   0.0000,   1.0000],  # Heave
            [  3.5355,  -3.5355,  -3.5355,   3.5355,  -9.0000,   9.0000,  -9.0000,   9.0000,   0.0000,  -0.0000,   0.0000,   0.0000],  # Roll
            [ -3.5355,  -3.5355,   3.5355,   3.5355, -10.5000, -10.5000,  10.5000,  10.5000,   0.0000,   0.0000,   0.0000,   0.0000],  # Pitch
            [ 25.4558, -25.4558,   4.2426,  -4.2426,   0.0000,   0.0000,   0.0000,  -0.0000,  -0.0000,   0.0000,  -0.0000,   0.0000]  # Yaw
        ])

        # Pseudo-inverse for control allocation
        self.B_pinv = np.linalg.pinv(self.B)

        # Thruster set assignments
        self.set_A_thrusters = [0, 1, 2, 3, 4, 5]  # ['Surge', 'Sway', 'Roll', 'Pitch']
        self.set_B_thrusters = [6, 7, 8, 9, 10, 11]  # ['Heave', 'Yaw']

    def allocate(self, tau_desired):
        """
        Compute thruster commands for desired force/torque vector.
        tau_desired: [Surge, Sway, Heave, Roll, Pitch, Yaw]
        Returns: thruster command vector u
        """
        tau = np.array(tau_desired, dtype=float)
        u = self.B_pinv @ tau
        return np.clip(u, -1.0, 1.0)


if __name__ == '__main__':
    mixer = ThrusterMixer()
    # Example: pure forward surge
    u = mixer.allocate([1, 0, 0, 0, 0, 0])
    print('Surge command:', u)