import numpy as np
from typing import List

from spins.gridlock import Direction


def build_dipole_source(omega: complex, dxes: List[np.ndarray],
                        eps: List[np.ndarray], position: List[int],
                        axis: Direction or int, power: float,
                        phase: complex) -> List[np.ndarray]:
    """Builds a dipole source.

    Args:
        omega: The frequency of the mode.
        dxes: List of cell widths.
        eps: Permittivity distribution.
        position: Permittivity
        axis: Direction of propagation.
        power: Power emitted by the source.
        phase: Complex number used for phase.

    Returns:
        Current source J.
    """
    position = [int(x) for x in position]
    dx = dxes[0][int(axis)][position[int(axis)]]
    eps_source = eps[int(axis)][tuple(position)]

    # The 0.0266 constant is an empirically determined constant.
    j_norm = np.sqrt(power) / np.sqrt(0.02661928208866846 * (omega**2 * dx**3))

    J = np.zeros_like(eps)
    J[int(axis)][tuple(position)] = 1 / eps_source * j_norm * phase / abs(phase) 
    return J

def build_dipole_source_off_axis(omega: complex, dxes: List[np.ndarray],
                        eps: List[np.ndarray], position: List[int],
                        axis: List[float], power: float,
                        phase: List[complex]) -> List[np.ndarray]:
    """Builds a dipole source.

    Args:
        omega: The frequency of the mode.
        dxes: List of cell widths.
        eps: Permittivity distribution.
        position: Permittivity
        axis: Amplitudes of dipole at directions [x, y, z]..
        power: Power emitted by the source.
        phase: Complex number used for phase.

    Returns:
        Current source J.
    """

    J = np.zeros_like(eps)
    for ax, amp, phi in zip([0, 1, 2], axis, phase):
        position = [int(x) for x in position]
            
        dx = dxes[0][int(ax)][position[int(ax)]]
        eps_source = eps[int(ax)][tuple(position)]
    
        # The 0.0266 constant is an empirically determined constant.
        j_norm = np.sqrt(power) / np.sqrt(0.02661928208866846 * (omega**2 * dx**3))        
        
        J[int(ax)][tuple(position)] = amp / eps_source * j_norm * phi / abs(phi)
    return J