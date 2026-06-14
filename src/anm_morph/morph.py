"""
morph.py

Generate coordinate trajectories by displacing structures along ANM modes.
"""

from __future__ import annotations

import numpy as np


def validate_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Validate and return coordinates as a float array."""

    coordinates = np.asarray(coordinates, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (n_atoms, 3).")

    return coordinates


def validate_mode(mode: np.ndarray, n_atoms: int | None = None) -> np.ndarray:
    """Validate and return a mode as a float array."""

    mode = np.asarray(mode, dtype=float)

    if mode.ndim != 2 or mode.shape[1] != 3:
        raise ValueError("mode must have shape (n_atoms, 3).")

    if n_atoms is not None and mode.shape[0] != n_atoms:
        raise ValueError(
            f"mode has {mode.shape[0]} atoms, but expected {n_atoms}."
        )

    return mode


def rmsd(displacements: np.ndarray) -> float:
    """
    Calculate RMSD-like magnitude of a displacement vector.

    Parameters
    ----------
    displacements:
        Array with shape (n_atoms, 3).

    Returns
    -------
    float
        Root-mean-square displacement.
    """

    displacements = validate_coordinates(displacements)
    return float(np.sqrt(np.mean(np.sum(displacements**2, axis=1))))


def normalize_mode(mode: np.ndarray) -> np.ndarray:
    """
    Normalize a mode to unit Euclidean norm.

    Parameters
    ----------
    mode:
        Mode array with shape (n_atoms, 3).

    Returns
    -------
    np.ndarray
        Normalized mode.
    """

    mode = validate_mode(mode)
    norm = np.linalg.norm(mode)

    if norm == 0.0:
        raise ValueError("Cannot normalize a zero mode.")

    return mode / norm


def scale_mode_to_rmsd(
    mode: np.ndarray,
    target_rmsd: float,
) -> np.ndarray:
    """
    Scale a mode so its RMS displacement equals target_rmsd.

    Parameters
    ----------
    mode:
        Mode array with shape (n_atoms, 3).
    target_rmsd:
        Desired RMS displacement in Angstrom.

    Returns
    -------
    np.ndarray
        Scaled mode array.
    """

    if target_rmsd < 0:
        raise ValueError("target_rmsd must be non-negative.")

    mode = validate_mode(mode)
    current_rmsd = rmsd(mode)

    if current_rmsd == 0.0:
        raise ValueError("Cannot scale a mode with zero RMSD.")

    return mode * (target_rmsd / current_rmsd)


def displace_along_mode(
    coordinates: np.ndarray,
    mode: np.ndarray,
    rmsd_step: float,
) -> np.ndarray:
    """
    Displace coordinates along a mode by a given RMSD step.

    Parameters
    ----------
    coordinates:
        Reference coordinates with shape (n_atoms, 3).
    mode:
        Mode array with shape (n_atoms, 3).
    rmsd_step:
        Signed RMSD displacement in Angstrom.
        Positive and negative values move in opposite directions.

    Returns
    -------
    np.ndarray
        Displaced coordinates with shape (n_atoms, 3).
    """

    coordinates = validate_coordinates(coordinates)
    mode = validate_mode(mode, n_atoms=coordinates.shape[0])

    scaled_mode = scale_mode_to_rmsd(mode, abs(rmsd_step))

    if rmsd_step < 0:
        scaled_mode = -scaled_mode

    return coordinates + scaled_mode


def generate_mode_trajectory(
    coordinates: np.ndarray,
    mode: np.ndarray,
    max_rmsd: float = 2.0,
    n_frames: int = 21,
    both_directions: bool = True,
) -> np.ndarray:
    """
    Generate a coordinate trajectory along one mode.

    Parameters
    ----------
    coordinates:
        Reference coordinates with shape (n_atoms, 3).
    mode:
        Mode array with shape (n_atoms, 3).
    max_rmsd:
        Maximum RMSD displacement in Angstrom.
    n_frames:
        Number of frames to generate.
    both_directions:
        If True, generate frames from -max_rmsd to +max_rmsd.
        If False, generate frames from 0 to +max_rmsd.

    Returns
    -------
    np.ndarray
        Coordinate trajectory with shape (n_frames, n_atoms, 3).
    """

    coordinates = validate_coordinates(coordinates)
    mode = validate_mode(mode, n_atoms=coordinates.shape[0])

    if max_rmsd < 0:
        raise ValueError("max_rmsd must be non-negative.")

    if n_frames < 2:
        raise ValueError("n_frames must be at least 2.")

    if both_directions:
        rmsd_steps = np.linspace(-max_rmsd, max_rmsd, n_frames)
    else:
        rmsd_steps = np.linspace(0.0, max_rmsd, n_frames)

    coordsets = np.array(
        [
            displace_along_mode(
                coordinates,
                mode,
                rmsd_step=step,
            )
            for step in rmsd_steps
        ],
        dtype=float,
    )

    return coordsets


def combine_modes(
    modes: list[np.ndarray] | np.ndarray,
    weights: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    """
    Combine multiple modes into one displacement direction.

    Parameters
    ----------
    modes:
        Modes with shape (n_modes, n_atoms, 3).
    weights:
        Optional weights with shape (n_modes,). If None, all modes
        receive equal weight.

    Returns
    -------
    np.ndarray
        Combined normalized mode with shape (n_atoms, 3).
    """

    modes = np.asarray(modes, dtype=float)

    if modes.ndim != 3 or modes.shape[2] != 3:
        raise ValueError("modes must have shape (n_modes, n_atoms, 3).")

    n_modes = modes.shape[0]

    if weights is None:
        weights = np.ones(n_modes, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if weights.shape != (n_modes,):
        raise ValueError(
            f"weights must have shape {(n_modes,)}, got {weights.shape}."
        )

    combined = np.sum(modes * weights[:, np.newaxis, np.newaxis], axis=0)

    return normalize_mode(combined)