"""
anm.py

Minimal anisotropic network model implementation.

The module operates only on Cartesian coordinates. PDB parsing and writing
are handled separately in pdb_io.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ANMResult:
    """Container for ANM eigenvalues and eigenvectors."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    hessian: np.ndarray


def build_hessian(
    coordinates: np.ndarray,
    cutoff: float = 15.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Build the ANM Hessian matrix.

    Parameters
    ----------
    coordinates:
        Coordinate array with shape (n_atoms, 3).
    cutoff:
        Interaction cutoff distance in Angstrom.
    gamma:
        Uniform spring constant.

    Returns
    -------
    np.ndarray
        Hessian matrix with shape (3 * n_atoms, 3 * n_atoms).
    """

    coordinates = np.asarray(coordinates, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (n_atoms, 3).")

    n_atoms = coordinates.shape[0]

    if n_atoms < 2:
        raise ValueError("At least two atoms are required to build an ANM.")

    if cutoff <= 0:
        raise ValueError("cutoff must be positive.")

    if gamma <= 0:
        raise ValueError("gamma must be positive.")

    hessian = np.zeros((3 * n_atoms, 3 * n_atoms), dtype=float)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            rij = coordinates[j] - coordinates[i]
            distance = np.linalg.norm(rij)

            if distance == 0.0:
                raise ValueError(f"Atoms {i} and {j} have identical coordinates.")

            if distance > cutoff:
                continue

            unit = rij / distance
            block = -gamma * np.outer(unit, unit)

            i_slice = slice(3 * i, 3 * i + 3)
            j_slice = slice(3 * j, 3 * j + 3)

            hessian[i_slice, j_slice] = block
            hessian[j_slice, i_slice] = block

            hessian[i_slice, i_slice] -= block
            hessian[j_slice, j_slice] -= block

    return hessian


def diagonalize_hessian(
    hessian: np.ndarray,
    zero_mode_tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize a Hessian matrix.

    Parameters
    ----------
    hessian:
        Hessian matrix with shape (3 * n_atoms, 3 * n_atoms).
    zero_mode_tolerance:
        Eigenvalues below this value are considered near-zero modes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Eigenvalues and eigenvectors sorted in ascending eigenvalue order.
    """

    hessian = np.asarray(hessian, dtype=float)

    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be a square matrix.")

    if hessian.shape[0] % 3 != 0:
        raise ValueError("hessian dimension must be divisible by 3.")

    eigenvalues, eigenvectors = np.linalg.eigh(hessian)

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Remove tiny negative values caused by numerical precision.
    eigenvalues[np.abs(eigenvalues) < zero_mode_tolerance] = 0.0

    return eigenvalues, eigenvectors


def calculate_anm(
    coordinates: np.ndarray,
    cutoff: float = 15.0,
    gamma: float = 1.0,
) -> ANMResult:
    """
    Build and diagonalize the ANM Hessian.

    Parameters
    ----------
    coordinates:
        Coordinate array with shape (n_atoms, 3).
    cutoff:
        Interaction cutoff distance in Angstrom.
    gamma:
        Uniform spring constant.

    Returns
    -------
    ANMResult
        Hessian, eigenvalues, and eigenvectors.
    """

    hessian = build_hessian(coordinates, cutoff=cutoff, gamma=gamma)
    eigenvalues, eigenvectors = diagonalize_hessian(hessian)

    return ANMResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        hessian=hessian,
    )


def get_nonzero_modes(
    result: ANMResult,
    n_modes: int = 3,
    n_zero_modes: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the first non-rigid-body ANM modes.

    For a connected 3D elastic network, the first six modes correspond to
    global translations and rotations.

    Parameters
    ----------
    result:
        ANMResult from calculate_anm.
    n_modes:
        Number of non-zero modes to return.
    n_zero_modes:
        Number of rigid-body modes to skip. Usually 6.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Selected eigenvalues and eigenvectors.

        Eigenvectors are returned with shape (n_modes, n_atoms, 3).
    """

    if n_modes <= 0:
        raise ValueError("n_modes must be positive.")

    start = n_zero_modes
    stop = start + n_modes

    if stop > result.eigenvectors.shape[1]:
        raise ValueError(
            f"Requested modes up to index {stop}, but only "
            f"{result.eigenvectors.shape[1]} modes are available."
        )

    selected_values = result.eigenvalues[start:stop]
    selected_vectors = result.eigenvectors[:, start:stop].T

    n_atoms = result.eigenvectors.shape[0] // 3
    selected_vectors = selected_vectors.reshape(n_modes, n_atoms, 3)

    return selected_values, selected_vectors


def count_zero_modes(
    eigenvalues: np.ndarray,
    tolerance: float = 1e-6,
) -> int:
    """Count near-zero eigenvalues."""

    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return int(np.sum(np.abs(eigenvalues) < tolerance))


def normalize_mode(mode: np.ndarray) -> np.ndarray:
    """
    Normalize a mode vector.

    Parameters
    ----------
    mode:
        Mode with shape (n_atoms, 3).

    Returns
    -------
    np.ndarray
        Normalized mode with Euclidean norm 1.
    """

    mode = np.asarray(mode, dtype=float)

    if mode.ndim != 2 or mode.shape[1] != 3:
        raise ValueError("mode must have shape (n_atoms, 3).")

    norm = np.linalg.norm(mode)

    if norm == 0:
        raise ValueError("Cannot normalize a zero mode.")

    return mode / norm