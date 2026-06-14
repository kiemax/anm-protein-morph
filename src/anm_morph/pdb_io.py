"""
pdb_io.py

Minimal PDB parsing and writing utilities for C-alpha based ANM/NMA.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class AtomRecord:
    """Container for one ATOM/HETATM record."""

    record_name: str
    serial: int
    atom_name: str
    alt_loc: str
    res_name: str
    chain_id: str
    res_id: int
    insertion_code: str
    coord: np.ndarray
    occupancy: float
    temp_factor: float
    element: str
    charge: str
    raw_line: str


def parse_pdb(
    pdb_path: str | Path,
    chain: str | None = None,
    atom_name: str | None = None,
    include_hetatm: bool = False,
) -> list[AtomRecord]:
    """
    Parse atoms from a PDB file.

    Parameters
    ----------
    pdb_path:
        Path to the input PDB file.
    chain:
        Optional chain identifier, e.g. "A".
    atom_name:
        Optional atom filter. Default is "CA" for C-alpha ANM.
        Use None to keep all atoms.
    include_hetatm:
        Whether to include HETATM records.

    Returns
    -------
    list[AtomRecord]
        Parsed atom records matching the selection.
    """

    pdb_path = Path(pdb_path)
    records = {"ATOM"}
    if include_hetatm:
        records.add("HETATM")

    atoms: list[AtomRecord] = []

    with pdb_path.open("r") as handle:
        for line in handle:
            if line[:6].strip() not in records:
                continue

            parsed_atom_name = line[12:16].strip()
            parsed_chain = line[21].strip()

            if chain is not None and parsed_chain != chain:
                continue

            if atom_name is not None and parsed_atom_name != atom_name:
                continue

            alt_loc = line[16].strip()

            # Keep blank or primary altloc only.
            if alt_loc not in {"", "A"}:
                continue

            atom = AtomRecord(
                record_name=line[:6].strip(),
                serial=int(line[6:11]),
                atom_name=parsed_atom_name,
                alt_loc=alt_loc,
                res_name=line[17:20].strip(),
                chain_id=parsed_chain,
                res_id=int(line[22:26]),
                insertion_code=line[26].strip(),
                coord=np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=float,
                ),
                occupancy=float(line[54:60] or 0.0),
                temp_factor=float(line[60:66] or 0.0),
                element=line[76:78].strip(),
                charge=line[78:80].strip(),
                raw_line=line.rstrip("\n"),
            )
            atoms.append(atom)

    if not atoms:
        raise ValueError(
            f"No atoms found in {pdb_path} with "
            f"chain={chain!r}, atom_name={atom_name!r}."
        )

    return atoms

def select_calpha(
    atoms: Iterable[AtomRecord],
    chain: str | None = None,
) -> list[int]:
    """Return indices of C-alpha atoms, optionally restricted to one chain."""

    indices = []

    for i, atom in enumerate(atoms):
        if atom.atom_name != "CA":
            continue

        if chain is not None and atom.chain_id != chain:
            continue

        indices.append(i)

    if not indices:
        raise ValueError(f"No C-alpha atoms found for chain={chain!r}.")

    return indices


def update_selected_coordinates(
    atoms: Iterable[AtomRecord],
    indices: Iterable[int],
    coordinates: np.ndarray,
) -> list[AtomRecord]:
    """
    Return full atom list where only selected atom coordinates are replaced.
    Useful for inserting morphed C-alpha coordinates into the full structure.
    """

    atoms = list(atoms)
    indices = list(indices)
    coordinates = np.asarray(coordinates, dtype=float)

    if coordinates.shape != (len(indices), 3):
        raise ValueError(
            f"Expected coordinates with shape {(len(indices), 3)}, "
            f"got {coordinates.shape}."
        )

    updated_atoms = [
        AtomRecord(
            record_name=atom.record_name,
            serial=atom.serial,
            atom_name=atom.atom_name,
            alt_loc=atom.alt_loc,
            res_name=atom.res_name,
            chain_id=atom.chain_id,
            res_id=atom.res_id,
            insertion_code=atom.insertion_code,
            coord=atom.coord.copy(),
            occupancy=atom.occupancy,
            temp_factor=atom.temp_factor,
            element=atom.element,
            charge=atom.charge,
            raw_line=atom.raw_line,
        )
        for atom in atoms
    ]

    for atom_index, coord in zip(indices, coordinates):
        updated_atoms[atom_index].coord = np.array(coord, dtype=float)

    return updated_atoms


def get_coordinates(atoms: Iterable[AtomRecord]) -> np.ndarray:
    """Return an ``(n_atoms, 3)`` coordinate array."""

    coords = np.array([atom.coord for atom in atoms], dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must have shape (n_atoms, 3).")

    return coords


def update_coordinates(
    atoms: Iterable[AtomRecord],
    coordinates: np.ndarray,
) -> list[AtomRecord]:
    """
    Return new AtomRecord objects with updated coordinates.

    This does not mutate the input atoms.
    """

    atoms = list(atoms)
    coordinates = np.asarray(coordinates, dtype=float)

    if coordinates.shape != (len(atoms), 3):
        raise ValueError(
            f"Expected coordinates with shape {(len(atoms), 3)}, "
            f"got {coordinates.shape}."
        )

    updated_atoms: list[AtomRecord] = []

    for atom, coord in zip(atoms, coordinates):
        updated_atoms.append(
            AtomRecord(
                record_name=atom.record_name,
                serial=atom.serial,
                atom_name=atom.atom_name,
                alt_loc=atom.alt_loc,
                res_name=atom.res_name,
                chain_id=atom.chain_id,
                res_id=atom.res_id,
                insertion_code=atom.insertion_code,
                coord=np.array(coord, dtype=float),
                occupancy=atom.occupancy,
                temp_factor=atom.temp_factor,
                element=atom.element,
                charge=atom.charge,
                raw_line=atom.raw_line,
            )
        )

    return updated_atoms


def format_pdb_atom_line(atom: AtomRecord) -> str:
    """Format an AtomRecord as a PDB ATOM/HETATM line."""

    x, y, z = atom.coord

    return (
        f"{atom.record_name:<6}"
        f"{atom.serial:>5d} "
        f"{atom.atom_name:^4}"
        f"{atom.alt_loc:1}"
        f"{atom.res_name:>3} "
        f"{atom.chain_id:1}"
        f"{atom.res_id:>4d}"
        f"{atom.insertion_code:1}   "
        f"{x:>8.3f}"
        f"{y:>8.3f}"
        f"{z:>8.3f}"
        f"{atom.occupancy:>6.2f}"
        f"{atom.temp_factor:>6.2f}"
        f"          "
        f"{atom.element:>2}"
        f"{atom.charge:>2}"
    )


def write_pdb(
    atoms: Iterable[AtomRecord],
    output_path: str | Path,
    remark: str | None = None,
) -> None:
    """Write atoms to a single-model PDB file."""

    output_path = Path(output_path)

    with output_path.open("w") as handle:
        if remark:
            handle.write(f"REMARK {remark}\n")

        for atom in atoms:
            handle.write(format_pdb_atom_line(atom) + "\n")

        handle.write("END\n")


def write_multimodel_pdb(
    atoms: Iterable[AtomRecord],
    coordsets: np.ndarray,
    output_path: str | Path,
    remark: str | None = None,
) -> None:
    """
    Write a multi-model PDB trajectory.

    Parameters
    ----------
    atoms:
        Template atoms.
    coordsets:
        Coordinate array with shape ``(n_models, n_atoms, 3)``.
    output_path:
        Output PDB path.
    remark:
        Optional REMARK line.
    """

    atoms = list(atoms)
    coordsets = np.asarray(coordsets, dtype=float)

    expected_shape = (len(atoms), 3)

    if coordsets.ndim != 3 or coordsets.shape[1:] != expected_shape:
        raise ValueError(
            f"Expected coordsets with shape (n_models, {len(atoms)}, 3), "
            f"got {coordsets.shape}."
        )

    output_path = Path(output_path)

    with output_path.open("w") as handle:
        if remark:
            handle.write(f"REMARK {remark}\n")

        for model_idx, coords in enumerate(coordsets, start=1):
            handle.write(f"MODEL     {model_idx:>4d}\n")

            model_atoms = update_coordinates(atoms, coords)

            for atom in model_atoms:
                handle.write(format_pdb_atom_line(atom) + "\n")

            handle.write("ENDMDL\n")

        handle.write("END\n")