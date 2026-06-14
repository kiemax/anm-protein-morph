"""
relax.py

OpenMM-based structural relaxation for ANM-derived C-alpha morphs.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from openmm import CustomExternalForce, LangevinIntegrator, unit
from openmm.app import (
    ForceField,
    Modeller,
    NoCutoff,
    PDBFile,
    Simulation,
)

from anm_morph.pdb_io import (
    get_coordinates,
    parse_pdb,
    select_calpha,
    update_selected_coordinates,
    write_multimodel_pdb,
    write_pdb,
)


def minimize_pdb(
    input_pdb: str | Path,
    output_pdb: str | Path,
    restrained_atom_names: set[str] | None = None,
    restraint_k: float = 1000.0,
    max_iterations: int = 500,
    forcefield_files: tuple[str, ...] = ("amber14-all.xml",),
) -> None:
    """
    Minimize a PDB structure with optional positional restraints.

    Parameters
    ----------
    input_pdb:
        Input full-atom PDB file.
    output_pdb:
        Output minimized PDB file.
    restrained_atom_names:
        Atom names to restrain. Use {"CA"} to keep morphed C-alpha atoms
        close to their target positions.
    restraint_k:
        Harmonic restraint strength in kJ mol^-1 nm^-2.
    max_iterations:
        Maximum minimization iterations.
    forcefield_files:
        OpenMM force field XML files.
    """

    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)

    pdb = PDBFile(str(input_pdb))

    modeller = Modeller(pdb.topology, pdb.positions)

    forcefield = ForceField(*forcefield_files)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=None,
    )

    if restrained_atom_names:
        restraint = CustomExternalForce(
            "0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
        )
        restraint.addGlobalParameter(
            "k",
            restraint_k * unit.kilojoule_per_mole / unit.nanometer**2,
        )
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")

        for atom, position in zip(modeller.topology.atoms(), modeller.positions):
            if atom.name in restrained_atom_names:
                restraint.addParticle(
                    atom.index,
                    [
                        position.x,
                        position.y,
                        position.z,
                    ],
                )

        system.addForce(restraint)

    integrator = LangevinIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )

    simulation = Simulation(
        modeller.topology,
        system,
        integrator,
    )

    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=max_iterations)

    state = simulation.context.getState(getPositions=True)
    minimized_positions = state.getPositions()

    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    with output_pdb.open("w") as handle:
        PDBFile.writeFile(
            modeller.topology,
            minimized_positions,
            handle,
            keepIds=True,
        )


def minimize_with_ca_restraints(
    input_pdb: str | Path,
    output_pdb: str | Path,
    restraint_k: float = 1000.0,
    max_iterations: int = 500,
) -> None:
    """
    Convenience wrapper: minimize while restraining C-alpha atoms.
    """

    minimize_pdb(
        input_pdb=input_pdb,
        output_pdb=output_pdb,
        restrained_atom_names={"CA"},
        restraint_k=restraint_k,
        max_iterations=max_iterations,
    )

def relax_ca_target_trajectory(
    starting_pdb: str | Path,
    ca_target_coordsets: np.ndarray,
    output_dir: str | Path,
    chain: str | None = None,
    restraint_k: float = 1000.0,
    max_iterations: int = 500,
    write_intermediate_targets: bool = True,
    trajectory_name: str = "relaxed_trajectory.pdb",
) -> Path:
    """
    Sequentially relax a structure toward a series of C-alpha target coordinates.

    Parameters
    ----------
    starting_pdb:
        Prepared full-atom starting structure.
    ca_target_coordsets:
        C-alpha target coordinates with shape (n_frames, n_ca, 3), in Angstrom.
    output_dir:
        Directory where intermediate and relaxed frames are written.
    restraint_k:
        Harmonic restraint strength in kJ mol^-1 nm^-2.
    max_iterations:
        Maximum minimization iterations per frame.
    write_intermediate_targets:
        Whether to write pre-minimization target PDBs.
    trajectory_name:
        Name of the final multi-model relaxed PDB trajectory.

    Returns
    -------
    Path
        Path to the final multi-model relaxed PDB trajectory.
    """

    starting_pdb = Path(starting_pdb)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ca_target_coordsets = np.asarray(ca_target_coordsets, dtype=float)

    if ca_target_coordsets.ndim != 3 or ca_target_coordsets.shape[2] != 3:
        raise ValueError(
            "ca_target_coordsets must have shape (n_frames, n_ca, 3)."
        )

    current_pdb = starting_pdb
    relaxed_full_coordsets = []

    for frame_idx, target_ca_coords in enumerate(ca_target_coordsets, start=1):
        current_atoms = parse_pdb(current_pdb, atom_name=None)
        ca_indices = select_calpha(current_atoms, chain = chain)

        if target_ca_coords.shape != (len(ca_indices), 3):
            raise ValueError(
                f"Frame {frame_idx}: expected C-alpha coordinates with shape "
                f"{(len(ca_indices), 3)}, got {target_ca_coords.shape}."
            )

        target_atoms = update_selected_coordinates(
            current_atoms,
            ca_indices,
            target_ca_coords,
        )

        target_pdb = output_dir / f"frame_{frame_idx:03d}_target.pdb"
        relaxed_pdb = output_dir / f"frame_{frame_idx:03d}_relaxed.pdb"

        if write_intermediate_targets:
            write_pdb(
                target_atoms,
                target_pdb,
                remark=f"C-alpha target frame {frame_idx}",
            )
            minimization_input = target_pdb
        else:
            # OpenMM needs a real file input, so write a temporary target anyway.
            write_pdb(
                target_atoms,
                target_pdb,
                remark=f"C-alpha target frame {frame_idx}",
            )
            minimization_input = target_pdb

        minimize_with_ca_restraints(
            input_pdb=minimization_input,
            output_pdb=relaxed_pdb,
            restraint_k=restraint_k,
            max_iterations=max_iterations,
        )

        relaxed_atoms = parse_pdb(relaxed_pdb, atom_name=None)
        relaxed_full_coordsets.append(get_coordinates(relaxed_atoms))

        current_pdb = relaxed_pdb

        print(f"Relaxed frame {frame_idx}/{len(ca_target_coordsets)}")

    final_atoms = parse_pdb(current_pdb, atom_name=None)

    trajectory_path = output_dir / trajectory_name

    write_multimodel_pdb(
        final_atoms,
        np.asarray(relaxed_full_coordsets),
        trajectory_path,
        remark="Sequentially relaxed ANM trajectory",
    )

    if not write_intermediate_targets:
        for target_pdb in output_dir.glob("frame_*_target.pdb"):
            target_pdb.unlink(missing_ok=True)

    return trajectory_path