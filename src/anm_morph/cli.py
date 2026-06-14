from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from anm_morph.anm import calculate_anm, get_nonzero_modes, count_zero_modes
from anm_morph.morph import generate_mode_trajectory, combine_modes
from anm_morph.pdb_io import parse_pdb, select_calpha, get_coordinates
from anm_morph.prepare import prepare_pdb
from anm_morph.relax import relax_ca_target_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ANM-based protein morph trajectories."
    )

    parser.add_argument("--pdb", required=True, help="Input PDB file.")
    parser.add_argument("--outdir", default="outputs", help="Output directory.")
    parser.add_argument("--chain", default=None, help="Optional chain ID.")

    parser.add_argument(
        "--modes",
        nargs="+",
        type=int,
        default=[7],
        help="ANM mode numbers to export, using 1-based numbering. Default: 7.",
    )

    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine selected modes into one trajectory.",
    )

    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Weights for combined modes. Must match number of modes.",
    )

    parser.add_argument("--cutoff", type=float, default=15.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--max-rmsd", type=float, default=2.0)
    parser.add_argument("--frames", type=int, default=11)

    parser.add_argument(
        "--one-direction",
        action="store_true",
        help="Generate trajectory from 0 to max RMSD instead of -max to +max.",
    )

    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Use input PDB directly without PDBFixer preparation.",
    )

    parser.add_argument(
        "--no-relax",
        action="store_true",
        help="Only generate C-alpha target trajectories; skip OpenMM relaxation.",
    )

    parser.add_argument("--restraint-k", type=float, default=1000.0)
    parser.add_argument("--max-iterations", type=int, default=500)

    return parser.parse_args()


def run() -> None:
    args = parse_args()

    input_pdb = Path(args.pdb)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prepared_pdb = outdir / "prepared.pdb"

    if args.skip_prepare:
        prepared_pdb = input_pdb
    else:
        print("Preparing PDB...")
        prepare_pdb(
            input_pdb=input_pdb,
            output_pdb=prepared_pdb,
            #chain=args.chain,
        )

    print("Parsing prepared structure...")
    atoms = parse_pdb(prepared_pdb, atom_name=None)
    ca_indices = select_calpha(atoms, chain=args.chain)
    ca_atoms = [atoms[i] for i in ca_indices]
    ca_coords = get_coordinates(ca_atoms)

    print(f"C-alpha atoms: {len(ca_indices)}")

    print("Calculating ANM...")
    result = calculate_anm(
        ca_coords,
        cutoff=args.cutoff,
        gamma=args.gamma,
    )

    zero_modes = count_zero_modes(result.eigenvalues)
    print(f"Zero modes: {zero_modes}")

    max_requested_mode = max(args.modes)
    if min(args.modes) <= 6:
        raise ValueError("Modes 1-6 are rigid-body modes. Use mode 7 or higher.")

    n_modes_needed = max_requested_mode - 6

    values, modes = get_nonzero_modes(
        result,
        n_modes=n_modes_needed,
        n_zero_modes=6,
    )

    selected_modes = []
    selected_values = []

    for mode_number in args.modes:
        idx = mode_number - 7
        selected_modes.append(modes[idx])
        selected_values.append(values[idx])

    selected_modes = np.asarray(selected_modes)

    print("Selected modes:")
    for mode_number, eigenvalue in zip(args.modes, selected_values):
        print(f"  Mode {mode_number}: eigenvalue {eigenvalue:.6f}")

    both_directions = not args.one_direction

    if args.combine:
        print("Combining modes...")

        combined_mode = combine_modes(
            selected_modes,
            weights=args.weights,
        )

        ca_target_coordsets = generate_mode_trajectory(
            ca_coords,
            combined_mode,
            max_rmsd=args.max_rmsd,
            n_frames=args.frames,
            both_directions=both_directions,
        )

        mode_label = "_".join(str(m) for m in args.modes)
        mode_outdir = outdir / f"combined_modes_{mode_label}"

        if args.no_relax:
            raise NotImplementedError(
                "--no-relax output writing for target-only trajectories "
                "should be added after CLI skeleton is stable."
            )

        trajectory_path = relax_ca_target_trajectory(
            starting_pdb=prepared_pdb,
            ca_target_coordsets=ca_target_coordsets,
            output_dir=mode_outdir,
            chain=args.chain,
            restraint_k=args.restraint_k,
            max_iterations=args.max_iterations,
            trajectory_name="relaxed_trajectory.pdb",
        )

        print(f"Wrote relaxed trajectory: {trajectory_path}")

    else:
        for mode_number, mode in zip(args.modes, selected_modes):
            print(f"Generating trajectory for mode {mode_number}...")

            ca_target_coordsets = generate_mode_trajectory(
                ca_coords,
                mode,
                max_rmsd=args.max_rmsd,
                n_frames=args.frames,
                both_directions=both_directions,
            )

            mode_outdir = outdir / f"mode_{mode_number}"

            if args.no_relax:
                raise NotImplementedError(
                    "--no-relax output writing for target-only trajectories "
                    "should be added after CLI skeleton is stable."
                )

            trajectory_path = relax_ca_target_trajectory(
                starting_pdb=prepared_pdb,
                ca_target_coordsets=ca_target_coordsets,
                output_dir=mode_outdir,
                chain=args.chain,
                restraint_k=args.restraint_k,
                max_iterations=args.max_iterations,
                trajectory_name="relaxed_trajectory.pdb",
            )

            print(f"Wrote relaxed trajectory: {trajectory_path}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()