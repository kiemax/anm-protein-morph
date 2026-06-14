"""
prepare.py

Prepare PDB structures for OpenMM relaxation using PDBFixer.
"""

from __future__ import annotations

from pathlib import Path

from openmm.app import PDBFile
from pdbfixer import PDBFixer


def prepare_pdb(
    input_pdb: str | Path,
    output_pdb: str | Path,
    ph: float = 7.0,
    keep_heterogens: bool = False,
    add_hydrogens: bool = True,
) -> None:
    """
    Prepare a PDB structure for OpenMM.

    This fixes missing atoms and optionally adds hydrogens.
    """

    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)

    fixer = PDBFixer(filename=str(input_pdb))

    fixer.findMissingResidues()
    fixer.missingResidues = {}   # do not add missing residues
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    if not keep_heterogens:
        fixer.removeHeterogens(keepWater=False)

    if add_hydrogens:
        fixer.addMissingHydrogens(pH=ph)

    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    with output_pdb.open("w") as handle:
        PDBFile.writeFile(
            fixer.topology,
            fixer.positions,
            handle,
            keepIds=True,
        )