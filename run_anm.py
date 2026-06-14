import numpy as np

from anm_morph.pdb_io import (
    parse_pdb,
    select_calpha,
    get_coordinates,
    update_selected_coordinates,
    write_pdb,
)

from anm_morph.anm import (
    calculate_anm,
    get_nonzero_modes,
)

from anm_morph.morph import (
    generate_mode_trajectory,
)

from anm_morph.prepare import prepare_pdb

from anm_morph.relax import relax_ca_target_trajectory

# load full structure
atoms = parse_pdb(
    "examples/4KR5.pdb",
    chain = "A",
    atom_name=None,
)
write_pdb(
    atoms,
    "examples/4kr5_chainA.pdb"
)
# select CA atoms
ca_indices = select_calpha(atoms)

ca_atoms = [atoms[i] for i in ca_indices]
ca_coords = get_coordinates(ca_atoms)

# ANM
result = calculate_anm(
    ca_coords,
    cutoff=15.0,
)

_, modes = get_nonzero_modes(
    result,
    n_modes=1,
)

# trajectory along mode 7
#coordsets = generate_mode_trajectory(
#    ca_coords,
#    modes[0],
#    max_rmsd=2.0,
#    n_frames=21,
#)
coordsets = generate_mode_trajectory(
    ca_coords,
    modes[0],
    max_rmsd=8.0,
    n_frames=5,
    both_directions=True,
)
# build full-atom frames
full_coordsets = []

for frame_ca in coordsets:

    frame_atoms = update_selected_coordinates(
        atoms,
        ca_indices,
        frame_ca,
    )

    full_coordsets.append(
        get_coordinates(frame_atoms)
    )

#write_multimodel_pdb(
#    atoms,
#    np.asarray(full_coordsets),
#    "4kr5_mode7.pdb",
#)
#
#print("Trajectory written to lysozyme_mode7.pdb")

prepare_pdb(
    "examples/4kr5_chainA.pdb",
    "outputs/4kr5_prepared.pdb",
)




trajectory_path = relax_ca_target_trajectory(
    starting_pdb="outputs/4kr5_prepared.pdb",
    ca_target_coordsets=coordsets,
    output_dir="outputs/mode7_relaxed",
    chain = "A",
    restraint_k=1000.0,
    max_iterations=500,
)

print(f"Wrote relaxed trajectory: {trajectory_path}")