from anm_morph.pdb_io import (
    parse_pdb,
    select_calpha,
    get_coordinates,
)

from anm_morph.anm import (
    calculate_anm,
    count_zero_modes,
    get_nonzero_modes,
)

atoms = parse_pdb(
    "examples/4KR5.pdb",
    chain = "A",
    atom_name=None,
)

ca_idx = select_calpha(atoms)

coords = get_coordinates(
    [atoms[i] for i in ca_idx]
)

result = calculate_anm(
    coords,
    cutoff=15.0,
)

print(f"Cα atoms: {len(ca_idx)}")
print(f"Hessian shape: {result.hessian.shape}")
print(f"Zero modes: {count_zero_modes(result.eigenvalues)}")

values, modes = get_nonzero_modes(
    result,
    n_modes=3,
)

print("\nFirst three non-trivial modes:")
for i, value in enumerate(values, start=7):
    print(f"Mode {i}: {value:.6f}")

print("\nMode array shape:")
print(modes.shape)