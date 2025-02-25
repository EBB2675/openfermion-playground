from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator, jordan_wigner

# Define the molecular geometry and parameters.
geometry = [("H", (0, 0, 0)), ("H", (0, 0, 0.74))]  # coordinates in Angstroms
basis = "sto-3g"
multiplicity = 1  # singlet state
charge = 0
description = "h2_example"

# Create a MolecularData object.
molecule = MolecularData(geometry, basis, multiplicity, charge, description)

# Run the electronic structure calculation with PySCF.
# The keyword arguments specify which methods to run.
molecule = run_pyscf(
    molecule,
    run_scf=True,
    run_mp2=True,
    run_cisd=True,
    run_ccsd=True,
    run_fci=True
)

# Now obtain the molecular Hamiltonian.
fermion_hamiltonian = molecule.get_molecular_hamiltonian()

# Convert to a fermion operator.
fermion_op = get_fermion_operator(fermion_hamiltonian)

# Transform to a qubit operator using the Jordan-Wigner transformation.
qubit_hamiltonian = jordan_wigner(fermion_op)

print("Qubit Hamiltonian:")
print(qubit_hamiltonian)
