import numpy as np
from openfermion import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner, bravyi_kitaev
from openfermion.linalg import get_sparse_operator
from openfermionpyscf import run_pyscf
import scipy.sparse.linalg as sla
from scipy.linalg import expm

# For VQE using Qiskit:
from qiskit_aer import Aer
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.exceptions import AlgorithmError

##############################
# custom VQE 
##############################
class MyVQE(VQE):
    def evaluate_energy(self, parameters: np.ndarray) -> float:
        """
        Override energy evaluation: Bind the parameters (or initial_point if parameters are empty)
        to the ansatz and pack the resulting circuit with the operator into a single tuple (pub).
        """
        try:
            # If no parameters are provided, use the stored initial_point.
            if parameters is None or len(parameters) == 0:
                parameters = self.initial_point
            # Bind the parameters to the ansatz.
            bound_circuit = self.ansatz.bind_parameters(parameters)
            # Create a single PUB as (bound_circuit, operator)
            pub = (bound_circuit, self.operator)
            # Run the estimator with a list containing the single pub.
            job = self.estimator.run([pub])
            result = job.result()[0]
        except Exception as exc:
            raise AlgorithmError("The primitive job to evaluate the energy failed!") from exc
        return result.data.evs[0]

##############################
# conversion 
##############################
def qubit_operator_to_sparse_pauli_op(qubit_operator):
    """
    Convert an OpenFermion QubitOperator to a Qiskit SparsePauliOp.
    This function determines the global number of qubits and pads all Pauli strings to that length.
    """
    global_n = 0
    for term in qubit_operator.terms.keys():
        if term:
            max_idx = max(q for q, op in term)
            global_n = max(global_n, max_idx + 1)
    if global_n == 0:
        global_n = 1
    pauli_dict = {}
    for term, coeff in qubit_operator.terms.items():
        pauli_str = ['I'] * global_n
        for qubit_index, op in term:
            pauli_str[qubit_index] = op
        pauli_str = "".join(pauli_str)
        if pauli_str in pauli_dict:
            pauli_dict[pauli_str] += coeff
        else:
            pauli_dict[pauli_str] = coeff
    pauli_list = list(pauli_dict.keys())
    coeffs = list(pauli_dict.values())
    return SparsePauliOp(pauli_list, coeffs)

##############################
# 1. define system 
##############################
geometry = [("H", (0, 0, 0)), ("H", (0, 0, 0.74))]  
basis = "sto-3g"
multiplicity = 1
charge = 0
description = "h2_example"

molecule = MolecularData(geometry, basis, multiplicity, charge, description)
molecule = run_pyscf(
    molecule,
    run_scf=True,
    run_mp2=True,
    run_cisd=True,
    run_ccsd=True,
    run_fci=True
)

##############################
# 2. Construct Hamiltonians
##############################
fermionic_hamiltonian = molecule.get_molecular_hamiltonian()
fermion_op = get_fermion_operator(fermionic_hamiltonian)
qubit_hamiltonian_jw = jordan_wigner(fermion_op)
qubit_hamiltonian_bk = bravyi_kitaev(fermion_op)

print("Jordan–Wigner Qubit Hamiltonian:")
print(qubit_hamiltonian_jw)
print("\nBravyi–Kitaev Qubit Hamiltonian:")
print(qubit_hamiltonian_bk)

##############################
# 3. Exact Diagonalization
##############################
sparse_op = get_sparse_operator(qubit_hamiltonian_jw)
ground_energy, ground_state = sla.eigsh(sparse_op, k=1, which='SA')
print("\nExact Diagonalization (Jordan–Wigner) Ground State Energy:")
print(ground_energy[0])

##############################
# 4. Time Evolution Simulation
##############################
dt = 0.1
H_matrix = sparse_op.toarray()
U = expm(-1j * H_matrix * dt)
psi_t = U @ ground_state
energy_t = np.vdot(psi_t, H_matrix @ psi_t)
print("\nEnergy expectation after one time step (Trotter evolution):")
print(np.real(energy_t))

##############################
# 5. VQE Simulation Using MyVQE
##############################
qiskit_op = qubit_operator_to_sparse_pauli_op(qubit_hamiltonian_jw)
num_qubits = qiskit_op.num_qubits
ansatz = TwoLocal(num_qubits, ['ry', 'rz'], 'cz', reps=2, entanglement='full')
optimizer = SLSQP(maxiter=200)
backend = Aer.get_backend('statevector_simulator')

vqe_solver = MyVQE(estimator=StatevectorEstimator(), ansatz=ansatz, optimizer=optimizer)
vqe_solver.initial_point = [0.0] * ansatz.num_parameters

vqe_result = vqe_solver.compute_minimum_eigenvalue(operator=qiskit_op)
print("\nVQE Estimated Ground State Energy:")
print(vqe_result.eigenvalue.real)
