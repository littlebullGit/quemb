# Author(s): Derek Wang (VQE implementation)
"""
VQE (Variational Quantum Eigensolver) solver for Bootstrap Embedding.

This module implements a VQE solver with:
- UCCSD ansatz using Qiskit
- Adaptive 3-stage convergence strategy
- FCIDUMP Hamiltonian file parsing
- Jordan-Wigner fermion-to-qubit mapping
- Statevector simulation for exact RDM calculation
- Warm-start capability for BE iterations
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal
from warnings import warn

import numpy as np
from attrs import Factory, define, field
from numpy import ndarray
from pyscf.scf.hf import RHF

from quemb.molbe.pfrag import Frags
from quemb.shared.typing import Matrix

# Qiskit imports
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.primitives import StatevectorEstimator as Estimator
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms import VQE as QiskitVQE
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.operators import FermionicOp

    QISKIT_AVAILABLE = True
except ImportError as e:
    QISKIT_AVAILABLE = False
    # Create dummy types for type hints when Qiskit is not available
    SparsePauliOp = Any  # type: ignore
    Statevector = Any  # type: ignore
    QuantumCircuit = Any  # type: ignore

    warn(
        f"Qiskit not available. VQE solver will not work. "
        f"Install with: pip install qiskit qiskit-nature qiskit-algorithms\n"
        f"Error: {e}"
    )


@define(frozen=True)
class VQE_ArgsUser:
    """
    User-facing VQE configuration arguments.

    Parameters
    ----------
    hamiltonian_dir : str
        Directory containing fragment Hamiltonians in FCIDUMP format.
        Files should be named like: h10_be2f0, h10_be2f1, etc.
        Default: 'store_h10_files/be2/'

    adaptive_convergence : bool
        Enable adaptive 3-stage convergence strategy.
        If True, VQE convergence tightens as BE converges.
        If False, use fixed convergence parameters.
        Default: True

    # Stage 1: Early BE iterations (BE iter 1-3)
    stage1_max_iter : int
        Maximum VQE iterations for stage 1. Default: 50
    stage1_energy_tol : float
        Energy convergence tolerance for stage 1. Default: 1e-3
    stage1_be_threshold : float
        BE energy change threshold to exit stage 1. Default: 1e-2

    # Stage 2: Middle BE iterations (BE iter 4-7)
    stage2_max_iter : int
        Maximum VQE iterations for stage 2. Default: 100
    stage2_energy_tol : float
        Energy convergence tolerance for stage 2. Default: 1e-4
    stage2_be_threshold : float
        BE energy change threshold to exit stage 2. Default: 1e-3

    # Stage 3: Final BE iterations (BE iter 8+)
    stage3_max_iter : int
        Maximum VQE iterations for stage 3. Default: 200
    stage3_energy_tol : float
        Energy convergence tolerance for stage 3. Default: 1e-6

    # COBYLA optimizer settings
    cobyla_rhobeg : float
        Initial step size for COBYLA. Default: 0.1
    cobyla_rhoend : float
        Final step size for COBYLA. Default: 1e-6

    # Warm start settings
    warm_start : bool
        Use previous VQE parameters as initial guess.
        Significantly speeds up convergence in later BE iterations.
        Default: True

    verbose : int
        Verbosity level (0-3). Default: 0
        0: Silent
        1: BE iteration info
        2: VQE convergence info
        3: Full debug output
    """

    hamiltonian_dir: Final[str] = "store_h10_files/be2/"

    # Adaptive convergence
    adaptive_convergence: Final[bool] = True

    # Stage 1 parameters
    stage1_max_iter: Final[int] = 50
    stage1_energy_tol: Final[float] = 1e-3
    stage1_be_threshold: Final[float] = 1e-2

    # Stage 2 parameters
    stage2_max_iter: Final[int] = 100
    stage2_energy_tol: Final[float] = 1e-4
    stage2_be_threshold: Final[float] = 1e-3

    # Stage 3 parameters
    stage3_max_iter: Final[int] = 200
    stage3_energy_tol: Final[float] = 1e-6

    # COBYLA optimizer
    cobyla_rhobeg: Final[float] = 0.1
    cobyla_rhoend: Final[float] = 1e-6

    # Warm start
    warm_start: Final[bool] = True

    # Verbosity
    verbose: Final[int] = 0


class VQEState:
    """
    Global state for VQE solver across BE iterations.

    Stores warm-start parameters and BE convergence history.
    """
    def __init__(self):
        self.fragment_params: dict[str, ndarray] = {}  # frag_name -> optimal parameters
        self.be_energy_history: list[float] = []  # BE iteration energies
        self.current_be_iter: int = 0

    def get_stage(self, be_energy_change: float, args: VQE_ArgsUser) -> int:
        """Determine VQE convergence stage based on BE convergence."""
        if not args.adaptive_convergence:
            return 2  # Use stage 2 (medium) as default

        if be_energy_change < args.stage2_be_threshold:
            return 3  # Tight convergence
        elif be_energy_change < args.stage1_be_threshold:
            return 2  # Medium convergence
        else:
            return 1  # Loose convergence

    def get_vqe_params(self, stage: int, args: VQE_ArgsUser) -> tuple[int, float]:
        """Get VQE max_iter and energy_tol for given stage."""
        if stage == 1:
            return args.stage1_max_iter, args.stage1_energy_tol
        elif stage == 2:
            return args.stage2_max_iter, args.stage2_energy_tol
        else:  # stage == 3
            return args.stage3_max_iter, args.stage3_energy_tol

    def update_be_iteration(self, be_energy: float):
        """Update BE iteration counter and energy history."""
        self.be_energy_history.append(be_energy)
        self.current_be_iter += 1

    def get_be_energy_change(self) -> float:
        """Calculate BE energy change from previous iteration."""
        if len(self.be_energy_history) < 2:
            return 1.0  # Large value for first iteration
        return abs(self.be_energy_history[-1] - self.be_energy_history[-2])


# Global VQE state (persists across BE iterations)
_vqe_state = VQEState()


def parse_fcidump_hamiltonian(filepath: Path) -> tuple[SparsePauliOp, int, int, float]:
    """
    Parse FCIDUMP-format Hamiltonian file.

    File format:
    &FCI NORB=N, NELEC=M, MS2=S, ...
    value  p  q  r  s   (two-electron integrals)
    value  p  q  0  0   (one-electron integrals)
    value  0  0  0  0   (core energy)

    Parameters
    ----------
    filepath : Path
        Path to FCIDUMP file

    Returns
    -------
    hamiltonian : SparsePauliOp
        Qubit Hamiltonian in Pauli operator form
    norb : int
        Number of orbitals
    nelec : int
        Number of electrons
    core_energy : float
        Nuclear/core repulsion energy
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for VQE solver")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse header
    header = lines[0]
    norb_match = re.search(r'NORB\s*=\s*(\d+)', header)
    nelec_match = re.search(r'NELEC\s*=\s*(\d+)', header)

    if not norb_match or not nelec_match:
        raise ValueError(f"Could not parse NORB/NELEC from header: {header}")

    norb = int(norb_match.group(1))
    nelec = int(nelec_match.group(1))

    # Parse integrals
    h1 = np.zeros((norb, norb))  # One-electron integrals
    h2 = np.zeros((norb, norb, norb, norb))  # Two-electron integrals
    core_energy = 0.0

    for line in lines[4:]:  # Skip header lines
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        value = float(parts[0])
        p, q, r, s = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])

        if p == 0 and q == 0 and r == 0 and s == 0:
            # Core energy
            core_energy = value
        elif r == 0 and s == 0:
            # One-electron integral (convert to 0-indexed)
            h1[p-1, q-1] = value
            if p != q:
                h1[q-1, p-1] = value  # Hermitian
        else:
            # Two-electron integral (convert to 0-indexed)
            h2[p-1, q-1, r-1, s-1] = value

    # Build FermionicOp Hamiltonian using spin orbitals
    # We need to convert spatial orbitals to spin orbitals
    # For each spatial orbital p, we have spin orbitals 2*p (alpha) and 2*p+1 (beta)
    fermionic_terms = {}

    # One-electron terms: sum over spin
    for p in range(norb):
        for q in range(norb):
            if abs(h1[p, q]) > 1e-12:
                # Alpha spin
                term_alpha = f"+_{2*p} -_{2*q}"
                fermionic_terms[term_alpha] = h1[p, q]
                # Beta spin
                term_beta = f"+_{2*p+1} -_{2*q+1}"
                fermionic_terms[term_beta] = h1[p, q]

    # Two-electron terms: sum over spins
    # H2 in FCIDUMP is in physicist's notation: <pq|rs>
    # Hamiltonian term: 0.5 * sum_pqrs <pq|rs> a+_p a+_q a_s a_r
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    if abs(h2[p, q, r, s]) > 1e-12:
                        coeff = 0.5 * h2[p, q, r, s]

                        # Alpha-alpha interaction
                        term = f"+_{2*p} +_{2*q} -_{2*s} -_{2*r}"
                        fermionic_terms[term] = fermionic_terms.get(term, 0.0) + coeff

                        # Alpha-beta interaction
                        term = f"+_{2*p} +_{2*q+1} -_{2*s+1} -_{2*r}"
                        fermionic_terms[term] = fermionic_terms.get(term, 0.0) + coeff

                        # Beta-alpha interaction
                        term = f"+_{2*p+1} +_{2*q} -_{2*s} -_{2*r+1}"
                        fermionic_terms[term] = fermionic_terms.get(term, 0.0) + coeff

                        # Beta-beta interaction
                        term = f"+_{2*p+1} +_{2*q+1} -_{2*s+1} -_{2*r+1}"
                        fermionic_terms[term] = fermionic_terms.get(term, 0.0) + coeff

    # Create FermionicOp
    fermionic_op = FermionicOp(fermionic_terms, num_spin_orbitals=2*norb)

    # Map to qubits using Jordan-Wigner
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(fermionic_op)

    # Remove negligible imaginary parts introduced by numerical noise
    if hasattr(qubit_op, "simplify"):
        qubit_op = qubit_op.simplify(atol=1e-12)

    # Ensure Hermiticity by projecting coefficients onto the reals
    if hasattr(qubit_op, "coeffs"):
        coerced_coeffs = np.real_if_close(qubit_op.coeffs, tol=1e-9)
        max_imag = float(np.max(np.abs(np.imag(coerced_coeffs)))) if coerced_coeffs.size else 0.0
        if max_imag > 0.0:
            warn(
                "Projected qubit Hamiltonian coefficients to reals; "
                f"residual imaginary magnitude={max_imag:.2e}"
            )
        qubit_op = SparsePauliOp(qubit_op.paulis, np.real(coerced_coeffs))

    return qubit_op, norb, nelec, core_energy


def build_uccsd_ansatz(norb: int, nelec: int) -> QuantumCircuit:
    """
    Build UCCSD (Unitary Coupled Cluster Singles and Doubles) ansatz.

    Parameters
    ----------
    norb : int
        Number of spatial orbitals
    nelec : int
        Number of electrons

    Returns
    -------
    ansatz : QuantumCircuit
        Parameterized UCCSD circuit
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for VQE solver")

    from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

    mapper = JordanWignerMapper()

    # For RHF (restricted, closed-shell), we assume all electrons are spin-up
    # nelec total electrons means nelec/2 alpha, nelec/2 beta for closed shell
    # But for odd nelec, we use (nelec+1)//2 alpha, nelec//2 beta
    n_alpha = (nelec + 1) // 2
    n_beta = nelec - n_alpha
    num_particles = (n_alpha, n_beta)

    # Initial state: Hartree-Fock
    hf_state = HartreeFock(norb, num_particles, mapper)

    # UCCSD ansatz
    ansatz = UCCSD(
        num_spatial_orbitals=norb,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=hf_state
    )

    return ansatz


def compute_rdms_from_statevector(
    statevector: Statevector,
    norb: int,
    nelec: int
) -> tuple[ndarray, ndarray]:
    """
    Compute 1-RDM and 2-RDM from VQE statevector.

    Uses Jordan-Wigner mapping to compute fermionic RDMs.

    Parameters
    ----------
    statevector : Statevector
        Optimized VQE statevector
    norb : int
        Number of spatial orbitals
    nelec : int
        Number of electrons

    Returns
    -------
    rdm1 : ndarray (norb, norb)
        One-particle reduced density matrix
    rdm2 : ndarray (norb, norb, norb, norb)
        Two-particle reduced density matrix
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for VQE solver")

    nqubits = 2 * norb

    # Initialize RDMs
    rdm1 = np.zeros((norb, norb), dtype=complex)
    rdm2 = np.zeros((norb, norb, norb, norb), dtype=complex)

    # 1-RDM: <a+_p a_q>
    for p in range(norb):
        for q in range(norb):
            # Spin-up sector only (since we have RHF with beta=0)
            op_str = f"+_{p} -_{q}"
            fermionic_op = FermionicOp({op_str: 1.0}, num_spin_orbitals=nqubits)

            mapper = JordanWignerMapper()
            pauli_op = mapper.map(fermionic_op)

            # Expectation value
            rdm1[p, q] = statevector.expectation_value(pauli_op)

    # 2-RDM: <a+_p a+_q a_s a_r>
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    # Spin-up sector only
                    op_str = f"+_{p} +_{q} -_{s} -_{r}"
                    fermionic_op = FermionicOp({op_str: 1.0}, num_spin_orbitals=nqubits)

                    mapper = JordanWignerMapper()
                    pauli_op = mapper.map(fermionic_op)

                    # Expectation value
                    rdm2[p, q, r, s] = statevector.expectation_value(pauli_op)

    # Convert to real (imaginary parts should be negligible)
    rdm1 = rdm1.real
    rdm2 = rdm2.real

    return rdm1, rdm2


def solve_vqe(
    mf: RHF,
    frag: Frags,
    vqe_args: VQE_ArgsUser,
    be_energy: float | None = None,
) -> tuple[Matrix, Matrix]:
    """
    Solve fragment using VQE with UCCSD ansatz.

    This function:
    1. Loads pre-computed Hamiltonian from FCIDUMP file
    2. Builds UCCSD ansatz circuit
    3. Runs VQE with adaptive convergence
    4. Computes 1-RDM and 2-RDM from optimized statevector
    5. Supports warm-start for faster convergence

    Parameters
    ----------
    mf : RHF
        Mean-field object (for interface compatibility, not used)
    frag : Frags
        Fragment object containing fragment index
    vqe_args : VQE_ArgsUser
        VQE configuration parameters
    be_energy : float, optional
        Current BE total energy (for adaptive convergence)

    Returns
    -------
    rdm1 : ndarray (norb, norb)
        One-particle reduced density matrix
    rdm2 : ndarray (norb, norb, norb, norb)
        Two-particle reduced density matrix
    """
    if not QISKIT_AVAILABLE:
        raise ImportError(
            "Qiskit is required for VQE solver. "
            "Install with: pip install qiskit qiskit-nature qiskit-algorithms"
        )

    global _vqe_state

    # Update BE iteration tracking
    if be_energy is not None:
        _vqe_state.update_be_iteration(be_energy)

    # Determine convergence stage
    be_energy_change = _vqe_state.get_be_energy_change()
    stage = _vqe_state.get_stage(be_energy_change, vqe_args)
    max_iter, energy_tol = _vqe_state.get_vqe_params(stage, vqe_args)

    if vqe_args.verbose >= 1:
        print(f"VQE Fragment {frag.dname}: Stage {stage}, "
              f"BE ΔE={be_energy_change:.2e}, "
              f"max_iter={max_iter}, tol={energy_tol:.2e}")

    # Load Hamiltonian from file
    ham_dir = Path(vqe_args.hamiltonian_dir)
    # Fragment name should match file pattern, e.g., "be2f0" -> "h10_be2f0"
    frag_name = str(frag.dname)  # e.g., "be2f0"
    ham_file = ham_dir / f"h10_{frag_name}"

    if not ham_file.exists():
        raise FileNotFoundError(
            f"Hamiltonian file not found: {ham_file}\n"
            f"Expected format: {vqe_args.hamiltonian_dir}/h10_{{fragment_name}}"
        )

    # Parse Hamiltonian
    qubit_hamiltonian, norb, nelec, core_energy = parse_fcidump_hamiltonian(ham_file)

    if vqe_args.verbose >= 2:
        print(f"  Loaded Hamiltonian: norb={norb}, nelec={nelec}, "
              f"nqubits={2*norb}, core_energy={core_energy:.6f}")

    # Build UCCSD ansatz
    ansatz = build_uccsd_ansatz(norb, nelec)

    # Initial parameters (warm-start or cold-start)
    if vqe_args.warm_start and frag_name in _vqe_state.fragment_params:
        initial_point = _vqe_state.fragment_params[frag_name]
        if vqe_args.verbose >= 2:
            print(f"  Using warm-start parameters (size={len(initial_point)})")
    else:
        initial_point = np.zeros(ansatz.num_parameters)
        if vqe_args.verbose >= 2:
            print(f"  Using cold-start (zeros, size={len(initial_point)})")

    # Setup COBYLA optimizer
    # Note: Qiskit 2.x COBYLA doesn't accept rhobeg/rhoend directly
    optimizer = COBYLA(
        maxiter=max_iter,
        tol=energy_tol,
    )

    # Setup VQE
    estimator = Estimator()
    vqe = QiskitVQE(estimator, ansatz, optimizer, initial_point=initial_point)

    # Run VQE
    if vqe_args.verbose >= 2:
        print(f"  Running VQE optimization...")

    result = vqe.compute_minimum_eigenvalue(qubit_hamiltonian)

    # Extract results
    optimal_energy = result.eigenvalue.real + core_energy
    optimal_params = result.optimal_point

    if vqe_args.verbose >= 1:
        print(f"  VQE converged: E={optimal_energy:.8f}, "
              f"iterations={result.cost_function_evals}")

    # Store optimal parameters for warm-start
    if vqe_args.warm_start:
        _vqe_state.fragment_params[frag_name] = optimal_params

    # Compute statevector from optimal parameters
    bound_circuit = ansatz.assign_parameters(optimal_params)
    statevector = Statevector(bound_circuit)

    # Compute RDMs from statevector
    if vqe_args.verbose >= 2:
        print(f"  Computing RDMs from statevector...")

    rdm1, rdm2 = compute_rdms_from_statevector(statevector, norb, nelec)

    if vqe_args.verbose >= 2:
        print(f"  RDM1 trace: {np.trace(rdm1):.6f} (expected: {nelec})")
        print(f"  RDM2 computed successfully")

    return rdm1, rdm2
