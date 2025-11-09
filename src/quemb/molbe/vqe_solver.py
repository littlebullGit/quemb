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
- Optional diagnostics for iteration-level convergence traces
"""

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal
from warnings import warn
from time import perf_counter

import h5py
import numpy as np
from attrs import Factory, define, field
from numpy import ndarray
from pyscf import ao2mo
from pyscf.scf.hf import RHF
from pyscf.tools import fcidump

from quemb.molbe.pfrag import Frags
from quemb.shared.typing import Matrix

# Qiskit imports
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter
    from qiskit.primitives import BackendEstimatorV2
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SPSA, SLSQP
    from qiskit_algorithms import VQE as QiskitVQE
    from qiskit_algorithms.exceptions import AlgorithmError
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.operators import ElectronicIntegrals, FermionicOp
    from qiskit_aer import AerSimulator

    QISKIT_AVAILABLE = True
except ImportError as e:
    QISKIT_AVAILABLE = False
    # Create dummy types for type hints when Qiskit is not available
    SparsePauliOp = Any  # type: ignore
    Statevector = Any  # type: ignore
    QuantumCircuit = Any  # type: ignore
    ElectronicEnergy = Any  # type: ignore
    ElectronicIntegrals = Any  # type: ignore

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

    # Optimizer settings
    optimizer_name : str
        Optimizer to use ('SPSA', 'COBYLA', 'L_BFGS_B', 'SLSQP'). Default: 'SPSA'
    cobyla_rhobeg : float
        Initial step size for COBYLA. Default: 0.1
    cobyla_rhoend : float
        Final step size for COBYLA. Default: 1e-6

    max_restarts : int
        Number of optimizer runs to attempt (including the first run). Each
        additional run starts from a random initial point; the lowest-energy
        solution is selected. Default: 3
    restart_energy_tol : float
        Skip further restarts when the improvement of the best energy falls
        below this threshold. Default: 1e-6
    random_seed : int | None
        Seed for random initial points. Default: None (use entropy)

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
    show_progress_bar : bool
        Display a textual progress bar during optimizer evaluations. Default: False
    track_iteration_history : bool
        Record optimizer iteration diagnostics for later inspection.
        Default: False
    track_density_matrices : bool
        Compute intermediate one-particle density matrices at callback checkpoints.
        Default: False
    density_sample_interval : int
        Interval (in optimizer evaluations) between density matrix captures when
        ``track_density_matrices`` is enabled. Default: 1
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

    # Optimizer selection
    optimizer_name: Final[str] = "SPSA"  # Changed from COBYLA: SPSA reduces fragment asymmetry by 49%
    
    # COBYLA optimizer
    cobyla_rhobeg: Final[float] = 0.1
    cobyla_rhoend: Final[float] = 1e-6

    # Restart settings
    max_restarts: Final[int] = 3
    restart_energy_tol: Final[float] = 1e-6
    random_seed: Final[int | None] = None

    # Warm start
    warm_start: Final[bool] = True

    # Verbosity
    verbose: Final[int] = 0
    # Diagnostics / tracing
    show_progress_bar: Final[bool] = False
    track_iteration_history: Final[bool] = False
    track_density_matrices: Final[bool] = False
    density_sample_interval: Final[int] = 1


class VQEState:
    """
    Global state for VQE solver across BE iterations.

    Stores warm-start parameters and BE convergence history.
    """
    def __init__(self):
        self.fragment_params: dict[str, ndarray] = {}  # frag_name -> optimal parameters
        self.be_energy_history: list[float] = []  # BE iteration energies
        self.current_be_iter: int = 0
        self.fragment_iteration_history: dict[str, list[dict[str, Any]]] = {}
        self.fragment_energies: dict[str, float] = {}
        self.fragment_rdm1: dict[str, ndarray] = {}
        self.fragment_rdm2: dict[str, ndarray] = {}
        self.fragment_timings: dict[str, dict[str, float]] = {}

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

    Parameters
    ----------
    filepath : Path
        Path to FCIDUMP file

    Returns
    -------
    hamiltonian : SparsePauliOp
        Qubit Hamiltonian in Pauli operator form
    norb : int
        Number of spatial orbitals
    nelec : int
        Number of electrons
    core_energy : float
        Core/nuclear energy shift (not included in the qubit Hamiltonian)
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for VQE solver")

    fc_data = fcidump.read(str(filepath))

    norb = int(fc_data["NORB"])
    nelec = int(fc_data["NELEC"])
    core_energy = float(fc_data["ECORE"])

    h1 = np.asarray(fc_data["H1"], dtype=float)
    # Restore the full two-electron tensor in chemists' notation <pq|rs>
    h2 = ao2mo.restore(1, fc_data["H2"], norb)

    # Build fermionic Hamiltonian using Qiskit Nature helpers (handles spin expansion)
    electronic_integrals = ElectronicIntegrals.from_raw_integrals(
        h1_a=h1,
        h2_aa=h2,
        h1_b=h1,
        h2_bb=h2,
        h2_ba=np.einsum("pqrs->qprs", h2),
    )
    electronic_energy = ElectronicEnergy(electronic_integrals)
    fermionic_op = electronic_energy.second_q_op()

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

    # For BE fragments: spin configuration depends on electron count
    # Odd nelec: use balanced (gives doublet S=1/2)
    # Even nelec: use balanced (gives singlet S=0)
    # Note: balanced means n_alpha = (nelec+1)//2, n_beta = nelec - n_alpha
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


def compute_rdm1_from_statevector(
    statevector: Statevector,
    norb: int,
    nelec: int,  # noqa: ARG001 - kept for signature parity with full RDM helper
) -> ndarray:
    """
    Compute 1-RDM from a VQE statevector.

    Parameters
    ----------
    statevector : Statevector
        Optimized or intermediate VQE statevector.
    norb : int
        Number of spatial orbitals.
    nelec : int
        Number of electrons (unused but retained for future extensions).

    Returns
    -------
    ndarray
        One-particle reduced density matrix (real-valued).
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required for VQE solver")

    nqubits = 2 * norb
    mapper = JordanWignerMapper()
    rdm1 = np.zeros((norb, norb), dtype=complex)

    for p in range(norb):
        for q in range(norb):
            value = 0.0 + 0.0j
            for spin in (0, 1):  # 0 -> alpha, 1 -> beta
                op_str = f"+_{2 * p + spin} -_{2 * q + spin}"
                fermionic_op = FermionicOp({op_str: 1.0}, num_spin_orbitals=nqubits)
                pauli_op = mapper.map(fermionic_op)
                value += statevector.expectation_value(pauli_op)
            rdm1[p, q] = value

    return rdm1.real


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
    mapper = JordanWignerMapper()
    spin_labels = (0, 1)
    spin_configs = (
        (0, 0, 0, 0),  # αα
        (0, 1, 1, 0),  # αβ
        (1, 0, 0, 1),  # βα
        (1, 1, 1, 1),  # ββ
    )

    # 1-RDM: <a+_p a_q>
    for p in range(norb):
        for q in range(norb):
            value = 0.0 + 0.0j
            for spin in spin_labels:
                op_str = f"+_{2 * p + spin} -_{2 * q + spin}"
                fermionic_op = FermionicOp({op_str: 1.0}, num_spin_orbitals=nqubits)
                pauli_op = mapper.map(fermionic_op)
                value += statevector.expectation_value(pauli_op)
            rdm1[p, q] = value

    # 2-RDM: <a+_p a+_q a_s a_r>
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    value = 0.0 + 0.0j
                    for spin_p, spin_q, spin_s, spin_r in spin_configs:
                        op_str = (
                            f"+_{2 * p + spin_p} +_{2 * q + spin_q} "
                            f"-_{2 * s + spin_s} -_{2 * r + spin_r}"
                        )
                        fermionic_op = FermionicOp({op_str: 1.0}, num_spin_orbitals=nqubits)
                        pauli_op = mapper.map(fermionic_op)
                        value += statevector.expectation_value(pauli_op)
                    rdm2[p, q, r, s] = value

    # Convert to real (imaginary parts should be negligible)
    rdm1 = rdm1.real
    rdm2 = rdm2.real

    return rdm1, rdm2


def regenerate_fcidump_with_heff(
    frag: Frags,
    output_dir: str | Path,
) -> Path:
    """
    Regenerate FCIDUMP file with current effective Hamiltonian in fragment MO basis.

    CRITICAL: VQE requires orthonormal orbitals for Jordan-Wigner mapping.
    The embedding AO basis is NOT orthonormal (AO overlap matrix has off-diagonal
    elements), so we MUST transform to fragment MO basis before creating FCIDUMP.

    This function:
    1. Loads current effective Hamiltonian from fragment (in AO basis)
    2. Transforms h1e and h2e from embedding AO basis to fragment MO basis
    3. Writes FCIDUMP in orthonormal MO basis for VQE

    Parameters
    ----------
    frag : Frags
        Fragment object with _effective_h1e stored from recent SCF
    output_dir : str or Path
        Directory to write updated FCIDUMP file

    Returns
    -------
    Path
        Path to the regenerated FCIDUMP file

    Notes
    -----
    The Hamiltonian is extracted from frag._effective_h1e (embedding AO basis),
    which is stored in pfrag.py:285 when SCF is called.

    The transformation to MO basis uses frag.mo_coeffs:
    - h1e_mo = C^T @ h1e_ao @ C
    - h2e_mo = C^T @ C^T @ C^T @ C^T @ h2e_ao (4-index transformation)

    This ensures VQE:
    1. Uses orthonormal orbitals (required for Jordan-Wigner)
    2. Sees chemical potential updates across BE iterations
    3. Works in same basis as traditional solvers expect for RDM output
    """
    # Load 2-electron integrals from HDF5 file (in embedding AO basis)
    with h5py.File(frag.eri_file, "r") as f:
        eri = f[frag.dname][()]
    eri_ao = ao2mo.restore(1, eri, frag.nao)

    # Get CURRENT effective one-electron Hamiltonian (in embedding AO basis)
    assert hasattr(frag, '_effective_h1e'), "SCF must be run before regenerating FCIDUMP"
    h1e_ao = frag._effective_h1e

    # Get MO coefficients (transform from embedding AO to fragment MO)
    assert frag.mo_coeffs is not None, "MO coefficients not available"
    C = frag.mo_coeffs

    # Transform h1e from AO to MO basis: h1e_mo = C^T @ h1e_ao @ C
    h1e_mo = np.einsum('ip,ij,jq->pq', C, h1e_ao, C, optimize=True)

    # Transform h2e from AO to MO basis: h2e_mo[pqrs] = C_ip C_jq C_kr C_ls h2e_ao[ijkl]
    h2e_mo = np.einsum('ip,jq,kr,ls,ijkl->pqrs', C, C, C, C, eri_ao, optimize=True)

    # Use transformed integrals
    h1e = h1e_mo
    h2e = h2e_mo

    # Write to FCIDUMP file with unique name to avoid race conditions
    output_path = Path(output_dir)
    output_file = output_path / f"h10_{frag.dname}_current"

    # DEBUG: Print what we're about to write to FCIDUMP
    print(f"\n{'='*80}")
    print(f"DEBUG vqe_solver.py - Writing FCIDUMP in fragment MO basis")
    print(f"{'='*80}")
    print(f"Basis: Fragment MO (orthonormal orbitals)")
    print(f"h1e_mo.shape: {h1e.shape}")
    print(f"h1e_mo diagonal: {np.diag(h1e)}")
    print(f"h2e_mo.shape: {h2e.shape}")
    print(f"norb (MO basis): {C.shape[1]}")
    print(f"nelec: {2 * frag.nsocc}")
    print(f"h1e_mo matrix:\n{h1e}")
    print(f"{'='*80}\n")

    # Write FCIDUMP in fragment MO basis (orthonormal)
    norb_mo = C.shape[1]
    fcidump.from_integrals(
        str(output_file),
        h1e,
        h2e,
        norb_mo,          # Number of MO orbitals
        2 * frag.nsocc,   # Number of electrons
        ms=0,             # Total spin
    )

    return output_file


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

    global_start = perf_counter()
    phase_timings: dict[str, float] = {}

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

    # Regenerate FCIDUMP with current effective Hamiltonian
    # This ensures VQE sees the updated chemical potential from BE optimization
    ham_dir = Path(vqe_args.hamiltonian_dir)
    frag_name = str(frag.dname)  # Needed for warm-start logic
    ham_file = regenerate_fcidump_with_heff(frag, ham_dir)

    if vqe_args.verbose >= 2:
        print(f"  Regenerated FCIDUMP with current heff: {ham_file}")

    # Parse Hamiltonian (now includes current chemical potential)
    qubit_hamiltonian, norb, nelec, core_energy = parse_fcidump_hamiltonian(ham_file)
    t_after_parse = perf_counter()
    phase_timings["load_hamiltonian"] = t_after_parse - global_start

    if vqe_args.verbose >= 2:
        print(f"  Loaded Hamiltonian: norb={norb}, nelec={nelec}, "
              f"nqubits={2*norb}, core_energy={core_energy:.6f}")

    # Build UCCSD ansatz
    ansatz = build_uccsd_ansatz(norb, nelec)
    
    # Transpile ansatz to decompose high-level gates (EvolvedOps) into basic gates
    # This is required for Aer compatibility
    ansatz = transpile(ansatz, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=1)
    t_after_ansatz = perf_counter()
    phase_timings["build_ansatz"] = t_after_ansatz - t_after_parse
    ordered_parameters = list(ansatz.parameters)
    num_parameters = len(ordered_parameters)

    # Initial parameters (warm-start or cold-start)
    if vqe_args.warm_start and frag_name in _vqe_state.fragment_params:
        initial_point = _vqe_state.fragment_params[frag_name]
        if len(initial_point) != num_parameters:
            if vqe_args.verbose >= 1:
                print(
                    f"  Warm-start parameter length {len(initial_point)} mismatch "
                    f"with ansatz size {num_parameters}; reinitializing."
                )
            initial_point = np.zeros(num_parameters)
        elif vqe_args.verbose >= 2:
            print(f"  Using warm-start parameters (size={len(initial_point)})")
    else:
        initial_point = np.zeros(num_parameters)
        if vqe_args.verbose >= 2:
            print(f"  Using cold-start (zeros, size={len(initial_point)})")

    t_after_initial = perf_counter()
    phase_timings["initial_parameters"] = t_after_initial - t_after_ansatz

    # Setup optimizer based on selection
    optimizer_name = vqe_args.optimizer_name
    print(f"  Using optimizer: {optimizer_name}")
    
    if optimizer_name == "COBYLA":
        optimizer = COBYLA(
            maxiter=max_iter,
            tol=energy_tol,
        )
    elif optimizer_name == "L_BFGS_B":
        optimizer = L_BFGS_B(
            maxiter=max_iter,
            ftol=energy_tol,
        )
    elif optimizer_name == "SPSA":
        optimizer = SPSA(
            maxiter=max_iter,
            callback=None,  # We'll use VQE callback instead
        )
    elif optimizer_name == "SLSQP":
        optimizer = SLSQP(
            maxiter=max_iter,
            ftol=energy_tol,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Use COBYLA, L_BFGS_B, SPSA, or SLSQP")

    # Setup VQE with Aer backend (GPU or multi-threaded CPU)
    # Check environment variable for device preference
    import os
    device = os.environ.get('QISKIT_DEVICE', 'CPU').upper()

    backend = None

    if device == 'GPU':
        # GPU backend for 8-12x speedup on CUDA-enabled instances
        try:
            backend = AerSimulator(
                method='statevector',
                device='GPU',
                precision='single'  # Single precision for 2x faster
            )
            if vqe_args.verbose >= 1:
                print("  Using Aer GPU backend")
        except Exception as exc:
            if vqe_args.verbose >= 1:
                print(f"  GPU backend unavailable ({exc}); falling back to CPU")
            device = 'CPU'

    if backend is None:
        # Multi-threaded CPU backend for 3-4x speedup
        max_threads = int(os.environ.get('OMP_NUM_THREADS', '8'))
        backend = AerSimulator(
            method='statevector',
            device='CPU',
            max_parallel_threads=max_threads
        )
        if vqe_args.verbose >= 1:
            print(f"  Using Aer CPU backend with {max_threads} threads")

    t_after_backend = perf_counter()
    phase_timings["backend_setup"] = t_after_backend - t_after_initial

    rng = np.random.default_rng(vqe_args.random_seed)
    progress_bar_enabled = vqe_args.show_progress_bar and max_iter > 0
    progress_bar_width = 30
    callback_needed = (
        vqe_args.track_iteration_history
        or vqe_args.track_density_matrices
        or vqe_args.verbose >= 2
        or progress_bar_enabled
    )
    interval = max(vqe_args.density_sample_interval, 1)

    def _build_param_dict(values: Any) -> dict[Parameter, float] | None:
        try:
            vector = np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return None
        if vector.size < num_parameters:
            return None
        if vector.size > num_parameters:
            vector = vector[:num_parameters]
        return {
            ordered_parameters[idx]: float(vector[idx])
            for idx in range(num_parameters)
        }

    def execute_vqe(initial_point: np.ndarray, run_index: int, label: str) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        prev_energy: float | None = None
        prev_rdm1: ndarray | None = None
        callback_start_time = 0.0
        last_callback_time = 0.0
        progress_last_eval = -1
        progress_bar_drawn = False

        def render_progress(eval_count: int) -> None:
            nonlocal progress_last_eval, progress_bar_drawn
            if not progress_bar_enabled:
                return
            eval_int = int(eval_count)
            if eval_int == progress_last_eval:
                return
            progress_last_eval = eval_int
            clamped_eval = min(max(eval_int, 0), max_iter)
            fraction = clamped_eval / max_iter if max_iter else 1.0
            filled = min(progress_bar_width, int(fraction * progress_bar_width))
            bar = "#" * filled + "-" * (progress_bar_width - filled)
            print(
                f"  Progress [{bar}] {fraction * 100:6.2f}% ({clamped_eval}/{max_iter})",
                end="",
                flush=True,
            )
            progress_bar_drawn = True

        interval_local = interval

        def vqe_callback(eval_count: int, parameters: np.ndarray, mean: float, metadata: Any):  # type: ignore[override]
            nonlocal prev_energy, prev_rdm1, last_callback_time, callback_start_time

            energy_raw = float(np.real(mean))
            energy_total = energy_raw + core_energy
            record: dict[str, Any] = {
                "eval_count": int(eval_count),
                "energy": energy_total,
                "raw_energy": energy_raw,
            }

            current_time = perf_counter()
            if callback_start_time == 0.0:
                callback_start_time = current_time
            if last_callback_time == 0.0:
                last_callback_time = current_time
            record["elapsed_time"] = current_time - callback_start_time
            record["delta_time"] = current_time - last_callback_time
            last_callback_time = current_time

            std_dev_val: float | None = None
            if metadata is not None:
                if isinstance(metadata, dict):
                    variance = metadata.get("variance") or metadata.get("variances")
                    if variance is not None:
                        if isinstance(variance, (list, tuple, np.ndarray)):
                            if len(variance) > 0:
                                var_value = float(variance[0])
                                std_dev_val = float(np.sqrt(max(var_value, 0.0)))
                        else:
                            var_value = float(variance)
                            std_dev_val = float(np.sqrt(max(var_value, 0.0)))
                    std_dev_candidate = metadata.get("stddev") or metadata.get("standard_error")
                    if std_dev_candidate is not None and std_dev_val is None:
                        if isinstance(std_dev_candidate, (list, tuple, np.ndarray)):
                            if len(std_dev_candidate) > 0:
                                std_dev_val = float(std_dev_candidate[0])
                        else:
                            std_dev_val = float(std_dev_candidate)
                elif isinstance(metadata, (float, int, np.floating)):
                    std_dev_val = float(metadata)

            if std_dev_val is not None:
                record["stddev"] = std_dev_val

            delta_energy = energy_total - prev_energy if prev_energy is not None else None
            record["delta_energy"] = delta_energy
            prev_energy = energy_total

            if progress_bar_enabled:
                render_progress(eval_count)

            if vqe_args.track_density_matrices:
                compute_density = (eval_count % interval_local == 0) or prev_rdm1 is None
                if compute_density:
                    param_binding = _build_param_dict(parameters)
                    if param_binding is not None:
                        density_t0 = perf_counter()
                        bound_circuit = ansatz.assign_parameters(param_binding)
                        statevector_iter = Statevector(bound_circuit)
                        rdm1_iter = compute_rdms_from_statevector(statevector_iter, norb, nelec)
                        record["rdm1_trace"] = float(np.trace(rdm1_iter))
                        if prev_rdm1 is not None:
                            record["rdm1_delta"] = float(np.linalg.norm(rdm1_iter - prev_rdm1))
                        else:
                            record["rdm1_delta"] = None
                        record["rdm1_matrix"] = rdm1_iter
                        prev_rdm1 = rdm1_iter
                        record["density_time"] = perf_counter() - density_t0
                    else:
                        record["rdm1_trace"] = None
                        record["rdm1_delta"] = None
                        record["rdm1_matrix"] = None
                        record["density_time"] = None
                else:
                    record["rdm1_trace"] = None
                    record["rdm1_delta"] = None
                    record["rdm1_matrix"] = None
                    record["density_time"] = None
            else:
                record["density_time"] = None

            records.append(record)

            if vqe_args.verbose >= 2:
                delta_str = (
                    f" ΔE={delta_energy:+.3e}" if delta_energy is not None else ""
                )
                std_str = (
                    f" σ={record['stddev']:.2e}" if record.get("stddev") is not None else ""
                )
                density_str = ""
                if vqe_args.track_density_matrices:
                    rdm_delta = record.get("rdm1_delta")
                    if rdm_delta is not None:
                        density_str = f" Δ||ρ||={rdm_delta:+.3e}"
                print(
                    f"    iter {eval_count:>3}: E={energy_total:.10f} Ha{delta_str}{std_str}{density_str}"
                )

        if vqe_args.verbose >= 1:
            print(f"  VQE restart {run_index + 1} ({label})", flush=True)

        def build_vqe(selected_backend: AerSimulator, init_point: np.ndarray) -> QiskitVQE:
            estimator_local = BackendEstimatorV2(backend=selected_backend)
            vqe_kwargs: dict[str, Any] = {"initial_point": init_point}
            if callback_needed:
                vqe_kwargs["callback"] = vqe_callback
            return QiskitVQE(estimator_local, ansatz, optimizer, **vqe_kwargs)

        vqe = build_vqe(backend, initial_point)

        if vqe_args.verbose >= 2:
            print(f"  Running VQE optimization...")

        optimization_start = perf_counter()
        pre_opt = optimization_start - t_after_backend

        try:
            result = vqe.compute_minimum_eigenvalue(qubit_hamiltonian)
        except AlgorithmError as exc:
            if device == 'GPU':
                if vqe_args.verbose >= 1:
                    print(f"  GPU execution failed ({exc}); retrying on CPU backend")
                max_threads = int(os.environ.get('OMP_NUM_THREADS', '8'))
                cpu_backend = AerSimulator(
                    method='statevector',
                    device='CPU',
                    max_parallel_threads=max_threads
                )
                if vqe_args.verbose >= 1:
                    print(f"  Using Aer CPU backend with {max_threads} threads")
                vqe = build_vqe(cpu_backend, initial_point)
                result = vqe.compute_minimum_eigenvalue(qubit_hamiltonian)
            else:
                raise
        opt_end = perf_counter()
        optimizer_time = opt_end - optimization_start
        if progress_bar_enabled:
            render_progress(result.cost_function_evals)
            if progress_bar_drawn:
                print()

        optimal_energy = result.eigenvalue.real + core_energy
        optimal_params = np.asarray(result.optimal_point, dtype=float)

        if vqe_args.verbose >= 1:
            print(
                f"  VQE converged: E={optimal_energy:.8f}, "
                f"iterations={result.cost_function_evals}",
                flush=True,
            )

        final_param_binding = _build_param_dict(optimal_params)
        if final_param_binding is None:
            raise ValueError(
                "Failed to bind VQE optimal parameters to ansatz; "
                f"expected {num_parameters} values, "
                f"got {len(np.asarray(optimal_params).reshape(-1)) if optimal_params is not None else 0}."
            )
        bound_circuit = ansatz.assign_parameters(final_param_binding)
        statevector = Statevector(bound_circuit)

        if vqe_args.verbose >= 2:
            print(f"  Computing RDMs from statevector...")

        rdm_time_start = perf_counter()
        rdm1_local, rdm2_local = compute_rdms_from_statevector(statevector, norb, nelec)
        rdm_time_end = perf_counter()

        run_timings = {
            "pre_optimization": pre_opt,
            "optimizer": optimizer_time,
            "statevector_and_rdms": rdm_time_end - rdm_time_start,
        }

        return {
            "energy": float(optimal_energy),
            "raw_energy": float(result.eigenvalue.real),
            "params": optimal_params,
            "rdm1": rdm1_local,
            "rdm2": rdm2_local,
            "records": records,
            "iterations": int(result.cost_function_evals),
            "timings": run_timings,
        }

    candidate_points: list[tuple[str, np.ndarray]] = []
    stored_params = _vqe_state.fragment_params.get(frag_name)
    if vqe_args.warm_start and stored_params is not None:
        candidate_points.append(("warm", np.asarray(stored_params, dtype=float)))

    zero_point = np.zeros(num_parameters)
    if not candidate_points or not np.allclose(candidate_points[0][1], zero_point):
        candidate_points.append(("hf", zero_point))

    while len(candidate_points) < max(1, vqe_args.max_restarts):
        candidate_points.append(
            (
                f"random_{len(candidate_points)}",
                rng.uniform(-np.pi, np.pi, size=num_parameters),
            )
        )

    best_run: dict[str, Any] | None = None
    best_energy = float("inf")

    for idx, (label, init_point) in enumerate(candidate_points[: max(1, vqe_args.max_restarts)]):
        run_data = execute_vqe(init_point, idx, label)
        energy = run_data["energy"]
        if energy < best_energy:
            best_energy = energy
            best_run = run_data

        if idx > 0 and abs(best_energy - energy) < vqe_args.restart_energy_tol:
            if vqe_args.verbose >= 1:
                print(
                    f"  Restart improvement below {vqe_args.restart_energy_tol:.1e}; stopping restarts.",
                    flush=True,
                )
            break

    if best_run is None:
        raise RuntimeError("VQE failed to produce a valid result")

    optimal_energy = float(best_run["energy"])
    optimal_params = np.asarray(best_run["params"], dtype=float)
    rdm1 = best_run["rdm1"]
    rdm2 = best_run["rdm2"]
    callback_records = best_run["records"]

    timings = best_run["timings"]
    phase_timings["pre_optimization"] = timings["pre_optimization"]
    phase_timings["optimizer"] = timings["optimizer"]
    phase_timings["statevector_and_rdms"] = timings["statevector_and_rdms"]
    phase_timings["total"] = (
        phase_timings.get("load_hamiltonian", 0.0)
        + phase_timings.get("build_ansatz", 0.0)
        + phase_timings.get("initial_parameters", 0.0)
        + phase_timings.get("backend_setup", 0.0)
        + timings["pre_optimization"]
        + timings["optimizer"]
        + timings["statevector_and_rdms"]
    )
    phase_timings["optimizer_iterations"] = best_run["iterations"]

    if vqe_args.warm_start:
        _vqe_state.fragment_params[frag_name] = optimal_params

    if vqe_args.verbose >= 2:
        print(f"  RDM1 trace: {np.trace(rdm1):.6f} (expected: {nelec})")
        print(f"  RDM2 computed successfully")

    # Ensure final record reflects the converged solution
    if callback_needed:
        needs_final_record = True
        if callback_records:
            last_energy = callback_records[-1].get("energy")
            if last_energy is not None and np.isclose(last_energy, optimal_energy, atol=1e-12):
                needs_final_record = False
                callback_records[-1]["timings"] = phase_timings.copy()
        if needs_final_record:
            final_record: dict[str, Any] = {
                "eval_count": best_run["iterations"],
                "energy": optimal_energy,
                "raw_energy": float(best_run["raw_energy"]),
                "delta_energy": (
                    optimal_energy - callback_records[-1]["energy"]
                    if callback_records
                    else None
                ),
                "timings": phase_timings.copy(),
                "elapsed_time": phase_timings.get("total"),
                "delta_time": None,
            }
            if vqe_args.track_density_matrices:
                final_record["rdm1_trace"] = float(np.trace(rdm1))
                prev_matrix = None
                if callback_records:
                    prev_matrix = callback_records[-1].get("rdm1_matrix")
                if prev_matrix is not None:
                    final_record["rdm1_delta"] = float(np.linalg.norm(rdm1 - prev_matrix))
                else:
                    final_record["rdm1_delta"] = None
                final_record["density_time"] = phase_timings.get("statevector_and_rdms")
            else:
                final_record["rdm1_delta"] = None
                final_record["density_time"] = phase_timings.get("statevector_and_rdms")
            final_record["rdm1_matrix"] = rdm1
            callback_records.append(final_record)

    _vqe_state.fragment_timings[frag_name] = phase_timings.copy()

    # Cache diagnostics for downstream inspection
    if vqe_args.track_iteration_history:
        _vqe_state.fragment_iteration_history[frag_name] = [
            {
                key: (
                    value.copy()
                    if isinstance(value, np.ndarray)
                    else value.copy()
                    if isinstance(value, dict)
                    else value
                )
                for key, value in record.items()
            }
            for record in callback_records
        ]
    elif frag_name in _vqe_state.fragment_iteration_history:
        # Drop stale history if tracking disabled for this run
        _vqe_state.fragment_iteration_history.pop(frag_name, None)

    _vqe_state.fragment_energies[frag_name] = optimal_energy
    _vqe_state.fragment_rdm1[frag_name] = rdm1
    _vqe_state.fragment_rdm2[frag_name] = rdm2

    return rdm1, rdm2


def get_vqe_iteration_history(frag_name: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """
    Retrieve stored VQE iteration history.

    Parameters
    ----------
    frag_name : str | None
        Specific fragment name. If None, return history for all fragments.

    Returns
    -------
    dict
        Mapping of fragment name to list of iteration records.
    """
    if frag_name is not None:
        history = _vqe_state.fragment_iteration_history.get(frag_name, [])
        return {frag_name: deepcopy(history)}

    return {key: deepcopy(val) for key, val in _vqe_state.fragment_iteration_history.items()}


def get_vqe_fragment_observables(frag_name: str | None = None) -> dict[str, dict[str, Any]]:
    """
    Retrieve converged VQE observables for fragments.

    Parameters
    ----------
    frag_name : str | None
        Specific fragment name. If None, return data for all fragments.

    Returns
    -------
    dict
        Mapping of fragment name to observables (energy, RDM1, RDM2).
    """
    def _build_payload(name: str) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if name in _vqe_state.fragment_energies:
            payload["energy"] = _vqe_state.fragment_energies[name]
        if name in _vqe_state.fragment_rdm1:
            payload["rdm1"] = _vqe_state.fragment_rdm1[name].copy()
        if name in _vqe_state.fragment_rdm2:
            payload["rdm2"] = _vqe_state.fragment_rdm2[name].copy()
        if name in _vqe_state.fragment_timings:
            payload["timings"] = _vqe_state.fragment_timings[name].copy()
        return payload

    if frag_name is not None:
        return {frag_name: _build_payload(frag_name)}

    return {name: _build_payload(name) for name in _vqe_state.fragment_energies}
