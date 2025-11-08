# Author(s): Oinam Romesh Meitei


import logging
import warnings

import numpy as np
from attrs import Factory, define
from numpy import array, float64
from scipy import optimize

from quemb.kbe.pfrag import Frags as pFrags
from quemb.molbe.be_parallel import be_func_parallel
from quemb.molbe.pfrag import Frags
from quemb.molbe.solver import Solvers, UserSolverArgs, be_func, solve_error
from quemb.shared.external.optqn import FrankQN
from quemb.shared.helper import Timer
from quemb.shared.manage_scratch import WorkDir
from quemb.shared.typing import Matrix, Vector

logger = logging.getLogger(__name__)


@define
class BEOPT:
    """Perform BE optimization.

    Implements optimization algorithms for bootstrap optimizations, namely,
    chemical potential optimization and density matching. The main technique used in
    the optimization is a Quasi-Newton method. It interface to external
    (adapted version) module originally written by Hong-Zhou Ye.

    Parameters
    ----------
    pot :
       List of initial BE potentials. The last element is for the global
       chemical potential.
    Fobjs :
       Fragment object
    Nocc :
       No. of occupied orbitals for the full system.
    enuc :
       Nuclear component of the energy.
    scratch_dir :
        Scratch directory
    solver :
       High-level solver in bootstrap embedding. 'MP2', 'CCSD', 'FCI' are supported.
       Selected CI versions,
       'HCI', 'SHCI', & 'SCI' are also supported. Defaults to 'CCSD'
    only_chem :
       Whether to perform chemical potential optimization only.
       Refer to bootstrap embedding literatures.
    nproc :
       Total number of processors assigned for the optimization. Defaults to 1.
       When nproc > 1, Python multithreading
       is invoked.
    ompnum :
       If nproc > 1, ompnum sets the number of cores for OpenMP parallelization.
       Defaults to 4
    max_space :
       Maximum number of bootstrap optimizaiton steps, after which the optimization
       is called converged.
    conv_tol :
       Convergence criteria for optimization. Defaults to 1e-6
    ebe_hf :
       Hartree-Fock energy. Defaults to 0.0
    """

    pot: list[float]
    Fobjs: list[Frags] | list[pFrags]
    Nocc: int
    enuc: float
    scratch_dir: WorkDir
    solver: Solvers = "CCSD"
    nproc: int = 1
    ompnum: int = 4
    only_chem: bool = False
    use_cumulant: bool = True

    max_space: int = 500
    conv_tol: float = 1.0e-6
    relax_density: bool = False
    ebe_hf: float = 0.0

    iter: int = 0
    err: float = 0.0
    Ebe: Matrix[float64] = Factory(lambda: array([[0.0]]))

    solver_args: UserSolverArgs | None = None
    log_density_iterations: bool = False

    def objfunc(self, xk: list[float]) -> Vector[float64]:
        """
        Computes error vectors, RMS error, and BE energies.

        If nproc (set in initialization) > 1, a multithreaded function is called to
        perform high-level computations.

        Parameters
        ----------
        xk :
            Current potentials in the BE optimization.

        Returns
        -------
        list
            Error vectors.
        """

        # Choose the appropriate function based on the number of processors
        if self.nproc == 1:
            err_, errvec_, ebe_ = be_func(
                xk,
                self.Fobjs,
                self.Nocc,
                self.solver,
                self.enuc,
                only_chem=self.only_chem,
                relax_density=self.relax_density,
                scratch_dir=self.scratch_dir,
                solver_args=self.solver_args,
                use_cumulant=self.use_cumulant,
                eeval=True,
                return_vec=True,
            )
        else:
            err_, errvec_, ebe_ = be_func_parallel(
                xk,
                self.Fobjs,
                self.Nocc,
                self.solver,
                self.enuc,
                only_chem=self.only_chem,
                nproc=self.nproc,
                ompnum=self.ompnum,
                relax_density=self.relax_density,
                scratch_dir=self.scratch_dir,
                solver_args=self.solver_args,
                use_cumulant=self.use_cumulant,
                eeval=True,
                return_vec=True,
            )

        if self.log_density_iterations:
            print("    Row density matrices (current iteration):", flush=True)
            for findx, fobj in enumerate(self.Fobjs):
                rdm = getattr(fobj, "_rdm1", None)
                if rdm is None:
                    continue
                frag_name = getattr(fobj, "dname", f"frag_{findx}")
                rdm_array = np.array(rdm, dtype=float)
                print(f"      Fragment {frag_name}:", flush=True)
                print(
                    np.array2string(rdm_array, precision=6, suppress_small=True),
                    flush=True,
                )
                row_sums = np.sum(rdm_array, axis=1)
                print(
                    "        row sums: "
                    + np.array2string(row_sums, precision=6, suppress_small=True),
                    flush=True,
                )
            try:
                _, _, components = solve_error(
                    self.Fobjs,
                    self.Nocc,
                    only_chem=self.only_chem,
                    return_components=True,
                )
            except Exception as exc:  # pragma: no cover - diagnostics only
                print(f"    Warning: unable to compute density components ({exc})", flush=True)
                components = None
            if components:
                print("    Density matching components:", flush=True)
                def _fmt(value):
                    if value is None:
                        return "n/a"
                    if isinstance(value, (float, np.floating)):
                        return f"{float(value):+.6f}"
                    return str(value)

                for comp in components:
                    ctype = comp.get("type", "edge")
                    diff = comp.get("difference")
                    frag = comp.get("fragment")
                    pair = comp.get("pair")
                    center_val = comp.get("center_value")
                    edge_val = comp.get("edge_value")
                    if ctype == "chemical_potential":
                        print(
                            "      chemical potential:" \
                            f" ρ_sum={_fmt(edge_val)} target={_fmt(center_val)} Δ={_fmt(diff)}",
                            flush=True,
                        )
                    else:
                        ref_frag = comp.get("reference_fragment")
                        ref_pair = comp.get("reference_pair")
                        print(
                            "      edge:" \
                            f" frag={frag} pair={pair} ρ_edge={_fmt(edge_val)} "
                            f"ref_frag={ref_frag} ref_pair={ref_pair} ρ_ref={_fmt(center_val)} Δ={_fmt(diff)}",
                            flush=True,
                        )

        # Update error and BE energy
        self.err = err_
        self.Ebe = ebe_
        return errvec_

    def optimize(self, method, J0=None, trust_region=False):
        """Main kernel to perform BE optimization

        Parameters
        ----------
        method : str
           High-level quantum chemistry method.
        J0 : list of list of float, optional
           Initial Jacobian
        trust_region : bool, optional
           Use trust-region based QN optimization, by default False
        """

        print("-----------------------------------------------------", flush=True)
        print("             Starting BE optimization ", flush=True)
        print("             Solver : ", self.solver, flush=True)
        if self.only_chem:
            print("             Chemical Potential Optimization", flush=True)
        print("-----------------------------------------------------", flush=True)
        print(flush=True)
        if method == "QN":
            step0_timer = Timer("Time to complete Iteration 0")
            print("-- Beginning optimization iteration ", self.iter, flush=True)

            # Initial step
            f0 = self.objfunc(self.pot)

            print(
                f"Error in density matching      :   {self.err:>2.4e}",
                flush=True,
            )
            print(flush=True)

            # Initialize the Quasi-Newton optimizer
            optQN = FrankQN(
                self.objfunc, array(self.pot), f0, J0, max_space=self.max_space
            )

            logger.info(f"Step 0 time: {step0_timer.str_elapsed()}")
            if self.err < self.conv_tol:
                print(flush=True)
                print("CONVERGED w/o Optimization Steps", flush=True)
                print(flush=True)
            else:
                # Perform optimization steps
                for iter_ in range(self.max_space):
                    iter_timer = Timer("Time to complete Iteration " + str(self.iter))
                    print("-- In iter ", self.iter, flush=True)
                    optQN.next_step(self.iter, trust_region=trust_region)
                    self.iter += 1
                    print(
                        f"Error in density matching      :   {self.err:>2.4e}",
                        flush=True,
                    )
                    logger.info(f"Iteration time: {iter_timer.str_elapsed()}")
                    if self.err < self.conv_tol:
                        print(flush=True)
                        print("CONVERGED", flush=True)
                        logger.info(
                            step0_timer.str_elapsed(
                                "Total time to complete BE optimization"
                            )
                        )
                        break
                if self.err >= self.conv_tol:
                    warnings.warn(f"BE DID NOT CONVERGE IN {self.max_space} STEPS")
        elif method == "SCIPY":
            # Use SciPy optimization as alternative to Broyden
            print("Using SciPy optimization (Powell's hybrid method)")
            
            def objective_func(x):
                """Objective function for SciPy root finding"""
                return self.objfunc(x.tolist())
            
            # Initial step to get error vector size
            f0 = self.objfunc(self.pot)
            print(f"Initial density matching error: {self.err:>2.4e}")
            
            if self.err < self.conv_tol:
                print("CONVERGED w/o Optimization Steps")
            else:
                # Use Powell's hybrid method (hybr) - very robust for ill-conditioned problems
                try:
                    result = optimize.root(
                        objective_func, 
                        array(self.pot), 
                        method='hybr',  # Powell's hybrid method
                        tol=self.conv_tol,
                        options={
                            'maxfev': self.max_space * len(self.pot),  # Max function evaluations
                            'diag': None,  # Let algorithm choose scaling
                        }
                    )
                    
                    if result.success:
                        print("CONVERGED with SciPy optimization")
                        print(f"Final density matching error: {self.err:>2.4e}")
                        print(f"Function evaluations: {result.nfev}")
                        # Update final potentials
                        self.pot = result.x.tolist()
                    else:
                        print(f"SciPy optimization failed: {result.message}")
                        warnings.warn("BE optimization with SciPy failed")
                        
                except Exception as e:
                    print(f"SciPy optimization error: {e}")
                    warnings.warn(f"BE optimization with SciPy failed: {e}")
        else:
            raise ValueError("This optimization method for BE is not supported")
