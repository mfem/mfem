#!/usr/bin/env python3
"""
Analyze Schwarz preconditioner convergence on contact systems.

This script:
1. Loads J and D matrices from files
2. Forms the system matrix A = J^T D J
3. Implements PCG with additive and multiplicative Schwarz preconditioners
4. Plots convergence for comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.sparse as sp
from scipy.io import mmread
import argparse


def read_mfem_matrix(filename):
    """
    Read a matrix from MFEM .mat format.

    MFEM parallel format:
    - First line: row_start row_end col_start nnz
    - Subsequent lines: row col value (space-separated)

    Args:
        filename: Path to .mat file

    Returns:
        scipy sparse matrix
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Skip empty lines
    data_lines = [line.strip() for line in lines if line.strip()]

    # Parse header: row_start row_end col_start nnz
    header = data_lines[0].split()
    row_start = int(header[0])
    row_end = int(header[1])
    col_start = int(header[2])
    nnz = int(header[3])

    nrows_local = row_end - row_start

    # Parse entries: row col value
    rows = []
    cols = []
    data = []

    for line in data_lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            # MFEM outputs global row/col indices
            row = int(parts[0])
            col = int(parts[1])
            val = float(parts[2])

            rows.append(row)
            cols.append(col)
            data.append(val)

    # Determine matrix dimensions from actual data
    if len(rows) > 0:
        nrows = max(rows) + 1
        ncols = max(cols) + 1
    else:
        nrows = nrows_local
        ncols = col_start

    # Convert to scipy sparse matrix (CSR format)
    return sp.csr_matrix((data, (rows, cols)), shape=(nrows, ncols))


def build_subdomains_from_J(J, P, min_diag_value=0.0, D_diag=None, debug=False):
    """
    Build Schwarz subdomains from constraint Jacobian J.

    Each row of J corresponds to a constraint. The subdomain for that constraint
    includes all contact subspace DOFs (rows of P) that correspond to the
    full-space DOFs (columns) with non-zero entries in that J row.

    Args:
        J: Constraint Jacobian matrix (constraints x full DOFs)
        P: Transfer operator (contact subspace x full DOFs)
        min_diag_value: Minimal D diagonal value to include constraint
        D_diag: Diagonal of D matrix (for filtering)
        debug: Print debug information

    Returns:
        list of arrays: Each array contains contact subspace DOF indices for one subdomain
    """
    subdomains = []
    J_csr = sp.csr_matrix(J)
    P_csr = sp.csr_matrix(P)

    if debug:
        print(f"\nDEBUG: Building subdomains")
        print(f"  J shape: {J_csr.shape} (constraints x full DOFs)")
        print(f"  P shape: {P_csr.shape}")
        print(f"  P nnz: {P_csr.nnz}")

    # Build mapping: full DOF -> contact subspace DOFs
    # P is (full_dofs x contact_subspace), so P[i,j] != 0 means
    # full DOF i maps to contact subspace DOF j
    full_to_contact = {}
    for full_dof in range(P_csr.shape[0]):
        row_start = P_csr.indptr[full_dof]
        row_end = P_csr.indptr[full_dof + 1]
        contact_dofs = P_csr.indices[row_start:row_end]

        if len(contact_dofs) > 0:
            full_to_contact[full_dof] = list(contact_dofs)

    if debug:
        print(f"  Built mapping for {len(full_to_contact)} full DOFs -> contact DOFs")
        if len(full_to_contact) > 0:
            sample_full_dof = list(full_to_contact.keys())[0]
            print(f"  Example: full DOF {sample_full_dof} -> contact DOFs {full_to_contact[sample_full_dof]}")
        print(f"  Expected contact subspace size: {P_csr.shape[1]}")

    for i in range(J_csr.shape[0]):
        # Skip if D diagonal value is below threshold
        if D_diag is not None and min_diag_value > 0.0:
            if i < len(D_diag) and D_diag[i] < min_diag_value:
                continue

        # Get full-space DOF indices with non-zero entries in this constraint
        row_start = J_csr.indptr[i]
        row_end = J_csr.indptr[i + 1]
        full_dofs = J_csr.indices[row_start:row_end]

        if debug and i < 2:  # Debug first two constraints
            print(f"\n  Constraint {i}:")
            print(f"    Full DOFs in J row: {list(full_dofs)[:10]}... (total: {len(full_dofs)})")

        # Map to contact subspace DOFs
        contact_dofs = set()
        for full_dof in full_dofs:
            if full_dof in full_to_contact:
                contact_dofs.update(full_to_contact[full_dof])

        if debug and i < 2:
            print(f"    Mapped to contact DOFs: {sorted(list(contact_dofs))[:10]}... (total: {len(contact_dofs)})")

        if len(contact_dofs) > 0:
            subdomains.append(np.array(sorted(contact_dofs)))

    if debug:
        print(f"\n  Total subdomains created: {len(subdomains)}")

    return subdomains


class SchwarzPreconditioner:
    """Schwarz preconditioner (additive or multiplicative)."""

    def __init__(self, A, subdomains, variant='additive', relax_weight=1.0, no_scaling=False, uniform_weight=None, symmetrized=False):
        """
        Initialize Schwarz preconditioner.

        Args:
            A: System matrix (sparse)
            subdomains: List of DOF index arrays for each subdomain
            variant: 'additive' or 'multiplicative'
            relax_weight: Relaxation weight (damping parameter)
            no_scaling: If True, disable per-DOF scaling (set all scales to 1.0)
            uniform_weight: If specified, set all scales to this uniform value (overrides other scaling)
            symmetrized: If True, apply both corrections for symmetrized additive Schwarz
        """
        self.A = sp.csr_matrix(A)
        self.subdomains = subdomains
        self.variant = variant
        self.relax_weight = relax_weight
        self.symmetrized = symmetrized
        self.n = A.shape[0]

        # Compute scale for each DOF following HYPRE implementation
        # scale[dof] = relax_weight / (number of subdomains containing dof)
        subdomain_count = np.zeros(self.n)
        for subdomain in subdomains:
            subdomain_count[subdomain] += 1.0

        self.scale = np.zeros(self.n)
        if uniform_weight is not None:
            # Uniform scaling: set all scales to the specified value
            self.scale = np.full(self.n, uniform_weight)
        elif no_scaling:
            # Disable per-DOF scaling: set all scales to 1.0
            self.scale = np.ones(self.n)
        else:
            # Per-DOF scaling: scale[dof] = relax_weight / (number of subdomains containing dof)
            for i in range(self.n):
                if subdomain_count[i] > 0:
                    self.scale[i] = relax_weight / subdomain_count[i]

        # Precompute subdomain solvers
        self.subdomain_solvers = []
        num_cholesky = 0
        num_lu = 0
        num_failed = 0

        for i, subdomain_dofs in enumerate(subdomains):
            # Extract subdomain matrix
            A_sub = self.A[np.ix_(subdomain_dofs, subdomain_dofs)].toarray()

            # Check condition number
            try:
                cond = np.linalg.cond(A_sub)
                if cond > 1e12:
                    print(f"  Warning: Subdomain {i} has high condition number: {cond:.2e}")
            except:
                pass

            # Factor the subdomain matrix (Cholesky for SPD)
            try:
                L_sub = np.linalg.cholesky(A_sub)
                self.subdomain_solvers.append(('cholesky', subdomain_dofs, L_sub))
                num_cholesky += 1
            except np.linalg.LinAlgError as e:
                # Fall back to LU if Cholesky fails
                try:
                    from scipy.linalg import lu_factor
                    lu_sub = lu_factor(A_sub)
                    self.subdomain_solvers.append(('lu', subdomain_dofs, lu_sub))
                    num_lu += 1
                except Exception as e2:
                    # Skip this subdomain if factorization fails
                    num_failed += 1
                    print(f"  Warning: Subdomain {i} factorization failed: {e2}")
                    continue

        print(f"  Factorization: {num_cholesky} Cholesky, {num_lu} LU, {num_failed} failed")

    def apply(self, r):
        """
        Apply Schwarz preconditioner: z = M^{-1} r

        For symmetrized additive Schwarz, applies both:
        - First correction: ∑_k R_k^T D_k^{-1} (R_k A R_k^T)^{-1} R_k
        - Second correction: ∑_k R_k^T (R_k A R_k^T)^{-1} D_k^{-1} R_k

        Args:
            r: Residual vector

        Returns:
            z: Preconditioned residual
        """
        z = np.zeros_like(r)

        if self.variant == 'additive':
            # First correction: ∑_k R_k^T D_k^{-1} (R_k A R_k^T)^{-1} R_k
            # (standard additive Schwarz with per-DOF scaling applied after subdomain solve)
            for solver_info in self.subdomain_solvers:
                solver_type, dofs, factor = solver_info

                # Restrict residual to subdomain
                r_sub = r[dofs]

                # Solve subdomain system
                if solver_type == 'cholesky':
                    # L L^T z_sub = r_sub
                    z_sub = np.linalg.solve(factor.T, np.linalg.solve(factor, r_sub))
                else:  # lu
                    from scipy.linalg import lu_solve
                    z_sub = lu_solve(factor, r_sub)

                # Apply scaling and accumulate
                z[dofs] += self.scale[dofs] * z_sub * (0.5 if self.symmetrized else 1.0)

            # If symmetrized, add second correction: ∑_k R_k^T (R_k A R_k^T)^{-1} D_k^{-1} R_k
            # (per-DOF scaling applied before subdomain solve)
            if self.symmetrized:
                for solver_info in self.subdomain_solvers:
                    solver_type, dofs, factor = solver_info

                    # Apply scaling first, then restrict to subdomain
                    r_scaled = self.scale * r 
                    r_sub = r_scaled[dofs]

                    # Solve subdomain system
                    if solver_type == 'cholesky':
                        z_sub = np.linalg.solve(factor.T, np.linalg.solve(factor, r_sub))
                    else:  # lu
                        from scipy.linalg import lu_solve
                        z_sub = lu_solve(factor, r_sub)

                    # Accumulate (no additional scaling)
                    z[dofs] += z_sub * 0.5

        else:  # multiplicative
            # Multiplicative Schwarz following HYPRE implementation
            # Forward-backward symmetric sweep with per-DOF scaling
            z = np.zeros_like(r)

            # Forward sweep
            for solver_info in self.subdomain_solvers:
                solver_type, dofs, factor = solver_info

                # Compute residual on subdomain: aux = r - A * z
                aux = r[dofs] - (self.A @ z)[dofs]

                # Solve subdomain system: A_sub^{-1} * aux
                if solver_type == 'cholesky':
                    correction = np.linalg.solve(factor.T, np.linalg.solve(factor, aux))
                else:  # lu
                    from scipy.linalg import lu_solve
                    correction = lu_solve(factor, aux)

                # Update solution with per-DOF scaling
                z[dofs] += correction

            # Backward sweep
            for solver_info in reversed(self.subdomain_solvers):
                solver_type, dofs, factor = solver_info

                # Compute residual on subdomain: aux = r - A * z
                aux = r[dofs] - (self.A @ z)[dofs]

                # Solve subdomain system: A_sub^{-1} * aux
                if solver_type == 'cholesky':
                    correction = np.linalg.solve(factor.T, np.linalg.solve(factor, aux))
                else:  # lu
                    from scipy.linalg import lu_solve
                    correction = lu_solve(factor, aux)

                # Update solution with per-DOF scaling
                z[dofs] += correction

        return z


class IteratedSchwarzPreconditioner:
    """Wrapper that applies Schwarz preconditioner multiple times."""

    def __init__(self, base_precond, num_iters):
        """
        Initialize iterated Schwarz preconditioner.

        Args:
            base_precond: Base Schwarz preconditioner
            num_iters: Number of times to apply the preconditioner
        """
        self.base_precond = base_precond
        self.num_iters = num_iters

    def apply(self, r):
        """
        Apply Schwarz preconditioner multiple times: z = (M^{-1})^k r

        Args:
            r: Residual vector

        Returns:
            z: Preconditioned residual
        """
        z = r.copy()
        for i in range(self.num_iters):
            z = self.base_precond.apply(z)
        return z


def pcg_solve(A, b, precond=None, max_iter=1000, tol=1e-10, verbose=False):
    """
    Preconditioned Conjugate Gradient solver.

    Args:
        A: System matrix (sparse)
        b: Right-hand side
        precond: Preconditioner object with apply() method
        max_iter: Maximum iterations
        tol: Convergence tolerance
        verbose: Print detailed iteration info

    Returns:
        x: Solution vector
        residuals: List of residual norms at each iteration
    """
    n = len(b)
    x = np.zeros(n)
    r = b - A @ x

    residuals = [np.linalg.norm(r)]

    if precond is not None:
        z = precond.apply(r)
    else:
        z = r.copy()

    p = z.copy()
    rz = np.dot(r, z)

    for k in range(max_iter):
        Ap = A @ p
        pAp = np.dot(p, Ap)
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        res_norm = np.linalg.norm(r)
        residuals.append(res_norm)

        if verbose and k % 10 == 0:
            print(f"    Iter {k}: residual = {res_norm:.6e}, alpha = {alpha:.6e}, pAp = {pAp:.6e}")

        if np.isnan(res_norm) or np.isinf(res_norm):
            print(f"  PCG diverged at iteration {k+1}: residual = {res_norm}")
            break

        if res_norm < tol:
            print(f"  Converged in {k+1} iterations")
            break

        if precond is not None:
            z = precond.apply(r)
        else:
            z = r.copy()

        rz_new = np.dot(r, z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return x, residuals


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Schwarz preconditioner convergence',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mats-dir', type=str, default='mats',
                        help='Directory containing matrix files')
    parser.add_argument('--iteration', type=int, default=1,
                        help='Iteration number to analyze')
    parser.add_argument('--min-diag', type=float, default=0.0,
                        help='Minimum D diagonal value to include')
    parser.add_argument('--max-iter', type=int, default=500,
                        help='Maximum PCG iterations')
    parser.add_argument('--tol', type=float, default=1e-10,
                        help='PCG convergence tolerance')
    parser.add_argument('--relax-weight', type=float, default=1.0,
                        help='Relaxation weight for Schwarz preconditioner')
    parser.add_argument('--no-scaling', action='store_true',
                        help='Disable per-DOF scaling (set all scales to 1.0)')
    parser.add_argument('--uniform-weight', type=float, default=None,
                        help='Add curve for additive Schwarz with uniform weight (set all scales to this value)')
    parser.add_argument('--schwarz-iters', type=int, default=0,
                        help='Number of Schwarz iterations to apply in preconditioner (0 = disable iterated variants)')
    args = parser.parse_args()

    mats_dir = Path(args.mats_dir)

    # Construct filenames
    j_file = mats_dir / f"J_matrix_iter_{args.iteration}.mat.00000"
    d_file = mats_dir / f"D_matrix_iter_{args.iteration}.mat.00000"
    p_file = mats_dir / f"P_matrix_iter_{args.iteration}.mat.00000"
    ptap_file = mats_dir / f"PTAP_matrix_iter_{args.iteration}.mat.00000"

    print("=" * 70)
    print("Schwarz Preconditioner Convergence Analysis")
    print("=" * 70)
    print(f"Matrix directory: {mats_dir}")
    print(f"Iteration: {args.iteration}")
    print(f"Min D diagonal: {args.min_diag}")
    print()

    # Load matrices
    print("Loading matrices...")
    if not j_file.exists():
        print(f"Error: J matrix file not found: {j_file}")
        return
    if not d_file.exists():
        print(f"Error: D matrix file not found: {d_file}")
        return
    if not p_file.exists():
        print(f"Error: P matrix file not found: {p_file}")
        return
    if not ptap_file.exists():
        print(f"Error: PTAP matrix file not found: {ptap_file}")
        return

    J = read_mfem_matrix(j_file)
    D = read_mfem_matrix(d_file)
    P = read_mfem_matrix(p_file)
    PTAP = read_mfem_matrix(ptap_file)

    print(f"J shape: {J.shape} (constraints x DOFs)")
    print(f"D shape: {D.shape}")
    print(f"P shape: {P.shape} (contact subspace x DOFs)")
    print(f"PTAP shape: {PTAP.shape} (projected system)")
    print()

    # Extract D diagonal
    D_csr = sp.csr_matrix(D)
    D_diag = np.array(D_csr.diagonal())

    print(f"D diagonal statistics:")
    print(f"  Min: {D_diag.min():.6e}")
    print(f"  Max: {D_diag.max():.6e}")
    print(f"  Mean: {D_diag.mean():.6e}")
    print()

    # Use PTAP as the system matrix
    print("Using PTAP as system matrix (P^T * (Huu + J^T D J) * P)...")
    A = sp.csr_matrix(PTAP)

    print(f"A shape: {A.shape}")
    print(f"A nnz: {A.nnz}")
    print()

    # Build subdomains
    print("Building Schwarz subdomains...")
    subdomains = build_subdomains_from_J(J, P, args.min_diag, D_diag, debug=False)

    num_subdomains = len(subdomains)
    subdomain_sizes = [len(s) for s in subdomains]

    # Check overlap
    all_dofs = set()
    for subdomain in subdomains:
        all_dofs.update(subdomain)
    total_dof_count = sum(subdomain_sizes)
    unique_dofs = len(all_dofs)
    overlap_ratio = (total_dof_count - unique_dofs) / unique_dofs if unique_dofs > 0 else 0

    print(f"Number of subdomains: {num_subdomains}")
    if num_subdomains > 0:
        print(f"Subdomain sizes: min={min(subdomain_sizes)}, "
              f"max={max(subdomain_sizes)}, mean={np.mean(subdomain_sizes):.1f}")
        print(f"DOF coverage: {unique_dofs}/{A.shape[0]} unique DOFs covered")
        print(f"Overlap: {total_dof_count} total DOF instances, ratio={overlap_ratio:.2f}")

        uncovered = set(range(A.shape[0])) - all_dofs
        if len(uncovered) > 0:
            print(f"  WARNING: {len(uncovered)} DOFs are not covered by any subdomain")
            print(f"  This means the preconditioner will leave these DOFs unchanged")
    print()

    if num_subdomains == 0:
        print("Error: No subdomains created. Check min_diag threshold.")
        return

    # Create right-hand side (random vector)
    np.random.seed(42)
    n = A.shape[0]
    b = np.random.randn(n)
    b = b / np.linalg.norm(b)

    results = {}

    # 1. No preconditioner (baseline)
    print("Running PCG without preconditioner...")
    x_none, res_none = pcg_solve(A, b, precond=None,
                                  max_iter=args.max_iter, tol=args.tol)
    results['No preconditioner'] = res_none
    print()

    # 2. Additive Schwarz (with scaling)
    print("Building additive Schwarz preconditioner (with scaling)...")
    precond_add = SchwarzPreconditioner(A, subdomains, variant='additive', relax_weight=args.relax_weight, no_scaling=False)
    print(f"Initialized {len(precond_add.subdomain_solvers)} subdomain solvers")

    # Test preconditioner on initial residual
    print("Testing additive preconditioner...")
    test_r = np.random.randn(A.shape[0])
    test_r = test_r / np.linalg.norm(test_r)
    test_z = precond_add.apply(test_r)
    print(f"  ||r|| = {np.linalg.norm(test_r):.6e}")
    print(f"  ||z|| = {np.linalg.norm(test_z):.6e}")
    print(f"  r^T z = {np.dot(test_r, test_z):.6e}")

    print("Running PCG with additive Schwarz (with scaling)...")
    x_add, res_add = pcg_solve(A, b, precond=precond_add,
                                max_iter=args.max_iter, tol=args.tol, verbose=True)
    results['Additive Schwarz'] = res_add
    print()

    # 3. Additive Schwarz (symmetrized with scaling)
    print("Building additive Schwarz preconditioner (symmetrized with scaling)...")
    precond_add_sym = SchwarzPreconditioner(A, subdomains, variant='additive', relax_weight=args.relax_weight, no_scaling=False, symmetrized=True)
    print(f"Initialized {len(precond_add_sym.subdomain_solvers)} subdomain solvers")

    print("Running PCG with additive Schwarz (symmetrized with scaling)...")
    x_add_sym, res_add_sym = pcg_solve(A, b, precond=precond_add_sym,
                                       max_iter=args.max_iter, tol=args.tol, verbose=True)
    results['Additive Schwarz (symmetrized)'] = res_add_sym
    print()

    # 4. Additive Schwarz (without scaling)
    print("Building additive Schwarz preconditioner (without scaling)...")
    precond_add_no_scale = SchwarzPreconditioner(A, subdomains, variant='additive', relax_weight=args.relax_weight, no_scaling=True)
    print(f"Initialized {len(precond_add_no_scale.subdomain_solvers)} subdomain solvers")

    print("Running PCG with additive Schwarz (without scaling)...")
    x_add_no_scale, res_add_no_scale = pcg_solve(A, b, precond=precond_add_no_scale,
                                                   max_iter=args.max_iter, tol=args.tol, verbose=True)
    results['Additive Schwarz (no scaling)'] = res_add_no_scale
    print()

    # 4a. Iterated Additive Schwarz (without scaling) - only if schwarz_iters > 0
    if args.schwarz_iters > 0:
        print(f"Building iterated additive Schwarz preconditioner (without scaling, {args.schwarz_iters} iters)...")
        precond_add_no_scale_iterated = IteratedSchwarzPreconditioner(precond_add_no_scale, args.schwarz_iters)

        print(f"Running PCG with iterated additive Schwarz (without scaling, {args.schwarz_iters} iters)...")
        x_add_no_scale_iter, res_add_no_scale_iter = pcg_solve(A, b, precond=precond_add_no_scale_iterated,
                                                                 max_iter=args.max_iter, tol=args.tol, verbose=True)
        results[f'Additive Schwarz (no scaling, {args.schwarz_iters} iters)'] = res_add_no_scale_iter
        print()

    # 4. Additive Schwarz with uniform weight (if specified)
    if args.uniform_weight is not None:
        print(f"Building additive Schwarz preconditioner (uniform weight = {args.uniform_weight})...")
        precond_uniform = SchwarzPreconditioner(A, subdomains, variant='additive',
                                                relax_weight=args.relax_weight,
                                                uniform_weight=args.uniform_weight)
        print(f"Initialized {len(precond_uniform.subdomain_solvers)} subdomain solvers")

        print(f"Running PCG with additive Schwarz (uniform weight = {args.uniform_weight})...")
        x_uniform, res_uniform = pcg_solve(A, b, precond=precond_uniform,
                                           max_iter=args.max_iter, tol=args.tol, verbose=True)
        results[f'Additive Schwarz (uniform={args.uniform_weight})'] = res_uniform
        print()

    # 5. Multiplicative Schwarz
    print("Building multiplicative Schwarz preconditioner...")
    precond_mult = SchwarzPreconditioner(A, subdomains, variant='multiplicative', relax_weight=args.relax_weight, no_scaling=False)
    print(f"Initialized {len(precond_mult.subdomain_solvers)} subdomain solvers")

    # Test preconditioner on initial residual
    print("Testing preconditioner...")
    test_r = np.random.randn(n)
    test_r = test_r / np.linalg.norm(test_r)
    test_z = precond_mult.apply(test_r)
    print(f"  ||r|| = {np.linalg.norm(test_r):.6e}")
    print(f"  ||z|| = {np.linalg.norm(test_z):.6e}")
    print(f"  ||A*z|| = {np.linalg.norm(A @ test_z):.6e}")
    print(f"  r^T z = {np.dot(test_r, test_z):.6e}")

    print("Running PCG with multiplicative Schwarz...")
    x_mult, res_mult = pcg_solve(A, b, precond=precond_mult,
                                  max_iter=args.max_iter, tol=args.tol, verbose=True)
    results['Multiplicative Schwarz'] = res_mult
    print()

    # 5a. Iterated Multiplicative Schwarz - only if schwarz_iters > 0
    if args.schwarz_iters > 0:
        print(f"Building iterated multiplicative Schwarz preconditioner ({args.schwarz_iters} iters)...")
        precond_mult_iterated = IteratedSchwarzPreconditioner(precond_mult, args.schwarz_iters)

        print(f"Running PCG with iterated multiplicative Schwarz ({args.schwarz_iters} iters)...")
        x_mult_iter, res_mult_iter = pcg_solve(A, b, precond=precond_mult_iterated,
                                                max_iter=args.max_iter, tol=args.tol, verbose=True)
        results[f'Multiplicative Schwarz ({args.schwarz_iters} iters)'] = res_mult_iter
        print()

    # Plot convergence
    print("=" * 70)
    print("Plotting convergence...")
    print("=" * 70)

    plt.figure(figsize=(12, 8))

    colors = {'No preconditioner': 'gray',
              'Additive Schwarz': 'blue',
              'Additive Schwarz (symmetrized)': 'purple',
              'Additive Schwarz (no scaling)': 'cyan',
              'Multiplicative Schwarz': 'red'}
    markers = {'No preconditioner': 'o',
               'Additive Schwarz': 's',
               'Additive Schwarz (symmetrized)': 'p',
               'Additive Schwarz (no scaling)': 'D',
               'Multiplicative Schwarz': '^'}

    # Add color and marker for iterated variants if present
    if args.schwarz_iters > 0:
        colors[f'Additive Schwarz (no scaling, {args.schwarz_iters} iters)'] = 'lightblue'
        markers[f'Additive Schwarz (no scaling, {args.schwarz_iters} iters)'] = 'x'

        colors[f'Multiplicative Schwarz ({args.schwarz_iters} iters)'] = 'orange'
        markers[f'Multiplicative Schwarz ({args.schwarz_iters} iters)'] = '*'

    # Add color and marker for uniform weight curve if present
    if args.uniform_weight is not None:
        uniform_key = f'Additive Schwarz (uniform={args.uniform_weight})'
        colors[uniform_key] = 'green'
        markers[uniform_key] = 'v'

    for name, residuals in results.items():
        iters = range(len(residuals))
        plt.semilogy(iters, residuals,
                     label=name,
                     color=colors[name],
                     marker=markers[name],
                     markevery=max(1, len(residuals)//20),
                     linewidth=2,
                     markersize=6)

        print(f"{name}:")
        print(f"  Iterations: {len(residuals)-1}")
        print(f"  Final residual: {residuals[-1]:.6e}")
        print()

    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('Residual Norm', fontsize=13)
    plt.title(f'PCG Convergence Comparison (iter={args.iteration}, min_diag={args.min_diag})',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, which='both', linestyle='--')
    plt.tight_layout()

    # Save plot
    output_file = f'schwarz_convergence_iter{args.iteration}_mindiag{args.min_diag:.0e}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    plt.show()


if __name__ == "__main__":
    main()
