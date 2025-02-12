import pyomo.environ as pyo
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.special import legendre
from matplotlib.backends.backend_pdf import PdfPages
import os
import copy

# 1) Define the N Lagrange basis polynomials, L_j(x), j=0..N-1.
#    You can build them symbolically or load them if you have a direct formula.

def lobatto_nodes(N):
	roots = legendre(N-1).deriv().roots
	x = np.concatenate(([-1], roots, [1]))
	return np.sort(x)

def lagrange_basis_polynomial(j, x, gll_nodes):
    """
    Returns the L_j(x) value at x (assuming j-th Lagrange polynomial
    w.r.t. the given GLL nodes). For brevity, just illustrate a direct product form.
    """
    L = 1.0
    x_j = gll_nodes[j]
    for m, xm in enumerate(gll_nodes):
        if m != j:
            L *= (x - xm) / (x_j - xm)
    return L

def build_and_solve_model_L2_bounds(N, M, gll_nodes, K_sub=2):
    """
    N: number of GLL nodes => defines N Lagrange polynomials L_0..L_{N-1}.
    M: total number of piecewise-linear breakpoints => (M-1) subintervals.
    gll_nodes: array of the GLL nodes in [-1,1].
    K_sub: number of sample fractions *inside each subinterval* to enforce constraints.

    Decision vars:
      - x[i] in [-1,1], i=0..M-1, with x[0]=-1, x[M-1]=+1, x[i+1]>=x[i].
      - For each polynomial j, we have piecewise nodal values l[j,i], u[j,i].

    The objective is the sum of squared gap:
      sum_{i=0..M-2} sum_{k} sum_{j=0..N-1} (
         (u_j(x_{i,k}) - L_j(x_{i,k}))^2 + (L_j(x_{i,k}) - l_j(x_{i,k}))^2
      )
    where x_{i,k} is a sample point in [x[i],x[i+1]].
    """
    model = pyo.ConcreteModel()

    # ------------------
    # 1) Define indexes
    # ------------------
    model.i_set = pyo.RangeSet(0, M-1)     # for breakpoints x[i]
    model.j_set = pyo.RangeSet(0, N-1)     # for polynomials L_j

    # -------------------------
    # 2) Variables: x[i], etc.
    # -------------------------
    # x[i] are the subinterval breakpoints in [-1,1]
    model.x = pyo.Var(model.i_set, bounds=(-1,1))
    model.x[0].fix(-1)       # left endpoint
    model.x[M-1].fix(1)      # right endpoint

    xinit = lobatto_nodes(M)
    for i in range(1,M-1):
        model.x[i].value = xinit[i]

    # For each polynomial j, define nodal values l[j,i] and u[j,i]
    model.l = pyo.Var(model.j_set, model.i_set)
    model.u = pyo.Var(model.j_set, model.i_set)

    for i in range(N):
            for j in range(M):
                model.l[i,j].value = -1.0
                model.u[i,j].value = 1.0

    blow = np.zeros((N,M))
    bhigh = np.zeros((N,M))
    xt = np.zeros(M)

    # Force ordering: x[i+1] >= x[i] + a small gap
    def _order_rule(m, i):
        if i < M-1:
            return m.x[i+1] >= m.x[i] + 1e-2
        return pyo.Constraint.Skip
    model.order_c = pyo.Constraint(model.i_set, rule=_order_rule)

    # input(' ')
    # -------------------------------------------
    # 3) Build constraints & objective increment
    # -------------------------------------------
    model.con_bounds = pyo.ConstraintList()
    gap_exprs = []  # we'll store squared-gap expressions for the objective

    # We sample each subinterval [x[i], x[i+1]] at K_sub interior fractions
    # e.g. if K_sub=2, sample_fractions=[0.25, 0.75], etc.
    # Or you can pick any distribution in [0,1].
    sample_fractions = np.linspace(0, 1, K_sub) #you like
    # print(sample_fractions)
    # input(' ')

    for i in range(M-1):
        norm_fac = model.x[i+1]-model.x[i]
        norm_fac *= 1.0/K_sub
        for frac in sample_fractions:
            x_ik = model.x[i] + frac*(model.x[i+1] - model.x[i])
            # For each polynomial j, build piecewise-linear interpolation
            #   l_j(x_ik) = l[j,i] + slope_l * (x_ik - x[i])
            #   u_j(x_ik) = similarly
            def l_ik_expr(j):
                return (
                    model.l[j,i] +
                    (model.l[j,i+1] - model.l[j,i])*
                    ((x_ik - model.x[i]) / (model.x[i+1] - model.x[i]))
                )

            def u_ik_expr(j):
                return (
                    model.u[j,i] +
                    (model.u[j,i+1] - model.u[j,i])*
                    ((x_ik - model.x[i]) / (model.x[i+1] - model.x[i]))
                )

            for j in range(N):
                # Evaluate L_j at x_ik (symbolically recognized by Pyomo)
                L_j_ik = lagrange_basis_polynomial(j, x_ik, gll_nodes)

                # Enforce l_ik <= L_j_ik <= u_ik
                model.con_bounds.add(l_ik_expr(j) <= L_j_ik)
                model.con_bounds.add(L_j_ik <= u_ik_expr(j))

                # Add squared gaps to objective: (u_j-L_j)^2 + (L_j-l_j)^2
                gap_above = u_ik_expr(j) - L_j_ik
                gap_below = L_j_ik - l_ik_expr(j)
                gap_exprs.append(norm_fac*gap_above**2)
                gap_exprs.append(norm_fac*gap_below**2)

    # ----------------------
    # 4) Define Objective
    # ----------------------
    def _obj_rule(m):
        # sum of the stored squared-gap expressions
        return (sum(gap_exprs))
    model.obj = pyo.Objective(rule=_obj_rule, sense=pyo.minimize)

    # Solve
    solver = pyo.SolverFactory('ipopt')
    solver.options['halt_on_ampl_error'] = 'yes'
    solver.options['max_iter'] = 1000
    solver.options['mu_init'] = 1e-1
    solver.options['tol'] = 1e-15
    solver.options['linear_solver'] = 'mumps'  # or 'ma57' if available
    solver.options['mu_strategy'] = 'adaptive'
    solver.options['nlp_scaling_method'] = 'gradient-based'
    obj_init = model.obj()
    solver.solve(model, tee=True)

    # Extract the solution
    x_sol = np.array([pyo.value(model.x[i]) for i in model.i_set])
    u_sol = np.zeros((N, M))
    l_sol = np.zeros((N, M))
    for j in range(N):
        for i in range(M):
            u_sol[j, i] = pyo.value(model.u[j,i])
            l_sol[j, i] = pyo.value(model.l[j,i])

    return x_sol, u_sol, l_sol, model, obj_init, xt, blow, bhigh

def plot_bounds_for_each_basis_in_pdf(x_sol, u_sol, l_sol, gll_nodes, x_solt, l_solt, u_solt, pdf_filename="bounds_plots.pdf"):
    """
    x_sol: 1D array (length M) of shared interval breakpoints.
    u_sol: (N x M) array, with u_sol[j,i] = upper bound for basis j at x_sol[i].
    l_sol: (N x M) array, with l_sol[j,i] = lower bound for basis j at x_sol[i].
    gll_nodes: the GLL nodes used for the Lagrange basis polynomials.
    pdf_filename: name of the output PDF file.
    """
    N = len(gll_nodes)

    # We'll create a dense grid in [-1, 1] for plotting each polynomial
    x_dense = np.linspace(-1, 1, 400)

    with PdfPages(pdf_filename) as pdf:
        # Loop over each basis polynomial j
        for j in range(N):
            # 1) Create a new figure
            fig, ax = plt.subplots(figsize=(6,4))

            # 2) Compute the polynomial L_j on the dense grid
            L_vals = [lagrange_basis_polynomial(j, xd, gll_nodes) for xd in x_dense]
            ax.plot(x_dense, L_vals, 'k-', label=f'Lagrange basis $L_{j}$')

            # 3) Plot the piecewise-linear bounds for basis j
            #    -> We have M points (x_sol[i]), with bounds l_sol[j,i], u_sol[j,i].
            ax.plot(x_sol, l_sol[j,:], 'bo--', label='Lower bound', linewidth=1)
            ax.plot(x_sol, u_sol[j,:], 'ro--', label='Upper bound', linewidth=1)

            ax.plot(x_solt, l_solt[j,:], 'bo-', label='Lower bound T', linewidth=1)
            ax.plot(x_solt, u_solt[j,:], 'ro-', label='Upper bound T', linewidth=1)

            # 4) Styling
            ax.set_title(f'Bounds for Lagrange basis j={j}')
            ax.set_xlabel('x')
            ax.set_ylabel('value')
            ax.grid(True)
            ax.legend(loc='best')
            ax.plot(gll_nodes,0*gll_nodes,'ko-')

            # 5) Add the figure to the PDF
            pdf.savefig(fig)
            plt.close(fig)  # Close to free memory

    print(f"Saved {N} plots (one per basis) to {pdf_filename}")

# Example usage
if __name__ == '__main__':
    global N, M
    parser = argparse.ArgumentParser(description="A script that processes some arguments")

	# Add arguments
    parser.add_argument('--N', type=int, help='Number of rows (N)', default=7)
    parser.add_argument('--M', type=int, help='Number of columns (M)', default=12)

    args = parser.parse_args()
    N = args.N
    M = args.M
    gll_nodes = lobatto_nodes(N)

    nsamples = 30 #samples per sub-interval

    x_sol, u_sol, l_sol, model, obj_init, xt, blow, bhigh = build_and_solve_model_L2_bounds(N, M, gll_nodes, nsamples)

    filename = f"bnddata_spts_lobatto_{N}_bpts_optip_{M}.pdf"

    plot_bounds_for_each_basis_in_pdf(x_sol, u_sol, l_sol, gll_nodes,
                                      xt, blow, bhigh, pdf_filename=filename)

    filename = f"bnddata_spts_lobatto_{N}_bpts_optip_{M}.txt"
    np.savetxt(filename, [N], fmt="%d", newline="\n")
    with open(filename, "a") as f:
        np.savetxt(f, gll_nodes, fmt="%.15f", newline="\n")
        np.savetxt(f, [M], fmt="%d", newline="\n")
        np.savetxt(f, x_sol, fmt="%.15f", newline="\n")
        for i in range(N):
            np.savetxt(f, l_sol[i,:], fmt="%.15f", newline="\n")
            np.savetxt(f, u_sol[i,:], fmt="%.15f", newline="\n")
