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

def build_and_solve_model(N, M, gll_nodes, num_samples_per_interval=2):
    """
    N: number of Lagrange basis polynomials (e.g., length of gll_nodes).
    M: total number of interval points (breakpoints). -> (M-1) subintervals.

    Returns (x_sol, u_sol, l_sol) where:
      - x_sol is length M, the shared interval points in [-1,1].
      - u_sol, l_sol are each N x M arrays, storing the piecewise
        linear bounds for each Lagrange polynomial.
    """
    model = pyo.ConcreteModel()

    # Indices
    model.Ix = pyo.RangeSet(0, M-1)     # for x-coordinates (0..M-1)
    model.Ij = pyo.RangeSet(0, N-1)     # for each polynomial j

    # x[i]: the shared breakpoints in [-1,1]
    model.x = pyo.Var(model.Ix, bounds=(-1,1))
    model.x[0].fix(-1)       # left endpoint
    model.x[M-1].fix(1)      # right endpoint

    xinit = lobatto_nodes(M)
    for i in range(1,M-1):
        model.x[i].value = xinit[i]

    blow = np.zeros((N,M))
    bhigh = np.zeros((N,M))

    #read initial condition from Tarik's output if it exists
    filename = f"bnddata_spts_lobatto_{N}_bpts_opt_{M}.txt"
    data = None
    ids = 0
    ide = 0
    if os.path.exists(filename):
        data = np.loadtxt(filename)
        ids = 0
        ide = 1+N
        datagll = data[ids:ide]
        ids = ide+1
        ide = 1+N+M+1
        dataint = data[ids:ide]
        for i in range(1,M-1):
            model.x[i].value = dataint[i]

    # for i in range(M):
        # print(model.x[i].value)


    # Force ordering x[i+1] >= x[i]
    def _order_rule(m, i):
        if i < M-1:
            return m.x[i+1] >= m.x[i] + 1e-2
        return pyo.Constraint.Skip
    model.order = pyo.Constraint(model.Ix, rule=_order_rule)

    # For each j, we have separate piecewise-linear "nodal" values:
    #    u[j,i], l[j,i], for i=0..M-1
    model.u = pyo.Var(model.Ij, model.Ix)
    model.l = pyo.Var(model.Ij, model.Ix)

    if data is not None:
        for i in range(N):
            for j in range(M):
                ids = ide
                ide = ids+1
                model.l[i,j].value = data[ids]-0
                blow[i,j] = model.l[i,j].value
                # print(j,i,data[ids],'k10l')
            for j in range(M):
                ids = ide
                ide = ids+1
                model.u[i,j].value = data[ids]+0
                bhigh[i,j] = model.u[i,j].value
                # print(j,i,data[ids],'k10u')

    # Constraint list to ensure l_j(x) <= L_j(x) <= u_j(x)
    model.bound_constraints = pyo.ConstraintList()

    # For each sub-interval i = 0..M-2, we sample points inside [x[i], x[i+1]].
    for i in range(M-1):
        # We'll pick some sample points via fractions in [0,1]
        sample_fractions = np.linspace(0, 1, num_samples_per_interval+2)[1:-1]
        for frac in sample_fractions:
            # Expression for x_{i,k}
            x_ik = model.x[i] + frac * (model.x[i+1] - model.x[i])

            # For each j, define piecewise-linear interpolation:
            #   u_j(x_ik) = u[j,i] + slope*(x_ik - x[i])
            # where slope = (u[j,i+1] - u[j,i]) / (x[i+1] - x[i])
            def u_ik_expr(j):
                return ( model.u[j,i] +
                         (model.u[j,i+1]-model.u[j,i])*
                         ( (x_ik - model.x[i]) / (model.x[i+1]-model.x[i]) ) )

            def l_ik_expr(j):
                return ( model.l[j,i] +
                         (model.l[j,i+1]-model.l[j,i])*
                         ( (x_ik - model.x[i]) / (model.x[i+1]-model.x[i]) ) )

            # Impose bounds for each Lagrange polynomial L_j
            for j in range(N):
                # Evaluate L_j at x_ik
                L_j_val = lagrange_basis_polynomial(j, x_ik, gll_nodes)
                model.bound_constraints.add(l_ik_expr(j) <= L_j_val)
                model.bound_constraints.add(L_j_val <= u_ik_expr(j))

    # Objective: Sum of integrals of (u_j - l_j) across x in [-1,1], for all j
    # For each j, the area is sum_{i=0..M-2} 0.5*(x[i+1]-x[i]) * [ (u_j[i]-l_j[i]) + (u_j[i+1]-l_j[i+1]) ]
    def _area_rule(m):
        total_area = 0
        for j in range(N):
            for i in range(M-1):
                length = (m.x[i+1] - m.x[i])
                avg_height = 0.5 * ((m.u[j,i] - m.l[j,i]) + (m.u[j,i+1] - m.l[j,i+1]))
                total_area += length * avg_height
        return total_area

    model.obj = pyo.Objective(rule=_area_rule, sense=pyo.minimize)

    # Solve
    solver = pyo.SolverFactory('ipopt')
    solver.options['halt_on_ampl_error'] = 'yes'
    solver.options['max_iter'] = 5000
    solver.options['mu_init'] = 1e-3
    obj_init = model.obj()
    solver.solve(model, tee=True)

    # Extract the solution
    x_sol = np.array([pyo.value(model.x[i]) for i in model.Ix])
    u_sol = np.zeros((N, M))
    l_sol = np.zeros((N, M))
    for j in range(N):
        for i in range(M):
            u_sol[j, i] = pyo.value(model.u[j,i])
            l_sol[j, i] = pyo.value(model.l[j,i])

    return x_sol, u_sol, l_sol, model, obj_init, blowt, bhight

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

    blow = np.zeros((N,M))
    bhigh = np.zeros((N,M))
    xt = np.zeros(M)
    xt[0] = -1.0
    xt[-1] = 1.0

    #read initial condition from Tarik's output if it exists
    filename = f"bnddata_spts_lobatto_{N}_bpts_opt_{M}.txt"
    data = None
    ids = 0
    ide = 0
    if os.path.exists(filename):
        data = np.loadtxt(filename)
        ids = 0
        ide = 1+N
        datagll = data[ids:ide]
        ids = ide+1
        ide = 1+N+M+1
        dataint = data[ids:ide]
        for i in range(1,M-1):
            model.x[i].value = dataint[i]
            xt[i] = model.x[i].value

    # Force ordering: x[i+1] >= x[i] + a small gap
    def _order_rule(m, i):
        if i < M-1:
            return m.x[i+1] >= m.x[i] + 1e-2
        return pyo.Constraint.Skip
    model.order_c = pyo.Constraint(model.i_set, rule=_order_rule)

    # For each polynomial j, define nodal values l[j,i] and u[j,i]
    model.l = pyo.Var(model.j_set, model.i_set)
    model.u = pyo.Var(model.j_set, model.i_set)

    # for i in range(M):
            # print(i,model.x[i].value,'k10x')

    if data is not None:
        for i in range(N):
            for j in range(M):
                ids = ide
                ide = ids+1
                model.l[i,j].value = data[ids]
                blow[i,j] = model.l[i,j].value
                # print(j,i,model.l[i,j].value,'k10l')
            for j in range(M):
                ids = ide
                ide = ids+1
                model.u[i,j].value = data[ids]
                bhigh[i,j] = model.u[i,j].value
                # print(j,i,model.u[i,j].value,'k10u')

    # input(' ')
    # -------------------------------------------
    # 3) Build constraints & objective increment
    # -------------------------------------------
    model.con_bounds = pyo.ConstraintList()
    gap_exprs = []  # we'll store squared-gap expressions for the objective

    # We sample each subinterval [x[i], x[i+1]] at K_sub interior fractions
    # e.g. if K_sub=2, sample_fractions=[0.25, 0.75], etc.
    # Or you can pick any distribution in [0,1].
    sample_fractions = np.linspace(0, 1, K_sub+2)[1:-1]  # skip endpoints if you like

    for i in range(M-1):
        for frac in sample_fractions:
            # x_{i,k} = x[i] + frac * (x[i+1] - x[i])
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
                gap_exprs.append(gap_above**2)
                gap_exprs.append(gap_below**2)

    # ----------------------
    # 4) Define Objective
    # ----------------------
    def _obj_rule(m):
        # sum of the stored squared-gap expressions
        return sum(gap_exprs)
    model.obj = pyo.Objective(rule=_obj_rule, sense=pyo.minimize)

    # Solve
    solver = pyo.SolverFactory('ipopt')
    solver.options['halt_on_ampl_error'] = 'yes'
    solver.options['max_iter'] = 100
    solver.options['mu_init'] = 1e-2
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
    # Example GLL nodes for N=4 (in [-1,1]) -- placeholder values:
    # (In practice, compute them with a GLL routine.)
    gll_nodes = lobatto_nodes(N)

    # x_sol, u_sol, l_sol, model, obj_init, blowt, bhight = build_and_solve_model(N, M, gll_nodes, 100)

    x_sol, u_sol, l_sol, model, obj_init, xsolt, blowt, bhight = build_and_solve_model_L2_bounds(N, M, gll_nodes, 100)

    # print("Breakpoints:", x_sol)
    # print("Upper-bound values:", u_sol)
    # print("Lower-bound values:", l_sol)
    print(N,M,obj_init,pyo.value(model.obj))

    filename = f"bnddata_spts_lobatto_{N}_bpts_optip_{M}.pdf"

    plot_bounds_for_each_basis_in_pdf(x_sol, u_sol, l_sol, gll_nodes,
                                      xsolt, blowt, bhight, pdf_filename=filename)

    filename = f"bnddata_spts_lobatto_{N}_bpts_optip_{M}.txt"
    np.savetxt(filename, [N], fmt="%d", newline="\n")
    with open(filename, "a") as f:
        np.savetxt(f, gll_nodes, fmt="%.15f", newline="\n")
        np.savetxt(f, [M], fmt="%d", newline="\n")
        np.savetxt(f, x_sol, fmt="%.15f", newline="\n")
        for i in range(N):
            np.savetxt(f, l_sol[i,:], fmt="%.15f", newline="\n")
            np.savetxt(f, u_sol[i,:], fmt="%.15f", newline="\n")
