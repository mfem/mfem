// Parallel heat dissipation minimization topology optimization with thickness constraint.
//
// Sample run:  mpirun -np 4 ./HeatTopOpt_maxlen
//              mpirun -np 4 ./HeatTopOpt_maxlen -vf 0.3 -mi 200

#include "mfem.hpp"
#include "DiffuTopOpt.hpp"
#include "qoi.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../pde_filter.hpp"
#include <memory>
#include <fstream>

using namespace std;
using namespace mfem;

void initialize_rays(int dim, int nrays, Vector *ray_starts, Vector *ray_ends);

int main(int argc, char *argv[])
{
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 1. Options.
    int    ref_levels   = 3;
    int    order        = 1;
    real_t r_f          = 0.03;       // min filter length
    real_t vol_fraction = 0.3;
    int    max_it       = 200;
    real_t tol          = 1e-3;       // stopping tol on iteration error
    real_t move         = 0.2;        // MMA move limit
    real_t epsilon      = 1e-2;       // thickness residual tolerance

    // Thickness constraint parameters
    int    nrays        = 10;         // number of thickness measurement rays
    int    nsamples     = 100;        // samples per ray for integration
    real_t alpha_min    = 0.05;       // minimum thickness bound
    real_t alpha_max    = 0.3;        // maximum thickness bound

    bool visualization = true;
    bool paraview      = false;

    const real_t k_min    = 1e-3;     // SIMP void conductivity (floors AMG conditioning)
    const real_t k_max    = 1.0;      // SIMP max conductivity
    const real_t exponent = 3.0;      // SIMP exponent

    OptionsParser args(argc, argv);
    args.AddOption(&ref_levels, "-r", "--refine", "uniform refinement levels");
    args.AddOption(&order, "-o", "--order", "finite element order");
    args.AddOption(&vol_fraction, "-vf", "--volume-fraction", "volume fraction");
    args.AddOption(&r_f, "-rf", "--r_fwidth", "min filter width");
    args.AddOption(&nrays, "-nr", "--nrays", "number of thickness measurement rays");
    args.AddOption(&nsamples, "-ns", "--nsamples", "samples per ray");
    args.AddOption(&alpha_min, "-amin", "--alpha_min", "minimum thickness bound");
    args.AddOption(&alpha_max, "-amax", "--alpha_max", "maximum thickness bound");
    args.AddOption(&epsilon, "-e", "--epsilon", "thickness residual tolerance");
    args.AddOption(&max_it, "-mi", "--max-it", "max optimization iterations");
    args.AddOption(&tol, "-tol", "--tol", "stopping tol on max design change");
    args.AddOption(&move, "-mv", "--move", "MMA move limit");
    args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                    "store solution in paraview");
    args.AddOption(&visualization, "-vis", "--visualization",
                    "-no-vis", "--no-visualization", "enable GLVis visualization");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0) { args.PrintUsage(cout); }
        return 1;
    }
    if (myid == 0) { args.PrintOptions(cout); }

    // 2. Build the mesh: heatsink domain (2D rectangular domain)
    //    2D: 2 x 2 rectangle
    //    Boundary attributes: central bottom region (0.5, 0) to (1.5, 0) = 1 (essential BC)
    //                         all other boundaries = 2
    Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true, 2.0, 2.0);
    const int dim = mesh.Dimension();

    const int ess_bc_attr = 1;
    const int other_bc_attr = 2;

    // 3. Refine the mesh
    for (int l = 0; l < ref_levels; l++)
    {
        mesh.UniformRefinement();
    }

    // Mark boundaries: essential BC = 1, others = 2
    for (int i = 0; i < mesh.GetNBE(); i++)
    {
        Element *be = mesh.GetBdrElement(i);
        Array<int> vertices;
        be->GetVertices(vertices);

        // Get vertex coordinates
        real_t *v0 = mesh.GetVertex(vertices[0]);
        real_t *v1 = mesh.GetVertex(vertices[1]);

        // Check if this boundary element is on bottom (y ≈ 0) and within [0.5, 1.5] in x
        real_t y_avg = 0.5 * (v0[1] + v1[1]);
        real_t x_min = std::min(v0[0], v1[0]);
        real_t x_max = std::max(v0[0], v1[0]);

        if (std::abs(y_avg) < 1e-8 && x_min >= 0.95 - 1e-8 && x_max <= 1.05 + 1e-8)
        {
            be->SetAttribute(ess_bc_attr);  // Central bottom: attribute 1
        }
        else
        {
            be->SetAttribute(other_bc_attr);  // All others: attribute 2
        }
    }
    mesh.SetAttributes();

    const int bottom_attr = ess_bc_attr;  // Essential BC attribute

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    // FindPointsGSLIB (used by ThicknessResidual) needs a nodal grid function.
    pmesh.SetCurvature(order);

    // 4. Define finite element collections and spaces
    H1_FECollection filter_fec(order, dim);
    L2_FECollection control_fec(order - 1, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);
    ParFiniteElementSpace control_fes(&pmesh, &control_fec);

    // Printing all true dofs.
    HYPRE_BigInt filter_size = filter_fes.GlobalTrueVSize();
    HYPRE_BigInt design_size = control_fes.GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "\nfilter dofs = " << filter_size
            << ",  design dofs = " << design_size << endl;
    }

    // 5. Initialized all the grid funcitons and coefficients
    // min filter varaibles
    ParGridFunction rho(&control_fes);
    ParGridFunction rho_filter(&filter_fes);
    rho = vol_fraction;
    rho_filter = vol_fraction;

    GridFunctionCoefficient rho_cf(&rho);

    // 5b. Setup thickness measurement rays (vertical rays across the domain)
    Vector *ray_starts = new Vector[nrays];
    Vector *ray_ends = new Vector[nrays];
    initialize_rays(dim, nrays, ray_starts, ray_ends);

    // Thickness design variables (one per ray)
    Vector alpha(nrays);
    alpha = 0.5 * (alpha_min + alpha_max);  // initialize to mid-range

    // Thickness constraint residual: ∑_i 1/2 (A_i - α_i)²
    ThicknessResidual thickness_residual(pmesh, rho_filter, ray_starts, ray_ends, nrays, alpha, nsamples);

    // 6. SIMP thermal conductivity coefficients
    ConstantCoefficient one_cf(1.0);
    SIMPCoefficient simp_cf(&rho_filter, k_min, k_max, exponent);                // k(rho~)
    SIMPGradCoefficient simp_grad_cf(&rho_filter, k_min, k_max, exponent);       // k'(rho~)

    // 7. Construct the heat diffusion solver using DiffusionSolver
    // Uniform heat source in the domain
    ConstantCoefficient heat_source_cf(1.0);

    DiffusionSolver *heat_solver = new DiffusionSolver(&pmesh, order, simp_cf);
    heat_solver->SetEssentialBC(bottom_attr, 0.0);  // T = 0 on bottom boundary (heat sink)
    heat_solver->SetHeatSource(heat_source_cf);

    // |grad T|^2 coefficient; multiply by k or k' as needed.
    GradTNorm2Coefficient gradT2_cf(&heat_solver->GetTemperature());

    // k(rho~) * |grad T|^2  — integrand of the objective J
    ProductCoefficient dissipation_cf(simp_cf, gradT2_cf);
    // k'(rho~) * |grad T|^2 — adjoint RHS for the filter (dJ/drho~)
    ProductCoefficient prod(simp_grad_cf, gradT2_cf);
    ProductCoefficient dJdrho_cf(-1.0, prod);

    // Objective functional: J = ∫ k(rho~) |grad T|^2 dx
    HeatObjective objective(MPI_COMM_WORLD, &filter_fes, dissipation_cf);

    // 7b. Min length scale filter solver
    toopt::PDEFilterOptions filter_opts;
    filter_opts.filter_radius = r_f;
    toopt::PDEFilter filter(filter_fes, control_fes, filter_opts);
    filter.Assemble();

    // 8. Volume constraint data:  g(rho) = (1, rho)/Vstar - 1.
    ParLinearForm vol_form(&control_fes);
    vol_form.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
    vol_form.Assemble();
    std::unique_ptr<HypreParVector> vol_w(vol_form.ParallelAssemble());

    real_t domain_volume;
    real_t loc = vol_w->Sum();
    MPI_Allreduce(&loc, &domain_volume, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
    const real_t Vstar = vol_fraction * domain_volume;

    // 9. MMA optimizer and its per-iteration work vectors.
    const int n = control_fes.GetTrueVSize();       // local rho design variables
    const int m = (myid == 0) ? nrays : 0;          // local alpha design variables
    Array<int> toffsets(3); toffsets[0] = 0; toffsets[1] = n; toffsets[2] = m; toffsets.PartialSum();

    const int num_con = 2;                          // constraints: volume + thickness
    Vector dcdrho(n), fival(num_con);
    Vector rho_tv(n), rho_old(n);

    rho.GetTrueDofs(rho_tv);

    // stacked design  x = [ rho ; alpha ]   (alpha block present on rank 0 only)
    BlockVector tx_local(toffsets);
    tx_local.GetBlock(0) = rho_tv;
    if (myid == 0) { tx_local.GetBlock(1) = alpha; }
    mfem_mma::MMAOptimizerParallel mma(MPI_COMM_WORLD, n+m, 2, tx_local);

    BlockVector tx_min(toffsets), tx_max(toffsets);
    BlockVector df0dx(toffsets);                    // objective gradient  df0/dx = [ dc/drho ; 0 ]
    BlockVector dthick(toffsets);                   // thickness constraint gradient  [ dR/drho ; dR/dalpha ]
    BlockVector dvol(toffsets);                     // volume constraint gradient is constant:  [ vol_w/Vstar ; 0 ]
    dvol.GetBlock(0) = *vol_w;  dvol.GetBlock(0) /= Vstar;
    dvol.GetBlock(1) = 0.0;

    Vector dfidx[num_con];  dfidx[0] = dvol;        // dfidx[1] set each iteration

    // 10. Visualizations
    // 10a. GLVis
    char vishost[] = "localhost";  int visport = 19916;
    socketstream sout;
    if (visualization) { sout.open(vishost, visport);  sout.precision(8); }

    // 10b. Paraview
    ParGridFunction phys_density(&filter_fes);
    ParaViewDataCollection paraview_dc("HeatTopOpt", &pmesh);

    if (paraview) {
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.RegisterField("density", &phys_density);
        paraview_dc.RegisterField("rho_filter", &rho_filter);
        paraview_dc.RegisterField("temperature", &heat_solver->GetTemperature());
    }

    // 10c. CSV convergence log (rank 0 only).
    std::ofstream csv;
    if (myid == 0)
    {
        csv.open("convergence.csv");
        csv << "it,objective,vol,res_max,eps,iterErr\n";
    }

    // 11. Optimization loop.
    int k = 0;
    real_t iterationError = 1.0;

    for (; k < max_it && iterationError > tol; k++)
    {
        // (1) forward filter:  (r_f^2 K + M) ρ~ = M_fc ρ
        filter.Mult(rho, rho_filter);

        // (2) state solve:  -div(k(ρ~) grad T) = q
        heat_solver->Solve();

        // (3) adjoint filter + objective gradient:
        //     w~  = (r_f^2 K + M)^{-1} ∫ (-k'(ρ~) |grad T|^2) φ_i
        //     dJ/drho = M_fc^T w~
        ParLinearForm adj_rhs(&filter_fes);
        adj_rhs.AddDomainIntegrator(new DomainLFIntegrator(dJdrho_cf));
        adj_rhs.Assemble();
        std::unique_ptr<HypreParVector> adj_rhs_tv(adj_rhs.ParallelAssemble());
        filter.MultTranspose(*adj_rhs_tv, dcdrho);

        // (4) objective gradient:  df0/dx = [ dJ/drho ; 0 ]
        df0dx.GetBlock(0) = dcdrho;
        df0dx.GetBlock(1) = 0.0;

        // (5) thickness constraint evaluation and gradient
        real_t thickness_res = thickness_residual.Eval();

        // Thickness gradient:
        //   w.r.t. alpha:  dR/dalpha_i = -(A_i - α_i)   (alpha block lives on rank 0)
        //   w.r.t. rho:    dR/drho     = filter^T ( Σ_i (A_i-α_i) Σ_k ds_i φ(x_ik) ),
        //                  since A_i = ∫ ρ~ ds depends on ρ through the filter.
        Vector thick_ell_filter;
        thickness_residual.GetGradRHS(thick_ell_filter);
        filter.MultTranspose(thick_ell_filter, dthick.GetBlock(0));
        if (myid == 0) { thickness_residual.GetGrad(dthick.GetBlock(1)); }
        dfidx[1] = dthick;

        // (6) MMA update
        rho.GetTrueDofs(rho_tv);

        // constraints:  volume constraint  and  ∑_i 1/2 (A_i - α_i)² − ε ≤ 0
        real_t vol = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / domain_volume;
        fival(0) = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / Vstar - 1.0;
        fival(1) = thickness_res - epsilon;

        // box constraints:  rho ∈ [0,1],  α_i ∈ [alpha_min, alpha_max]  (move limits)
        for (int i = 0; i < n; i++)
        {
            tx_min[i] = std::max(real_t(0), rho_tv[i] - move);
            tx_max[i] = std::min(real_t(1), rho_tv[i] + move);
        }
        for (int i = 0; i < m; i++)
        {
            tx_min[n+i] = std::max(alpha_min, alpha(i) - move);
            tx_max[n+i] = std::min(alpha_max, alpha(i) + move);
        }

        // stacked update  x = [ ρ ; α ]   (alpha block present on rank 0 only)
        tx_local.GetBlock(0) = rho_tv;
        if (myid == 0) { tx_local.GetBlock(1) = alpha; }
        rho_old = rho_tv;
        real_t obj_val = objective.Eval();
        mma.Update(tx_local, df0dx, obj_val, fival, dfidx, tx_min, tx_max);
        rho.SetFromTrueDofs(tx_local.GetBlock(0));
        // rank 0 owns alpha; broadcast the update so every rank's copy stays in sync.
        if (myid == 0) { alpha = tx_local.GetBlock(1); }
        MPI_Bcast(alpha.GetData(), nrays, MPITypeMap<real_t>::mpi_type, 0, MPI_COMM_WORLD);

        // measure iteration error
        ParGridFunction rho_old_gf(&control_fes);
        rho_old_gf.SetFromTrueDofs(rho_old);
        iterationError = rho_old_gf.ComputeL1Error(rho_cf);

        if (myid == 0)
        {
            mfem::out << "it " << setw(3) << k + 1
                    << "   J = " << scientific << setprecision(6) << obj_val
                    << "   vol = " << fixed << setprecision(4) << vol
                    << "   res_max = " << scientific << setprecision(3) << fival(1)
                    << "   eps = " << fixed << setprecision(4) << epsilon
                    << "   iterErr = " << setprecision(4) << iterationError << endl;

            csv << k + 1 << ','
                << scientific << setprecision(8) << obj_val << ','
                << vol << ','
                << fival(1) << ','
                << epsilon << ','
                << iterationError << '\n';
            csv.flush();
        }

        // physical density r(rho~) for both GLVis and the ParaView archive
        phys_density.ProjectCoefficient(simp_cf);

        if (visualization)
        {
            sout << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n" << pmesh << phys_density
                << "window_title 'Design density r(rho~)'" << flush;
        }

        if (paraview)
        {
            paraview_dc.SetCycle(k + 1);
            paraview_dc.SetTime(k + 1);
            paraview_dc.Save();
        }
    }

    if (myid == 0)
    {
        csv.close();
        mfem::out << "\nfinished after " << k << " iterations\n";
    }

    // Cleanup
    delete heat_solver;
    delete[] ray_starts;
    delete[] ray_ends;

    return 0;
}

void initialize_rays(int dim, int nrays, Vector *ray_starts, Vector *ray_ends)
{
    // Vertical rays for measuring thickness in 2D heatsink
    // Domain is [0, 2] x [0, 1]
    for (int r = 0; r < nrays; r++)
    {
        ray_starts[r].SetSize(dim);
        ray_ends[r].SetSize(dim);

        // Distribute rays horizontally across the domain
        real_t y_pos = 0.5 + 0.5 * r / (nrays - 1);

        // Vertical rays from bottom to top
        ray_starts[r](0) = 0.0;
        ray_ends[r](0)   = 2.0;

        ray_starts[r](1) = y_pos;
        ray_ends[r](1)   = y_pos;
    }
}