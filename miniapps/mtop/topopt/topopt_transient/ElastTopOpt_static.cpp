// =============================================================================
// Steady-State Topology Optimization
// =============================================================================
//
// Compliance-minimization topology optimization for linear elasticity
// with Helmholtz PDE filter and MMA optimizer.
// Foundation for transient topology optimization extension.
//
// PROBLEM:
//   Domain: 3×1 cantilever beam
//   BC: Clamped at x=0, free elsewhere
//   Load: Point load at (2.85, 0.5) downward
//   Objective: Minimize compliance ∫σ:ε dΩ
//   Constraint: Volume fraction ≤ V*
//
// METHOD:
//   - SIMP interpolation: E(ρ) = E_min + ρ³(E_max - E_min)
//   - Helmholtz filter: (r²∇² + 1)ρ̃ = ρ
//   - MMA optimizer with move limits
//   - State solver: BoomerAMG preconditioned PCG
//
// COMPILE:
//   make ElastTopOpt_transient -j8
//
// RUN:
//   srun -n 4 ./ElastTopOpt_transient -vf 0.3 -mi 100 -r 5 -pv
//
// OPTIONS:
//   -vf   : target volume fraction (default: 0.5)
//   -mi   : maximum iterations (default: 100)
//   -r    : mesh refinement levels (default: 5)
//   -fr   : Helmholtz filter radius (default: 0.05)
//   -pv   : enable ParaView output
//
// OUTPUT:
//   ParaView/ElasticityTopOpt.pvd
//
// =============================================================================

#include "mfem.hpp"
#include "../topopt_maxlen/ElastTopOpt.hpp"
#include "../QuantityOfInterest.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../pde_filter.hpp"
#include "../../mtop_solvers.hpp"
#include <memory>

using namespace std;
using namespace mfem;

void bodyload(const Vector &x, Vector &f);

int main(int argc, char *argv[])
{
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 1. Options.
    int    ref_levels   = 5;
    int    order        = 2;
    real_t filter_r     = 0.05;       // Helmholtz filter radius
    real_t vol_fraction = 0.5;
    int    max_it       = 100;
    real_t tol          = 1e-3;       // stopping tol on design change
    real_t move         = 0.2;        // MMA move limit
    bool   visualization = true;
    bool   paraview      = false;

    const real_t E_min    = 1e-6;     // SIMP void stiffness
    const real_t E_max    = 1.0;      // SIMP solid stiffness
    const real_t exponent = 3.0;      // SIMP exponent

    OptionsParser args(argc, argv);
    args.AddOption(&ref_levels, "-r", "--refine", "uniform refinement levels");
    args.AddOption(&order, "-o", "--order", "finite element order");
    args.AddOption(&vol_fraction, "-vf", "--volume-fraction", "volume fraction");
    args.AddOption(&filter_r, "-fr", "--filter-radius", "Helmholtz filter radius");
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

    // 2. Create mesh: 3x1 cantilever beam, clamped at x=0
    Mesh mesh = Mesh::MakeCartesian2D(3, 1, Element::QUADRILATERAL, true, 3.0, 1.0);
    int dim = mesh.Dimension();

    // Label boundary attributes: 1=clamped (x=0), 2=top/bottom, 3=free (x=3)
    for (int i = 0; i < mesh.GetNBE(); i++)
    {
        Element *be = mesh.GetBdrElement(i);
        Array<int> v;
        be->GetVertices(v);
        real_t *c1 = mesh.GetVertex(v[0]);
        real_t *c2 = mesh.GetVertex(v[1]);
        real_t cx = 0.5 * (c1[0] + c2[0]);
        real_t cy = 0.5 * (c1[1] + c2[1]);

        int attr = 3;  // default: free
        if (std::abs(cx) < 1e-10) {
            attr = 1;  // clamp at x = 0
        }
        else if (std::abs(cy - 1.0) < 1e-10 || std::abs(cy) < 1e-10) {
            attr = 2;  // top/bottom edges
        }
        be->SetAttribute(attr);
    }
    mesh.SetAttributes();

    // 3. Refine mesh and construct parallel mesh
    for (int l = 0; l < ref_levels; l++)
    {
        mesh.UniformRefinement();
    }
    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    // 4. Define finite element spaces
    H1_FECollection state_fec(order, dim);
    H1_FECollection filter_fec(order, dim);
    L2_FECollection control_fec(order - 1, dim, BasisType::GaussLobatto);

    ParFiniteElementSpace state_fes(&pmesh, &state_fec, dim);      // displacement u
    ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);         // filtered density ρ̃
    ParFiniteElementSpace control_fes(&pmesh, &control_fec);       // design variable ρ

    HYPRE_BigInt state_size  = state_fes.GlobalTrueVSize();
    HYPRE_BigInt filter_size = filter_fes.GlobalTrueVSize();
    HYPRE_BigInt design_size = control_fes.GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "\nState DOFs:   " << state_size
             << "\nFilter DOFs:  " << filter_size
             << "\nDesign DOFs:  " << design_size << endl;
    }

    // 5. Initialize design variables and coefficients
    ParGridFunction rho(&control_fes);          // design variable ρ
    ParGridFunction rho_filter(&filter_fes);    // filtered density ρ̃
    rho = vol_fraction;
    rho_filter = vol_fraction;

    GridFunctionCoefficient rho_cf(&rho);

    // Lamé parameters and SIMP material coefficients
    ConstantCoefficient one_cf(1.0);
    ConstantCoefficient E_cf(3.0), nu_cf(0.3);
    IsoElasticyLambdaCoeff lambda_cf(&E_cf, &nu_cf);
    IsoElasticySchearCoeff mu_cf(&E_cf, &nu_cf);
    SIMPCoefficient simp_cf(&rho_filter, E_min, E_max, exponent);           // r(ρ̃)
    SIMPGradCoefficient simp_grad_cf(&rho_filter, E_min, E_max, exponent);  // r'(ρ̃)

    // 6. Setup linear elasticity solver
    VectorFunctionCoefficient force(dim, bodyload);
    ProductCoefficient E_simp(simp_cf, E_cf);    // r(ρ̃) * E₀

    IsoLinElasticSolver elast(&pmesh, order);
    elast.SetVolForce(force);
    elast.SetMaterial(E_simp, nu_cf);
    elast.AddDispBC(1, -1, 0.0);                 // clamp at boundary attr 1
    elast.SetLinearSolver(1e-10, 1e-14, 10000);

    // Strain energy density for compliance sensitivity
    StrainEnergyDensityCoefficient energy_cf(&lambda_cf, &mu_cf,
                                              &elast.GetDisplacements());
    ProductCoefficient prod(energy_cf, simp_grad_cf);   // r'(ρ̃) * ψ₀
    ProductCoefficient dcdrho_cf(-1.0, prod);           // dc/dρ̃ = -r'(ρ̃)*ψ₀
    Compliance comp(MPI_COMM_WORLD, &filter_fes, simp_cf, energy_cf);

    // 7. Setup Helmholtz filter:  (r² K + M) ρ̃ = M_fc ρ
    toopt::PDEFilterOptions filter_opts;
    filter_opts.filter_radius = filter_r;
    toopt::PDEFilter filter(filter_fes, control_fes, filter_opts);
    filter.Assemble();

    // 8. Volume constraint:  g(ρ) = ∫ρ / V* - 1 ≤ 0
    ParLinearForm vol_form(&control_fes);
    vol_form.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
    vol_form.Assemble();
    std::unique_ptr<HypreParVector> vol_w(vol_form.ParallelAssemble());

    real_t domain_volume;
    real_t loc = vol_w->Sum();
    MPI_Allreduce(&loc, &domain_volume, 1, MPITypeMap<real_t>::mpi_type,
                  MPI_SUM, MPI_COMM_WORLD);
    const real_t Vstar = vol_fraction * domain_volume;

    // 9. MMA optimizer setup
    const int n = control_fes.GetTrueVSize();
    const int num_con = 1;                       // single constraint: volume

    Vector rho_tv(n), rho_old(n);
    Vector dcdrho(n), fival(num_con);
    Vector dvol(n);

    rho.GetTrueDofs(rho_tv);

    dvol = *vol_w;
    dvol /= Vstar;
    Vector dfidx[num_con];
    dfidx[0] = dvol;

    mfem_mma::MMAOptimizerParallel mma(MPI_COMM_WORLD, n, num_con, rho_tv);
    mma.SetAsymptotes(0.5, 0.7, 1.2);

    Vector rho_min(n), rho_max(n);

    // 10. Visualization setup
    char vishost[] = "localhost";
    int visport = 19916;
    socketstream sout;
    if (visualization)
    {
        sout.open(vishost, visport);
        sout.precision(8);
    }

    ParGridFunction phys_density(&filter_fes);
    ParaViewDataCollection paraview_dc("ElasticityTopOpt", &pmesh);
    if (paraview)
    {
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.RegisterField("density", &phys_density);
        paraview_dc.RegisterField("rho_filter", &rho_filter);
    }

    // 11. Optimization loop
    int k = 0;
    real_t iterationError = 1.0;

    for (; k < max_it && iterationError > tol; k++)
    {
        // (1) Forward filter:  (r² K + M) ρ̃ = M_fc ρ
        filter.Mult(rho, rho_filter);

        // (2) State solve:  K(ρ̃) u = f
        elast.Assemble();
        elast.FSolve();
        elast.GetDisplacements();

        // (3) Adjoint filter + objective gradient:
        //     dc/dρ = M_fc^T (r² K + M)^(-1) ∫ -r'(ρ̃) ψ₀(u) φ_i
        ParLinearForm adj_rhs(&filter_fes);
        adj_rhs.AddDomainIntegrator(new DomainLFIntegrator(dcdrho_cf));
        adj_rhs.Assemble();
        std::unique_ptr<HypreParVector> adj_rhs_tv(adj_rhs.ParallelAssemble());
        filter.MultTranspose(*adj_rhs_tv, dcdrho);

        // (5) Evaluate objective and constraint
        real_t compliance = comp.Eval();
        real_t vol = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / domain_volume;
        fival(0) = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / Vstar - 1.0;

        // (6) Box constraints with move limits
        rho.GetTrueDofs(rho_tv);
        rho_old = rho_tv;

        for (int i = 0; i < n; i++)
        {
            rho_min[i] = std::max(real_t(0.0), rho_tv[i] - move);
            rho_max[i] = std::min(real_t(1.0), rho_tv[i] + move);
        }

        // (7) MMA update
        mma.Update(rho_tv, dcdrho, compliance, fival, dfidx, rho_min, rho_max);
        rho.SetFromTrueDofs(rho_tv);

        // (8) Measure iteration error
        ParGridFunction rho_old_gf(&control_fes);
        rho_old_gf.SetFromTrueDofs(rho_old);
        iterationError = rho_old_gf.ComputeL1Error(rho_cf);

        if (myid == 0)
        {
            mfem::out << "it " << setw(3) << k + 1
                      << "   c = " << scientific << setprecision(6) << compliance
                      << "   vol = " << fixed << setprecision(4) << vol
                      << "   iterErr = " << setprecision(6) << iterationError << endl;
        }

        // (9) Visualization
        phys_density.ProjectCoefficient(simp_cf);

        if (visualization)
        {
            sout << "parallel " << num_procs << " " << myid << "\n"
                 << "solution\n" << pmesh << phys_density
                 << "window_title 'Design density r(ρ̃)'" << flush;
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
        mfem::out << "\nOptimization completed after " << k << " iterations.\n";
    }

    return 0;
}

// Body load function: concentrated load at (2.85, 0.5)
void bodyload(const Vector &x, Vector &f)
{
    const int dim = x.Size();
    const real_t xcenter = 2.85;
    const real_t ycenter = 0.5;
    const real_t radius = 0.05;

    f = 0.0;

    real_t xdiff = x[0] - xcenter;
    real_t ydiff = x[1] - ycenter;

    if (sqrt(xdiff*xdiff + ydiff*ydiff) < radius)
    {
        f[dim-1] = -1.0;  // downward load
    }
}
