// Linear elasticity topology optimization with a max thickness constraint.
// Thickness measure is calculated by solving an advection pde, one advection
// direction per ray angle, giving one thickness constraint per direction.
//
// Sample run:  mpirun -np 8 ./ElastTopOpt_ct -m "../../data/d_square_4_holes.msh" -vf 0.4 -amax 0.6
//              mpirun -np 8 ./ElastTopOpt_ct -m "../../data/circular_5_holes_pentagon.msh" -vf 0.4 -amax 0.6
//              mpirun -np 8 ./ElastTopOpt_ct -m "../../data/disk_6_holes.msh" -vf 0.4 -amax 0.6

#include "mfem.hpp"
#include "ElastTopOpt.hpp"
#include "qoi.hpp"
#include "pseudo_transient_solver.hpp"
#include "../../diffusion_mass_solver.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../mtop_solvers.hpp"
#include <memory>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;
using namespace mfem;

// Boundary data for one elasticity solve.
struct LoadCase
{
    Array<int>    clamp_attrs;
    Array<int>    load_attrs;
    Array<real_t> fx, fy, fz;
};

// n_dir directions evenly spaced over [start, start+span).
struct RaySet
{
    int    n_dir = 4;
    real_t span  = M_PI;
    real_t start = 0.0;
};

// Ray directions per mode.
struct RaySpec
{
    RaySet parallel;
    RaySet cone;
    real_t half_ang = M_PI / 12;   // cone half-opening angle (15 deg -> 30 deg cone)
};

// Full problem definition for one mesh
// one elasticity solve per load case.
struct MeshProblem
{
    Array<int>            domain_attr;
    Array<int>            outer_bdr_attrs;
    std::vector<LoadCase> cases;
    RaySpec               rays;
};

enum class RayMode { Parallel = 1, Cone = 2 };

std::vector<Vector> BuildRays(RayMode mode, const RaySpec &spec, int dim);
std::unique_ptr<VectorCoefficient> MakeRayCoeff(RayMode mode, const RaySpec &spec,
                                                const Vector &axis, int dim);
MeshProblem loadMesh(int myid, const char *mesh_file, Mesh &mesh);

int main(int argc, char *argv[])
{
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // Timer
    double init_time = MPI_Wtime();

    // 1. Options.
    const char *mesh_file = nullptr;
    int    ref_levels   = 0;
    int    order        = 2;
    real_t r_f          = 0.03;       // min filter length
    real_t alpha_min    = 1e-6;       // thickness variable lower bound
    real_t alpha_max    = 1.0;        // thickness variable upper bound
    real_t beta         = 2.0;        // Heaviside beta
    real_t eta          = 0.2;        // Heaviside eta
    real_t vol_fraction = 0.4;
    int    max_it       = 300;
    real_t tol          = 1e-4;       // stopping tol on iteration error
    real_t move         = 0.1;        // MMA move limit
    real_t epsilon      = 1e-2;       // thickness residual tolerance
    bool   pa           = false;
    const int seed      = 0;
    int    ray_type = 1;              // thickness ray strategy: 1=parallel, 2=cone

    bool visualization = true;
    bool paraview      = false;

    const real_t E_min    = 1e-6;     // SIMP void stiffness
    const real_t E_max    = 1.0;      // SIMP E max
    const real_t exponent = 1.0;      // SIMP exponent (applied to the projection)
    // --- PLAIN SIMP ---  use p = 3 when SIMP acts directly on rho~
    // const real_t exponent = 3.0;

    int    init_it   = 20;
    real_t decay     = 0.9;
    real_t eps_floor = 1e-5;
    int    decay_int = 10;

    int    beta_steps = 50;           // Heaviside beta continuation steps
    real_t beta_max   = 2.0;          // Heaviside beta max value

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "mesh file to use", true);
    args.AddOption(&ray_type, "-rt", "--ray-type", "ray type: 1 = parallel, 2 = cone");
    args.AddOption(&ref_levels, "-r", "--refine", "uniform refinement levels");
    args.AddOption(&order, "-o", "--order", "finite element order");
    args.AddOption(&vol_fraction, "-vf", "--volume-fraction", "volume fraction");
    args.AddOption(&r_f, "-rf", "--r_fwidth", "min filter width");
    args.AddOption(&alpha_min, "-amin", "--alpha_min", "lower bound for the thickness variables");
    args.AddOption(&alpha_max, "-amax", "--alpha_max", "upper bound for the thickness variables");
    args.AddOption(&beta, "-b", "--beta", "Heaviside beta");
    args.AddOption(&beta_steps, "-bs", "--beta-steps", "Heaviside beta continuation steps");
    args.AddOption(&beta_max, "-bm", "--beta-max", "Heaviside beta max value");
    args.AddOption(&eta, "-eta", "--eta", "Heaviside eta");
    args.AddOption(&epsilon, "-e", "--epsilon", "thickness residual tolerance (initial)");
    args.AddOption(&decay, "-d", "--decay", "decay rate of epsilon");
    args.AddOption(&decay_int, "-di", "--decay_int", "decay interval of epsilon");
    args.AddOption(&init_it, "-ii", "--init_it", "initial iterations before decay");
    args.AddOption(&max_it, "-mi", "--max-it", "max optimization iterations");
    args.AddOption(&tol, "-tol", "--tol", "stopping tol on max design change");
    args.AddOption(&move, "-mv", "--move", "MMA move limit");
    args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                    "store solution in paraview");
    args.AddOption(&visualization, "-vis", "--visualization",
                    "-no-vis", "--no-visualization", "enable GLVis visualization");
    args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa", "--no-partial-assembly",
                    "enable partial assembly (recommended for large problems)");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0) { args.PrintUsage(cout); }
        return 1;
    }
    if (!mesh_file)
    {
        if (myid == 0) { cout << "Error: -m <mesh file> is required." << endl; }
        return 1;
    }
    if (myid == 0) { args.PrintOptions(cout); }
    
    // initial (uniform) design density -- depends on the parsed options
    const real_t domain_init = alpha_max * vol_fraction;
    const RayMode ray_mode = (ray_type == 2) ? RayMode::Cone : RayMode::Parallel;
    
    // 2. Load the mesh and the problem description (domain, loads, ray spec).
    Mesh mesh;
    MeshProblem prob = loadMesh(myid, mesh_file, mesh);

    const int n_elast_solve = static_cast<int>(prob.cases.size());
    Array<int> &domain_attr     = prob.domain_attr;
    Array<int> &outer_bdr_attrs = prob.outer_bdr_attrs;

    // 3. Refine the mesh and construct pmesh / the design subdomain.
    const int dim = mesh.Dimension();

    vector<Vector> ray_dirs = BuildRays(ray_mode, prob.rays, dim);
    const int n_dir = static_cast<int>(ray_dirs.size());

    for (int l = 0; l < ref_levels; l++)
    {
        mesh.UniformRefinement();
    }

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    ParSubMesh design_domain = ParSubMesh::CreateFromDomain(pmesh, domain_attr);

    // 3b. Build ray fields and mark the outflow boundary for each direction.
    Array<int> candidate_be;        // extract only the outer boundary elements
    Array<int> candidate_attr;
    for (int i = 0; i < pmesh.GetNBE(); i++)
    {
        const int el_attr = pmesh.GetBdrAttribute(i);

        if (outer_bdr_attrs.Find(el_attr) < 0) continue;
        candidate_be.Append(i);
        candidate_attr.Append(el_attr);
    }

    vector<unique_ptr<VectorCoefficient>> ray_cf(n_dir);
    vector<unique_ptr<ParSubMesh>> outflow(n_dir);

    for (int r = 0; r < n_dir; r++)
    {
        ray_cf[r] = MakeRayCoeff(ray_mode, prob.rays, ray_dirs[r], dim);

        // mark outflow (v . n > 0) on the candidate boundary elements
        const int outflow_attr = 100 + r;
        for (int k = 0; k < candidate_be.Size(); k++)
        {
            const int idx = candidate_be[k];
            ElementTransformation *trans = pmesh.GetBdrElementTransformation(idx);
            const IntegrationPoint &ip = Geometries.GetCenter(
                                            pmesh.GetBdrElementGeometry(idx));
            trans->SetIntPoint(&ip);

            Vector vv(dim);
            ray_cf[r]->Eval(vv, *trans, ip);

            Vector normal(dim);
            CalcOrtho(trans->Jacobian(), normal);

            bool is_outflow = (vv * normal > 0);
            if (is_outflow) { pmesh.SetBdrAttribute(idx, outflow_attr); }
        }
        pmesh.SetAttributes();

        Array<int> submesh_attr;  submesh_attr.Append(outflow_attr);
        outflow[r] = make_unique<ParSubMesh>(ParSubMesh::CreateFromBoundary(pmesh, submesh_attr));

        // restore original attributes before marking the next ray
        for (int k = 0; k < candidate_be.Size(); k++)
        {
            const int original_attr = candidate_attr[k];
            pmesh.SetBdrAttribute(candidate_be[k], original_attr);
        }
        pmesh.SetAttributes();
    }

    // 4. Define finite element collections and spaces
    H1_FECollection state_fec(order, dim);
    H1_FECollection filter_fec(order, dim);
    H1_FECollection control_fec(order-1, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace state_fes(&design_domain, &state_fec, dim, Ordering::byVDIM);
    ParFiniteElementSpace filter_fes(&design_domain, &filter_fec);
    ParFiniteElementSpace control_fes(&design_domain, &control_fec);

    // Printing all true dofs.
    HYPRE_BigInt state_size   = state_fes.GlobalTrueVSize();
    HYPRE_BigInt filter_size  = filter_fes.GlobalTrueVSize();
    HYPRE_BigInt control_size = control_fes.GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "\nstate dofs = "   << state_size
             << ",  filter dofs = "  << filter_size
             << ",  design dofs = " << control_size << endl;
    }

    // 5. Initialize all the grid functions and coefficients
    ParGridFunction rho(&control_fes);
    ParGridFunction rho_filter(&filter_fes);

    rho.Randomize(seed);
    rho -= 0.5;
    rho *= 0.1;
    rho += domain_init;
    rho_filter = domain_init;

    GridFunctionCoefficient rho_cf(&rho);

    // 5b. Thickness design variables live on the outflow submeshes, one set per
    //     ray direction.  Solving for rho_a requires DG, so alpha also lives in DG.
    DG_FECollection dgfec(order, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace dgfes(&pmesh, &dgfec);

    vector<unique_ptr<DG_FECollection>> sub_dg_fec(n_dir);
    vector<unique_ptr<ParFiniteElementSpace>> sub_dg_fes(n_dir);
    vector<unique_ptr<ParGridFunction>> alpha(n_dir);

    for (int r = 0; r < n_dir; r++)
    {
        const int sub_dim = outflow[r]->Dimension();
        sub_dg_fec[r] = make_unique<DG_FECollection>(order, sub_dim, BasisType::GaussLobatto);
        sub_dg_fes[r] = make_unique<ParFiniteElementSpace>(outflow[r].get(), sub_dg_fec[r].get());

        alpha[r] = make_unique<ParGridFunction>(sub_dg_fes[r].get());
        *alpha[r] = domain_init;
    }

    // Lame constants and SIMP material coefficients
    ConstantCoefficient one_cf(1.0);
    ConstantCoefficient E_cf(3.0), nu_cf(0.3);
    IsoElasticyLambdaCoeff lambda_cf(&E_cf, &nu_cf);
    IsoElasticySchearCoeff mu_cf(&E_cf, &nu_cf);

    // Heaviside projections of rho~:  eroded -> stiffness, dilated -> volume,
    // intermediate -> the design that is actually reported.
    HeavisideCoefficient rho_erod_cf(&rho_filter, beta, 1-eta);
    HeavisideGradCoefficient rho_erod_grad_cf(&rho_filter, beta, 1-eta);

    HeavisideCoefficient rho_dila_cf(&rho_filter, beta, eta);
    HeavisideGradCoefficient rho_dila_grad_cf(&rho_filter, beta, eta);

    HeavisideCoefficient rho_inter_cf(&rho_filter, beta, 0.5);

    // SIMP on the eroded projection: r(rho_e) = E_min + rho_e^p (E_max - E_min)
    SIMPCoefficient simp_cf(rho_erod_cf, E_min, E_max, exponent);                // r(rho_e)
    SIMPGradCoefficient simp_grad_cf(rho_erod_cf, E_min, E_max, exponent);       // r'(rho_e)

    // --- PLAIN SIMP ---  SIMP directly on rho~, no projection
    // SIMPCoefficient simp_cf(&rho_filter, E_min, E_max, exponent);             // r(rho~)
    // SIMPGradCoefficient simp_grad_cf(&rho_filter, E_min, E_max, exponent);    // r'(rho~)

    // 6. Construct the solvers.
    // 6a. Linear elasticity solvers, one per load case.
    ProductCoefficient E_simp(simp_cf, E_cf);           // r(rho_e) * E0

    vector<unique_ptr<IsoLinElasticSolver>> elast(n_elast_solve);
    for (int i = 0; i < n_elast_solve; i++)
    {
        elast[i] = make_unique<IsoLinElasticSolver>(&design_domain, order, pa);
        const LoadCase &lc = prob.cases[i];
        for (int j = 0; j < lc.load_attrs.Size(); j++)
        {
            if (dim == 2)
            {
                elast[i]->AddSurfLoad(lc.load_attrs[j], lc.fx[j], lc.fy[j]);
            }
            else if (dim == 3)
            {
                elast[i]->AddSurfLoad(lc.load_attrs[j], lc.fx[j], lc.fy[j], lc.fz[j]);
            }
        }
        for (int j = 0; j < lc.clamp_attrs.Size(); j++)
        {
            elast[i]->AddDispBC(lc.clamp_attrs[j], -1, real_t(0));
        }
        elast[i]->SetMaterial(E_simp, nu_cf);
        elast[i]->SetLinearSolver(1e-10, 1e-14, 10000);
    }

    ParGridFunction u(&state_fes);
    StrainEnergyDensityCoefficient energy_cf(&lambda_cf, &mu_cf, &u);

    // dc/drho~ = - (r'(rho_e) * H_e'(rho~)) * psi0(u)
    ProductCoefficient drdrho_cf(simp_grad_cf, rho_erod_grad_cf);
    ProductCoefficient prod(energy_cf, drdrho_cf);
    ProductCoefficient dcdrho_tilde_cf(-1.0, prod);

    // --- PLAIN SIMP ---  dc/drho~ = -r'(rho~) * psi0
    // ProductCoefficient prod(energy_cf, simp_grad_cf);
    // ProductCoefficient dcdrho_tilde_cf(-1.0, prod);

    // 6b. Min length scale filter solver (diffusion-mass PDE filter).
    PDEFilter filter(control_fes, filter_fes);
    filter.SetFilterRadius(r_f);
    DiffusionMassSolver &filter_solver = filter.GetSolver();
    for (int a = 1; a <= design_domain.bdr_attributes.Max(); a++)
    {
        bool is_outer = (outer_bdr_attrs.Find(a) >= 0);
        if (is_outer)
        {
            filter_solver.Boundary().Add(a, 0.0);  // Outer boundaries: rho~ = 0
        }
        else
        {
            filter_solver.Boundary().Add(a, 1.0);  // Holes: rho~ = 1
        }
    }
    filter.Assemble();

    // PDEFilter::Mult solves the constrained system without eliminating the
    // essential columns, so nonzero Dirichlet data never reaches the interior
    // (rho~ decays to 0 next to the holes). The filter is linear, so recover the
    // missing A_fe*g lifting once via the FormLinearSystem path and add it back.
    Array<int> filter_bdr_marker(design_domain.bdr_attributes.Max());
    filter_bdr_marker = 1;
    Array<int> filter_ess_tdofs;
    filter_fes.GetEssentialTrueDofs(filter_bdr_marker, filter_ess_tdofs);

    Vector rho_filter_lift_tv;
    {
        ParGridFunction lift(&filter_fes);
        filter_solver.Solve(lift);
        lift.GetTrueDofs(rho_filter_lift_tv);
        rho_filter_lift_tv.SetSubVector(filter_ess_tdofs, real_t(0));
    }

    // 6c. Advection solvers for the thickness measure, one per ray direction,
    //     with the pseudo-transient time step set from the CFL condition.
    vector<unique_ptr<MaterialThicknessSolver>> advect(n_dir);
    DGMassInverse minv(dgfes);

    real_t cfl = 0.5;
    real_t hmin = infinity();
    for (int i = 0; i < pmesh.GetNE(); i++)
    {
        hmin = min(pmesh.GetElementSize(i, 1), hmin);
    }
    MPI_Allreduce(MPI_IN_PLACE, &hmin, 1,  MPITypeMap<real_t>::mpi_type, MPI_MIN,
                    pmesh.GetComm());
    real_t dt = cfl * hmin / (2 * order + 1);

    for (int r = 0; r < n_dir; r++)
    {
        advect[r] = make_unique<MaterialThicknessSolver>(filter_fes, dgfes, *ray_cf[r], pa);
        advect[r]->SetMinv(minv);
        advect[r]->GetSolver().SetTimeStep(dt);      // pseudo-transient time step
        advect[r]->GetSolver().SetTerminalTime(1.5);
    }

    // 7. Construct the quantity of interest objects
    Compliance comp(MPI_COMM_WORLD, &filter_fes, simp_cf, energy_cf);

    // volume of the dilated field, measured against V* = vol_fraction |Omega|
    VolumeResidual vol_qoi(MPI_COMM_WORLD, &filter_fes, &rho_dila_cf,
                           &rho_dila_grad_cf, vol_fraction);

    // Max-thickness constraint residual per ray:  1/2 ∫_Gamma_out,r (rho_a − α_r)²
    vector<unique_ptr<AdvectThicknessResidual>> adv_res(n_dir);
    for (int r = 0; r < n_dir; r++)
    {
        adv_res[r] = make_unique<AdvectThicknessResidual>(*outflow[r],
                                                          advect[r]->GetRhoA(),
                                                          *alpha[r]);
    }

    // --- PLAIN SIMP ---  linear volume constraint on the filtered density,
    //                     g(rho) = (1, rho~)/Vstar - 1.  The Dirichlet filter BCs
    //                     make the filter non-volume-preserving, so (1,rho) is
    //                     not equivalent.
    // ParLinearForm vol_form(&filter_fes);
    // vol_form.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
    // vol_form.Assemble();
    // std::unique_ptr<HypreParVector> vol_w(vol_form.ParallelAssemble());
    //
    // real_t domain_volume;
    // real_t loc = vol_w->Sum();
    // MPI_Allreduce(&loc, &domain_volume, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
    // const real_t Vstar = vol_fraction * domain_volume;
    //
    // // d(1,rho~)/drho = L^T w is constant: the filter operator never changes,
    // // and the Dirichlet lifting contributes a constant offset with zero derivative.
    // Vector dvol_drho(control_fes.GetTrueVSize());
    // filter.MultTranspose(*vol_w, dvol_drho);

    // 8. MMA optimizer and its per-iteration work vectors.
    // stacked design  x = [ rho ; alpha_0 ; alpha_1 ; ... ; alpha_{n_dir-1} ]
    const int n  = control_fes.GetTrueVSize();      // local rho design variables
    const int nf = filter_fes.GetTrueVSize();
    vector<int> m(n_dir);
    for (int r = 0; r < n_dir; r++) { m[r] = sub_dg_fes[r]->GetTrueVSize(); }

    Array<int> toffsets(n_dir + 2);
    toffsets[0] = 0;
    toffsets[1] = n;
    for (int r = 0; r < n_dir; r++) { toffsets[2 + r] = m[r]; }
    toffsets.PartialSum();

    const int num_con = 1 + n_dir;                  // constraints: volume + one thickness per ray

    // 8a. stacked design  x = [ rho ; alpha_0 ; ... ]
    Vector rho_tv(n), rho_old(n);
    vector<Vector> alpha_tv(n_dir);

    rho.GetTrueDofs(rho_tv);
    for (int r = 0; r < n_dir; r++) { alpha[r]->GetTrueDofs(alpha_tv[r]); }

    BlockVector tx_local(toffsets);
    tx_local.GetBlock(0) = rho_tv;
    for (int r = 0; r < n_dir; r++) { tx_local.GetBlock(1 + r) = alpha_tv[r]; }

    Vector a(num_con), c(num_con), d(num_con);
    a = 0.0; c = 1000.0; d = 0.0;
    mfem_mma::MMAOptimizerParallel mma(MPI_COMM_WORLD, toffsets.Last(), num_con, tx_local, a, c, d);

    // 8b. objective initialization
    BlockVector df0dx(toffsets);                    // objective gradient  df0/dx = [ dc/drho ; 0 ; ... ]
    Vector dcdrho(n);                               // compliance gradient  dc/drho

    // 8c. local constraints
    Vector fival(num_con);
    vector<Vector> dfidx(num_con);

    BlockVector dvol(toffsets);                     // volume gradient  [ dg/drho ; 0 ; ... ]
    dvol = 0.0; dfidx[0] = dvol;
    Vector dvol_tilde(nf);                          // dV_d/drho~

    // one full-size gradient BlockVector per ray-thickness constraint; only
    // block(0) (drho) and block(1+r) (dalpha_r) are ever nonzero.
    vector<BlockVector> dthick(n_dir, BlockVector(toffsets));

    // --- PLAIN SIMP ---  the linear volume gradient is constant:  [ L^T w/Vstar ; 0 ; ... ]
    // dvol.GetBlock(0) = dvol_drho;
    // dvol.GetBlock(0) /= Vstar;
    // dfidx[0] = dvol;

    // 8d. mma upper and lower bounds
    BlockVector tx_min(toffsets), tx_max(toffsets);

    // 9. Visualizations
    // 9a. GLVis
    char vishost[] = "localhost";  int visport = 19916;
    socketstream sout_proj, sout_filter;

    if (visualization) {
        sout_proj.open(vishost, visport);
        sout_proj.precision(8);

        sout_proj << "parallel " << num_procs << " " << myid << "\n"
            << "solution\n" << design_domain << rho_filter
            << "window_title 'Design density r(rho~)'\n"
            << "window_geometry 0 0 800 600\n"
            << "colorbar_numberformat '%.2f'\n"
            << "keys Rjlc*****\n" << flush;

        sout_filter.open(vishost, visport);
        sout_filter.precision(8);

        sout_filter << "parallel " << num_procs << " " << myid << "\n"
            << "solution\n" << design_domain << rho_filter
            << "window_title 'Filtered density rho~'\n"
            << "window_geometry 810 0 800 600\n"
            << "colorbar_numberformat '%.2f'\n"
            << "keys Rjlc*****\n" << flush;
    }

    // 9b. Paraview
    ParGridFunction phys_density(&filter_fes);
    std::ostringstream run_tag;
    run_tag << "ct_amax" << alpha_max << "_vf" << vol_fraction;
    ParaViewDataCollection paraview_dc(run_tag.str(), &design_domain);

    if (paraview) {
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.RegisterField("density", &phys_density);
        paraview_dc.RegisterField("rho_filter", &rho_filter);
    }

    // 9c. Initialization block runtime.
    double init_block_time = MPI_Wtime() - init_time;
    if (myid == 0)
    {
        mfem::out << "\nInitialization block runtime: " << fixed << setprecision(4)
                   << init_block_time << " s\n";
    }

    // 9d. CSV convergence log (rank 0 only).
    std::ofstream csv;
    if (myid == 0)
    {
        csv.open("convergence.csv");
        csv << "it,compliance,volume,res_thick,eps,iterErr,iter_time\n";
    }

    // 10. Optimization loop.
    real_t iterationError = 1.0;
    real_t init_comp = 1.0;

    // Track next iteration for epsilon decay and beta doubling
    int next_epsilon_decay = init_it;
    int next_beta_double = init_it + beta_steps;

    double opt_start_time = MPI_Wtime();

    int it = 1;
    for (; (it <= init_it) || (it <= max_it && iterationError > tol); it++)
    {
        double iter_start_time = MPI_Wtime() - opt_start_time;

        // (1) forward filter:  (r_f^2 K + M) ρ~ = M_fc ρ  (+ Dirichlet lifting)
        rho.GetTrueDofs(rho_tv);
        Vector rho_filter_tv(nf);
        filter.Mult(rho_tv, rho_filter_tv);
        rho_filter_tv += rho_filter_lift_tv;
        rho_filter.SetFromTrueDofs(rho_filter_tv);

        // (2) state solves:  K(ρ~) u = f   (self-adjoint compliance), averaged
        //     over the load cases, together with the adjoint filter rhs
        real_t compliance = 0.0;
        Vector adj_rhs_tv(nf);
        adj_rhs_tv = 0.0;
        for (int i = 0; i < n_elast_solve; i++)
        {
            elast[i]->Assemble();
            elast[i]->FSolve();
            u = elast[i]->GetDisplacements();
            compliance += comp.Eval();

            ParLinearForm adj_rhs(&filter_fes);
            adj_rhs.AddDomainIntegrator(new DomainLFIntegrator(dcdrho_tilde_cf));
            adj_rhs.Assemble();
            std::unique_ptr<HypreParVector> adj_rhs_e_tv(adj_rhs.ParallelAssemble());
            adj_rhs_tv += *adj_rhs_e_tv;
        }
        compliance  /= n_elast_solve;
        adj_rhs_tv  /= n_elast_solve;

        // (3) adjoint filter + objective gradient:
        //     w~  = (r_f^2 K + M)^{-1} ∫ (-r'(ρ~) psi_0) φ_i
        //     dc/drho = M_fc^T w~
        filter.MultTranspose(adj_rhs_tv, dcdrho);
        df0dx.GetBlock(0) = dcdrho;                     // objective gradient
        for (int r = 0; r < n_dir; r++) { df0dx.GetBlock(1 + r) = 0.0; }

        // (4) volume constraint and gradient on the dilated field:
        //       g        = V_d / V* - 1
        //       dg/drho~ = (H_d'(ρ~), φ_i) / V*
        fival(0) = vol_qoi.Eval() - 1.0;                // update constraint value
        real_t vol = (fival(0) + 1.0) * vol_fraction;   // current volume fraction
        vol_qoi.GetGrad(dvol_tilde);
        filter.MultTranspose(dvol_tilde, dvol.GetBlock(0));
        dfidx[0] = dvol;                                // update constraint gradient

        // --- PLAIN SIMP ---  linear volume constraint on rho~ (gradient is
        //                     constant, set once outside the loop)
        // const real_t vol_int = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_filter_tv);
        // real_t vol = vol_int / domain_volume;
        // fival(0) = vol_int / Vstar - 1.0;

        // (5) advect rho~ along each ray field to get the thickness measure rho_a,
        // (6) then the max-thickness constraint and gradient per direction:
        //       1/2 ∫(rho_a−α_r)² − ε ≤ 0
        //       dR/dalpha_r = (α_r − rho_a) on Gamma_out,r
        //       dR/drho     = M_fc^T N^T (rho_a − α_r)  via the adjoint advection solve
        real_t res_thick = -infinity();
        real_t fi_thick  = -infinity();
        for (int r = 0; r < n_dir; r++)
        {
            advect[r]->SetRhs(rho_filter_tv);
            advect[r]->FSolve();
            const real_t thickres = adv_res[r]->Eval();
            res_thick = std::max(res_thick, thickres);

            dthick[r] = 0.0;

            Vector dGdrhoa;
            adv_res[r]->GetGrad(dGdrhoa, dthick[r].GetBlock(1 + r));

            // transfer dGdrhoa back to the full-domain dgfes
            ParGridFunction g_sub(sub_dg_fes[r].get());  g_sub.SetFromTrueDofs(dGdrhoa);
            ParGridFunction g_full(&dgfes);              g_full = 0.0;
            outflow[r]->Transfer(g_sub, g_full);
            Vector rhs_full;  g_full.GetTrueDofs(rhs_full);

            // chain rule adjoint solve: dG/drho = M_fc^T N^T g
            advect[r]->SetAdjointRhs(rhs_full);
            advect[r]->ASolve();
            filter.MultTranspose(advect[r]->GetSensitivity(), dthick[r].GetBlock(0));

            fival(1 + r) = thickres / epsilon - 1.0;     // update constraint value
            dthick[r] /= epsilon;
            dfidx[1 + r] = dthick[r];                    // update constraint gradient

            fi_thick = std::max(fi_thick, fival(1 + r));
        }

        // (7) box constraints:  rho ∈ [0,1],  α_r ∈ [alpha_min, alpha_max]  (move limits)
        for (int r = 0; r < n_dir; r++) { alpha[r]->GetTrueDofs(alpha_tv[r]); }
        for (int i = 0; i < n; i++)
        {
            tx_min[i] = std::max(real_t(0), rho_tv[i] - move);
            tx_max[i] = std::min(real_t(1), rho_tv[i] + move);
        }
        for (int r = 0; r < n_dir; r++)
        {
            for (int i = 0; i < m[r]; i++)
            {
                tx_min[toffsets[1 + r] + i] = std::max(alpha_min, alpha_tv[r][i] - move);
                tx_max[toffsets[1 + r] + i] = std::min(alpha_max, alpha_tv[r][i] + move);
            }
        }

        // (8) MMA update on the stacked design  x = [ ρ ; α_0 ; ... ; α_{n_dir-1} ]
        tx_local.GetBlock(0) = rho_tv;
        for (int r = 0; r < n_dir; r++) { tx_local.GetBlock(1 + r) = alpha_tv[r]; }
        rho_old = rho_tv;

        // Normalize compliance and gradient by initial value
        if (it == 1) { init_comp = compliance; }
        compliance /= init_comp;
        df0dx /= init_comp;

        mma.Update(tx_local, df0dx, compliance, fival, dfidx.data(), tx_min, tx_max);
        rho.SetFromTrueDofs(tx_local.GetBlock(0));
        for (int r = 0; r < n_dir; r++) { alpha[r]->SetFromTrueDofs(tx_local.GetBlock(1 + r)); }

        // measure iteration error
        ParGridFunction rho_old_gf(&control_fes);
        rho_old_gf.SetFromTrueDofs(rho_old);
        iterationError = rho_old_gf.ComputeL1Error(rho_cf);

        double iter_end_time = MPI_Wtime() - opt_start_time;
        double iter_runtime  = iter_end_time - iter_start_time;

        // (9) reporting
        if (myid == 0)
        {
            const int w = 14;               // column width
            mfem::out << "\niteration " << it << '\n' << left
                    << setw(w) << "c"
                    << setw(w) << "volume"
                    << setw(w) << "res_thick_max"
                    << setw(w) << "eps"
                    << setw(w) << "fival_max"
                    << setw(w) << "beta"
                    << setw(w) << "iterErr" << '\n'
                    << string(7*w, '=') << '\n'
                    << fixed      << setprecision(6) << setw(w) << compliance
                    <<               setprecision(4) << setw(w) << vol
                    << scientific << setprecision(3) << setw(w) << res_thick
                    <<               setprecision(3) << setw(w) << epsilon
                    <<               setprecision(3) << setw(w) << fi_thick
                    << fixed      << setprecision(0) << setw(w) << beta
                    << scientific << setprecision(4) << setw(w) << iterationError << '\n'
                    << "iteration runtime: " << fixed << setprecision(2)
                    << iter_runtime << " s,   elapsed time at end: "
                    << iter_end_time << " s" << endl;

            csv << it << ','
                << scientific << setprecision(8) << compliance << ','
                << vol << ','
                << res_thick << ','
                << epsilon << ','
                << iterationError << ','
                << iter_runtime << '\n';
            csv.flush();
        }

        // (10) tighten the max-thickness tolerance and update beta
        // Epsilon decay: starts at init_it, then every decay_int iterations
        if (it == next_epsilon_decay)
        {
            epsilon = std::max(epsilon * decay, eps_floor);
            next_epsilon_decay += decay_int;
        }

        // Beta doubling: starts at init_it + beta_steps, then every beta_steps iterations
        if (it == next_beta_double && beta < beta_max)
        {
            beta *= 2;
            rho_erod_cf.SetBeta(beta);  rho_erod_grad_cf.SetBeta(beta);
            rho_dila_cf.SetBeta(beta);  rho_dila_grad_cf.SetBeta(beta);
            rho_inter_cf.SetBeta(beta);
            next_beta_double += beta_steps;
        }

        // physical density for both GLVis and the ParaView archive
        phys_density.ProjectCoefficient(rho_inter_cf);
        // --- PLAIN SIMP ---
        // phys_density.ProjectCoefficient(simp_cf);

        if (visualization)
        {
            sout_proj << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n" << design_domain << phys_density << flush;

            sout_filter << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n" << design_domain << rho_filter << flush;
        }

        // if (paraview)
        // {
        //     paraview_dc.SetCycle(it);
        //     paraview_dc.SetTime(it);
        //     paraview_dc.Save();
        // }
    }

    double total_runtime = MPI_Wtime() - init_time;

    if (myid == 0)
    {
        csv.close();
        mfem::out << "\nfinished after " << (it - 1) << " iterations"
                  << "\ntotal runtime is " << total_runtime << " s\n";
    }

    // Option: save only the final solution instead of all iterations
    if (paraview)
    {
        paraview_dc.SetCycle(it - 1);
        paraview_dc.SetTime(it - 1);
        paraview_dc.Save();
    }

    return 0;
}

// ray axis directions for the selected mode, evenly spaced over its ray set
std::vector<Vector> BuildRays(RayMode mode, const RaySpec &spec, int dim)
{
    const RaySet &s = (mode == RayMode::Cone) ? spec.cone : spec.parallel;

    std::vector<Vector> dirs;
    for (int i = 0; i < s.n_dir; i++)
    {
        const real_t ang = s.start + s.span * i / s.n_dir;
        Vector v(dim); v = 0.0;
        v(0) = cos(ang);
        v(1) = sin(ang);
        dirs.push_back(v);
    }
    return dirs;
}

// advection direction field for one ray: parallel is a constant axis, cone is a
// unit field diverging from a source 1/tan(half_ang) behind the origin
std::unique_ptr<VectorCoefficient> MakeRayCoeff(RayMode mode, const RaySpec &spec,
                                                const Vector &axis, int dim)
{
    if (mode == RayMode::Parallel)
    {
        return std::make_unique<VectorConstantCoefficient>(axis);
    }

    const real_t R = 1.1 / tan(spec.half_ang);
    Vector src(axis); src *= -R;
    auto field = [src, dim](const Vector &x, Vector &d)
    {
        d.SetSize(dim);
        subtract(x, src, d);
        const real_t n = d.Norml2();
        if (n > 0) { d /= n; }
    };
    return std::make_unique<VectorFunctionCoefficient>(dim, field);
}

// disk mesh: one load case, 18 rays swept over the full circle
static MeshProblem SetupDisk6Holes(Mesh &mesh, const char *mesh_file)
{
    mesh = Mesh(mesh_file);

    MeshProblem p;
    p.domain_attr.Append(1);
    p.outer_bdr_attrs = Array<int>({1});
    p.rays.parallel = { 18, 2 * M_PI, 0.0 };
    p.rays.cone     = { 18, 2 * M_PI, 0.0 };

    p.cases.resize(1);
    LoadCase &lc = p.cases[0];
    lc.clamp_attrs = Array<int>({ 7 });

    const real_t angles_deg_forces[] = {90.0, 162.0, 234.0, 306.0, 18.0};
    for (int j = 0; j < 5; j++)
    {
        const real_t ang = angles_deg_forces[j] * M_PI / 180.0;
        const real_t fdx = -sin(ang), fdy = cos(ang);
        lc.load_attrs.Append(2 + j);
        lc.fx.Append(-fdx);
        lc.fy.Append(-fdy);
        lc.fz.Append(0.0);
    }
    return p;
}

// square mesh: two load cases with swapped clamp/load, 4 parallel or 8 cone rays
static MeshProblem SetupSquare4Holes(Mesh &mesh, const char *mesh_file)
{
    mesh = Mesh(mesh_file);

    MeshProblem p;
    p.domain_attr.Append(1);
    p.outer_bdr_attrs = Array<int>({1, 2, 3, 4});
    p.rays.parallel = { 4, M_PI,     0.0 };
    p.rays.cone     = { 8, 2 * M_PI, 0.0 };

    p.cases.resize(2);

    // first elast solve
    p.cases[0].clamp_attrs = Array<int>({ 6, 7});
    p.cases[0].load_attrs  = Array<int>({ 5, 8});
    p.cases[0].fx          = Array<real_t>({ 1,-1});
    p.cases[0].fy          = Array<real_t>({-1, 1});
    p.cases[0].fz          = Array<real_t>({ 0, 0});

    // second elast solve
    p.cases[1].clamp_attrs = Array<int>({ 5, 8});
    p.cases[1].load_attrs  = Array<int>({ 6, 7});
    p.cases[1].fx          = Array<real_t>({-1, 1});
    p.cases[1].fy          = Array<real_t>({-1, 1});
    p.cases[1].fz          = Array<real_t>({ 0, 0});

    return p;
}

// pentagon mesh: five load cases, rays aligned with the edges
static MeshProblem SetupPentagon(Mesh &mesh, const char *mesh_file)
{
    mesh = Mesh(mesh_file);

    MeshProblem p;
    p.domain_attr.Append(1);
    p.outer_bdr_attrs = Array<int>({1});
    p.rays.parallel = { 5, 2 * M_PI, 0.0 };     // aligned with pentagon edges
    p.rays.cone     = { 5, 2 * M_PI, 0.0 };

    const int n_case = 5;
    const real_t angles_deg_forces[] = {90.0, 162.0, 234.0, 306.0, 18.0};
    real_t fdx[5], fdy[5];
    for (int k = 0; k < n_case; k++)
    {
        const real_t ang = angles_deg_forces[k] * M_PI / 180.0;
        fdx[k] = cos(ang);
        fdy[k] = sin(ang);
    }

    const int first_attr = 2;
    p.cases.resize(n_case);
    for (int i = 0; i < n_case; i++)
    {
        const int clamped_attr = first_attr + i;
        LoadCase &lc = p.cases[i];
        lc.clamp_attrs = Array<int>({ clamped_attr });

        for (int j = 0; j < n_case; j++)
        {
            const int attr = first_attr + j;
            if (attr == clamped_attr) { continue; }
            lc.load_attrs.Append(attr);
            lc.fx.Append(-fdx[j]);
            lc.fy.Append(-fdy[j]);
            lc.fz.Append(0.0);
        }
    }
    return p;
}

// select the per-mesh setup from the mesh file name
MeshProblem loadMesh(int myid, const char *mesh_file, Mesh &mesh)
{
    if (strstr(mesh_file, "disk_6_holes.msh") != NULL)
    {
        return SetupDisk6Holes(mesh, mesh_file);
    }
    else if (strstr(mesh_file, "d_square_4_holes.msh") != NULL)
    {
        return SetupSquare4Holes(mesh, mesh_file);
    }
    else if (strstr(mesh_file, "circular_5_holes_pentagon.msh") != NULL)
    {
        return SetupPentagon(mesh, mesh_file);
    }

    if (myid == 0) { mfem::out << "invalid mesh file" << endl; }
    return MeshProblem();
}
