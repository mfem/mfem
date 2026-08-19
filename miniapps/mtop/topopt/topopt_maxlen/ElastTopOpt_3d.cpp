// Linear elasticity topology optimization with a max thickness constraint, 3D.
// Thickness measure is calculated by solving an advection pde, one advection
// direction per ray, giving one thickness constraint per direction.  The design
// domain is a subdomain of the mesh, while the advection runs on the whole mesh.
//
// With no -m the built-in 3x1x1 Cartesian hex beam is used.
//
// Sample run:  mpirun -np 8 ./ElastTopOpt_3d -r 2 -rf 0.05 -vf 0.4

#include "mfem.hpp"
#include "ElastTopOpt.hpp"
#include "qoi.hpp"
#include "pseudo_transient_solver.hpp"
#include "../../diffusion_mass_solver.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../mtop_solvers.hpp"
#include "../../linear_elasticity_solver.hpp"
#include "checkpoint.hpp"
#include <memory>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;
using namespace mfem;

// Boundary/body data for one elasticity solve.
struct LoadCase
{
    Array<int>    clamp_attrs;
    Array<int>    load_attrs;                          // surface tractions
    Array<real_t> fx, fy, fz;
    void (*vol_load)(const Vector &, Vector &) = nullptr;   // optional body force
};

// Full problem definition for one mesh
// one elasticity solve per load case.
struct MeshProblem
{
    Array<int>            domain_attr;       // designable subdomain(s)
    Array<int>            outer_bdr_attrs;   // free surfaces: rho~ = 0 in the filter
    std::vector<LoadCase> cases;
};

std::vector<std::unique_ptr<VectorCoefficient>> SetupRays(int dim);
MeshProblem loadMesh(int myid, const char *mesh_file, Mesh &mesh);

// Extract Mesh with non-zero density
void SaveSolidSubmesh(ParMesh &pmesh, ParGridFunction &desi_density,
                      ParGridFunction &phys_density, const std::string &run_tag, 
                      int order, real_t threshold = 0.1);

int main(int argc, char *argv[])
{
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // Timer
    double init_time = MPI_Wtime();

    // 1. Options.
    const char *mesh_file = "";       // empty: use the built-in Cartesian beam
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

    int  cp          = 0;       // 0 = off, 1 = rho only, 2 = full state
    int  restart     = 0;       // 0 = off, 1 = rho only, 2 = full state
    int  pc_type     = 2;       // 0 = Jacobi, 1 = LOR diagonal AMG, 2 = LOR monolithic AMG
    bool lor_by_vdim = true;    // monolithic LOR (-pc 2) ordering: byVDIM vs byNODES
    const int seed   = 0;

    bool visualization = true;
    bool paraview      = false;

    const real_t E_min    = 1e-6;     // SIMP void stiffness
    const real_t E_max    = 1.0;      // SIMP E max
    const real_t exponent = 1.0;      // SIMP exponent (applied to the projection)
    // --- PLAIN SIMP ---  use p = 3 when SIMP acts directly on rho~
    // const real_t exponent = 3.0;

    int    init_it   = 25;
    real_t decay     = 0.5;
    real_t eps_floor = 1e-10;
    int    decay_int = 50;

    int    beta_steps = 100;           // Heaviside beta continuation steps
    real_t beta_max   = 2.0;          // Heaviside beta max value

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                    "mesh file to use; omit for the built-in Cartesian beam");
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
    args.AddOption(&pc_type, "-pc", "--elast-precond", "elasticity preconditioner: "
                    "0 = Jacobi, 1 = LOR diagonal AMG,  2 = LOR monolithic AMG");
    args.AddOption(&lor_by_vdim, "-vdim", "--by-vdim", "-nodes", "--by-nodes",
                    "monolithic LOR ordering: byVDIM / byNODES");
    args.AddOption(&cp, "-cp", "--checkpoint",
                    "checkpointing: 0 = off, 1 = rho only, 2 = rho + alpha + MMA state");
    args.AddOption(&restart, "-restart", "--restart",
                    "restart: 0 = off, 1 = load rho only, 2 = load full state and resume");
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

    // initial (uniform) design density -- depends on the parsed options
    const real_t domain_init = alpha_max * vol_fraction;

    // 2. Load the mesh and the problem description (domain, loads).
    Mesh mesh;
    MeshProblem prob = loadMesh(myid, mesh_file, mesh);

    const int n_elast_solve = static_cast<int>(prob.cases.size());
    Array<int> &domain_attr     = prob.domain_attr;
    Array<int> &outer_bdr_attrs = prob.outer_bdr_attrs;

    // 3. Refine the mesh and construct pmesh / the design subdomain.
    const int dim = mesh.Dimension();

    vector<unique_ptr<VectorCoefficient>> ray_cf = SetupRays(dim);
    const int n_dir = static_cast<int>(ray_cf.size());

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    for (int l = 0; l < ref_levels; l++)
    {
        pmesh.UniformRefinement();
    }

    ParSubMesh design_domain = ParSubMesh::CreateFromDomain(pmesh, domain_attr);

    // 3b. Mark the outflow boundary for each ray direction.
    Array<int> candidate_be;        // extract only the outer boundary elements
    Array<int> candidate_attr;
    for (int i = 0; i < pmesh.GetNBE(); i++)
    {
        const int el_attr = pmesh.GetBdrAttribute(i);

        if (outer_bdr_attrs.Find(el_attr) < 0) continue;
        candidate_be.Append(i);
        candidate_attr.Append(el_attr);
    }

    vector<unique_ptr<ParSubMesh>> outflow(n_dir);

    for (int r = 0; r < n_dir; r++)
    {
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
    ParFiniteElementSpace state_fes(&design_domain, &state_fec, dim, Ordering::byNODES);
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
        sub_dg_fec[r] = make_unique<DG_FECollection>(order, sub_dim, BasisType::Positive);
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
    ProductCoefficient lambda_simp_cf(simp_cf, lambda_cf);   // r(rho_e) * lambda0
    ProductCoefficient mu_simp_cf(simp_cf, mu_cf);           // r(rho_e) * mu0

    // setting up preconditioner
    LinearElasticitySolver::PreconditionerType elast_pc;
    if (pc_type == 0)
    {
        elast_pc = LinearElasticitySolver::PreconditionerType::Jacobi;
    }
    else if (pc_type == 1)
    {
        elast_pc = LinearElasticitySolver::PreconditionerType::LORDiagonalAMG;
    }
    else if (pc_type == 2)
    {
        elast_pc = LinearElasticitySolver::PreconditionerType::LORMonolithicAMG;
    }
    else
    {
        MFEM_ABORT("Unknown preconditioner! Elasticity preconditioner: "
                    "0 = Jacobi, 1 = LOR diagonal AMG, 2 = LOR monolithic AMG");
    }

    vector<unique_ptr<LinearElasticitySolver>> elast(n_elast_solve);
    vector<unique_ptr<VectorFunctionCoefficient>> vol_force(n_elast_solve);
    for (int i = 0; i < n_elast_solve; i++)
    {
        // apply boundary loads
        elast[i] = make_unique<LinearElasticitySolver>(state_fes);
        const LoadCase &lc = prob.cases[i];
        for (int j = 0; j < lc.load_attrs.Size(); j++)
        {
            Vector f(dim);
            f(0) = lc.fx[j];
            f(1) = lc.fy[j];
            if (dim == 3) { f(2) = lc.fz[j]; }

            auto load_cf = make_shared<VectorConstantCoefficient>(f);
            elast[i]->AddBoundaryLoad(lc.load_attrs[j], load_cf);
        }

        // apply volume force if it exists
        if (lc.vol_load)
        {
            vol_force[i] = make_unique<VectorFunctionCoefficient>(dim, lc.vol_load);
            for (int a = 0; a < design_domain.attributes.Size(); a++)
            {
                elast[i]->AddVolumeLoad(design_domain.attributes[a], *vol_force[i]);
            }
        }

        // add fixed boundaries
        for (int j = 0; j < lc.clamp_attrs.Size(); j++)
        {
            elast[i]->AddBoundaryID(lc.clamp_attrs[j]);
        }

        // configurate the elast solver
        elast[i]->SetLambda(lambda_simp_cf);
        elast[i]->SetMu(mu_simp_cf);
        elast[i]->SetPreconditionerType(elast_pc);
        elast[i]->SetMonolithicLOROrdering(
            lor_by_vdim ? Ordering::byVDIM : Ordering::byNODES);
        elast[i]->SetPrintLevel(0);
        elast[i]->SetRelTol(1e-10);
        elast[i]->SetAbsTol(1e-14);
        elast[i]->SetMaxIter(1000);
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

    // lifting for dirichlet bc
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
        advect[r] = make_unique<MaterialThicknessSolver>(filter_fes, dgfes, *ray_cf[r], true);
        advect[r]->SetMinv(minv);
        advect[r]->GetSolver().SetTimeStep(dt);      // pseudo-transient time step
        advect[r]->GetSolver().SetTerminalTime(100); // absolute stopping condition
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

    // initialize for normalization
    real_t init_comp = 1.0;

    rho.GetTrueDofs(rho_tv);
    for (int r = 0; r < n_dir; r++) { alpha[r]->GetTrueDofs(alpha_tv[r]); }

    // Initialize checkpoint system
    Checkpoint checkpoint("checkpoints", MPI_COMM_WORLD);
    int start_iteration = 1;
    const int cp_interval = 5;

    MFEM_VERIFY(cp >= 0 && cp <= 2, "-cp must be 0, 1 or 2.");
    MFEM_VERIFY(restart >= 0 && restart <= 2, "-restart must be 0, 1 or 2.");

    // restart = 1: load rho only
    if (restart == 1)
    {
        MFEM_VERIFY(checkpoint.RhoExists(),
                    "Restart from rho requested but no rho file found.");
        MFEM_VERIFY(checkpoint.LoadRho(rho_tv), "Failed to load rho.");

        if (myid == 0)
        {
            mfem::out << "\nWarm start: rho loaded, running from iteration 1\n";
        }
    }
    // restart = 2: full state, resume where the previous run stopped
    else if (restart == 2)
    {
        MFEM_VERIFY(checkpoint.Exists(),
                    "Restart requested but no checkpoint found.");
        MFEM_VERIFY(checkpoint.ValidateCompatibility(ref_levels, order, n_dir),
                    "Checkpoint incompatible with current run parameters.");
        MFEM_VERIFY(checkpoint.Load(rho_tv, alpha_tv),
                    "Failed to load checkpoint data.");

        start_iteration = checkpoint.GetIteration() + 1;   // `it` is 1-indexed
        epsilon = checkpoint.GetEpsilon();
        init_comp = checkpoint.GetInitComp();

        if (myid == 0)
        {
            mfem::out << "\nRestarting from iteration " << start_iteration
                      << " with epsilon = " << epsilon << "\n";
        }
    }

    rho.SetFromTrueDofs(rho_tv);
    for (int r = 0; r < n_dir; r++) { alpha[r]->SetFromTrueDofs(alpha_tv[r]); }

    BlockVector tx_local(toffsets);
    tx_local.GetBlock(0) = rho_tv;
    for (int r = 0; r < n_dir; r++) { tx_local.GetBlock(1 + r) = alpha_tv[r]; }

    Vector a(num_con), c(num_con), d(num_con);
    a = 0.0; c = 1000.0; d = 0.0;
    mfem_mma::MMAOptimizerParallel mma(MPI_COMM_WORLD, toffsets.Last(), num_con, tx_local, a, c, d);

    // Restore MMA state if restarting
    if (restart == 2 && start_iteration > 1 && checkpoint.GetXOld1().Size() > 0)
    {
        mma.RestoreState(checkpoint.GetXOld1(), checkpoint.GetXOld2(),
                         checkpoint.GetLowerAsymptotes(), checkpoint.GetUpperAsymptotes());
        if (myid == 0)
        {
            mfem::out << "MMA state restored from checkpoint.\n";
        }
    }

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
    socketstream sout;

    if (visualization) {
        sout.open(vishost, visport);
        sout.precision(8);

        sout << "parallel " << num_procs << " " << myid << "\n"
            << "solution\n" << design_domain << rho_filter
            << "window_title 'Projected density'\n"
            << "window_geometry 0 0 800 600\n"
            << "colorbar_numberformat '%.2f'\n"
            << "keys c\n" << flush;
    }

    // 9b. Paraview
    ParGridFunction phys_density(&filter_fes);
    std::ostringstream run_tag;
    run_tag << "3dbeam_amax" << alpha_max << "_vf" << vol_fraction;
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
        csv << "it,c,volume,max_rho_a,max_alpha,fival_max,eps,beta,iterErr,iter_time\n";
    }

    // 10. Optimization loop.
    real_t iterationError = 1.0;

    // Track next iteration for epsilon decay and beta doubling
    int next_epsilon_decay = init_it;
    int next_beta_double = init_it + beta_steps;

    // fast-forward the schedule counters 
    if (restart == 2)
    {
        if (decay_int > 0)
        {
            while (next_epsilon_decay < start_iteration) { next_epsilon_decay += decay_int; }
        }
        if (beta_steps > 0)
        {
            while (next_beta_double < start_iteration && beta < beta_max)
            {
                beta *= 2;
                next_beta_double += beta_steps;
            }
        }
        rho_erod_cf.SetBeta(beta);  rho_erod_grad_cf.SetBeta(beta);
        rho_dila_cf.SetBeta(beta);  rho_dila_grad_cf.SetBeta(beta);
        rho_inter_cf.SetBeta(beta);
    }

    double opt_start_time = MPI_Wtime();

    int it = start_iteration;
    for (; (it <= init_it) || (it <= max_it && iterationError > tol); it++)
    {
        double iter_start_time = MPI_Wtime() - opt_start_time;

        // (1) forward filter:  (r_f^2 K + M) ρ~ = M_fc ρ  (+ Dirichlet lifting)
        rho.GetTrueDofs(rho_tv);
        Vector rho_filter_tv(nf);
        filter.Mult(rho_tv, rho_filter_tv);
        rho_filter_tv += rho_filter_lift_tv;
        rho_filter.SetFromTrueDofs(rho_filter_tv);

        // construct dialated desgin coefficients
        ParGridFunction rho_dila_gf(&filter_fes);
        rho_dila_gf.ProjectCoefficient(rho_dila_cf);
        Vector rho_dila_tv(nf);
        rho_dila_gf.GetTrueDofs(rho_dila_tv);

        ParGridFunction rho_dila_grad_gf(&filter_fes);
        rho_dila_grad_gf.ProjectCoefficient(rho_dila_grad_cf);
        Vector rho_dila_grad_tv(nf);
        rho_dila_grad_gf.GetTrueDofs(rho_dila_grad_tv);

        // (2) state solves:  K(ρ~) u = f   (self-adjoint compliance), averaged
        //     over the load cases, together with the adjoint filter rhs
        real_t compliance = 0.0;
        Vector adj_rhs_tv(nf);
        adj_rhs_tv = 0.0;
        double elast_runtime = MPI_Wtime();
        for (int i = 0; i < n_elast_solve; i++)
        {
            elast[i]->SetNeedsAssembly();   // reset assembly flag
            elast[i]->Solve(u);
            compliance += comp.Eval();

            ParLinearForm adj_rhs(&filter_fes);
            adj_rhs.AddDomainIntegrator(new DomainLFIntegrator(dcdrho_tilde_cf));
            adj_rhs.Assemble();
            std::unique_ptr<HypreParVector> adj_rhs_e_tv(adj_rhs.ParallelAssemble());
            adj_rhs_tv += *adj_rhs_e_tv;
        }
        elast_runtime = MPI_Wtime() - elast_runtime;
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

        // (5) advect rho~ along each ray to get the thickness measure rho_a, then the
        //     max-thickness constraint and gradient per direction:
        //       1/2 ∫(rho_a−α_r)² − ε ≤ 0
        //       dR/dalpha_r = (α_r − rho_a) on Gamma_out,r
        //       dR/drho     = M_fc^T N^T (rho_a − α_r)  via the adjoint advection solve
        real_t fi_thick  = -infinity();
        real_t max_rho_a = -infinity();
        real_t max_alpha = -infinity();
        double adv_runtime = MPI_Wtime();
        for (int r = 0; r < n_dir; r++)
        {
            // forward
            advect[r]->SetRhs(rho_dila_tv);
            advect[r]->FSolve();
            const real_t thickres = adv_res[r]->Eval();

            // record max value of rho_a and alpha
            real_t local_max = advect[r]->GetRhoA().Max();
            real_t global_max = local_max;
            MPI_Allreduce(&local_max, &global_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                        advect[r]->GetRhoA().ParFESpace()->GetComm());

            max_rho_a = std::max(max_rho_a, global_max);

            local_max = alpha[r]->Max();
            MPI_Allreduce(&local_max, &global_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                        alpha[r]->ParFESpace()->GetComm());

            max_alpha = std::max(max_alpha, global_max);

            // evaluate adjoint gradient
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

            Vector dGdrho_tilde(advect[r]->GetSensitivity());
            dGdrho_tilde *= rho_dila_grad_tv;
            filter.MultTranspose(dGdrho_tilde, dthick[r].GetBlock(0));

            fival(1 + r) = thickres - epsilon;     // update constraint value
            // dthick[r] /= epsilon;
            dfidx[1 + r] = dthick[r];                    // update constraint gradient

            fi_thick = std::max(fi_thick, fival(1 + r));
        }
        adv_runtime = MPI_Wtime() - adv_runtime;

        // (6) box constraints:  rho ∈ [0,1],  α_r ∈ [alpha_min, alpha_max]  (move limits)
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

        // (7) MMA update on the stacked design  x = [ ρ ; α_0 ; ... ; α_{n_dir-1} ]
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
        double elapsed_time  = MPI_Wtime() - init_time;   // includes the setup block

        // (8) reporting
        if (myid == 0)
        {
            const int w = 12;               // column width
            mfem::out << "\nIteration " << it << '\n' << string(8*w, '=') << "\n\n" << left
                    << setw(w) << "c"
                    << setw(w) << "volume"
                    << setw(w) << "max_rho_a"
                    << setw(w) << "max_alpha"
                    << setw(w) << "fival_max"
                    << setw(w) << "eps"
                    << setw(w) << "beta"
                    << setw(w) << "iterErr" << '\n'
                    << string(8*w, '-') << '\n'
                    << fixed      << setprecision(6) << setw(w) << compliance
                    <<               setprecision(4) << setw(w) << vol
                    <<               setprecision(4) << setw(w) << max_rho_a
                    <<               setprecision(4) << setw(w) << max_alpha
                    << scientific << setprecision(2) << setw(w) << fi_thick
                    <<               setprecision(2) << setw(w) << epsilon
                    << fixed      << setprecision(0) << setw(w) << beta
                    << scientific << setprecision(2) << setw(w) << iterationError << "\n\n";

            // runtime outputs
            const int lw = 18;              // label width
            mfem::out << fixed << setprecision(2) << left
                    << setw(lw) << "elast solve"     << right << setw(8) << elast_runtime << " s\n" << left
                    << setw(lw) << "advection solve" << right << setw(8) << adv_runtime   << " s\n" << left
                    << setw(lw) << "iteration"       << right << setw(8) << iter_runtime  << " s\n" << left
                    << setw(lw) << "total elapsed"   << right << setw(8) << elapsed_time  << " s   "
                    << setprecision(0) << floor(elapsed_time/3600) << "h "
                    << fmod(floor(elapsed_time/60), 60) << "m" << endl;

            csv << it << ','
                << scientific << setprecision(8) << compliance << ','
                << vol << ','
                << max_rho_a << ','
                << max_alpha << ','
                << fi_thick << ','
                << epsilon << ','
                << fixed << setprecision(0) << beta << ','
                << scientific << setprecision(8) << iterationError << ','
                << fixed << setprecision(4) << iter_runtime << '\n';
            csv.flush();
        }

        // (9) tighten the max-thickness tolerance and update beta
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

        // Checkpoint every cp_interval iterations
        if (cp > 0 && it % cp_interval == 0)
        {
            rho.GetTrueDofs(rho_tv);

            if (cp == 1)
            {
                // rho only
                checkpoint.SaveRho(rho_tv, it);
            }
            else
            {
                for (int r = 0; r < n_dir; r++) { alpha[r]->GetTrueDofs(alpha_tv[r]); }

                // Save with MMA state for proper restart
                checkpoint.Save(rho_tv, alpha_tv,
                                mma.GetXOld1(), mma.GetXOld2(),
                                mma.GetLowerAsymptotes(), mma.GetUpperAsymptotes(),
                                it, n_dir, ref_levels, order, epsilon, init_comp);
            }
        }

        // physical density for both GLVis and the ParaView archive
        phys_density.ProjectCoefficient(rho_inter_cf);
        // --- PLAIN SIMP ---
        // phys_density.ProjectCoefficient(simp_cf);

        if (visualization)
        {
            sout << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n" << design_domain << phys_density << flush;
        }

        // save every 50 iterations
        // if (paraview && it % 50 == 0)
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

    // save the final solution
    // if (paraview)
    // {
    //     paraview_dc.SetCycle(it - 1);
    //     paraview_dc.SetTime(it - 1);
    //     paraview_dc.Save();
    // }

    // 11. Post process the solution mesh.
    if (paraview)
    {
        // phys_density lives on the design domain, so threshold that, not pmesh
        SaveSolidSubmesh(design_domain, rho_filter, phys_density, run_tag.str(), order, 0.4);
    }

    return 0;
}

// One advection direction field per ray. The 3D ray layout is problem specific
// (not a sweep of a plane), so the directions are listed here explicitly.
// An empty vector means n_dir = 0, i.e. the thickness constraint is off.
std::vector<std::unique_ptr<VectorCoefficient>> SetupRays(int dim)
{
    std::vector<std::unique_ptr<VectorCoefficient>> rays;

    // placeholder: push one VectorConstantCoefficient (or VectorFunctionCoefficient
    // for a diverging cone) per ray direction, e.g.
    //
    //   Vector axis(dim); axis = 0.0; axis(2) = 1.0;
    //   rays.push_back(std::make_unique<VectorConstantCoefficient>(axis));

    MFEM_CONTRACT_VAR(dim);
    return rays;
}

// save the thresholded design by clipping from the max value
void SaveSolidSubmesh(ParMesh &pmesh, ParGridFunction &desi_density,
                      ParGridFunction &phys_density, const std::string &run_tag, 
                      int order, real_t threshold)
{
    const int sol_attr = 1000;

    for (int i = 0; i < pmesh.GetNE(); i++)
    {
        real_t elem_max = -infinity();

        ElementTransformation *T = pmesh.GetElementTransformation(i);
        const FiniteElement *fe = desi_density.FESpace()->GetFE(i);
        const IntegrationRule &ir = fe->GetNodes();

        for (int j = 0; j < ir.GetNPoints(); j++)
        {
            const IntegrationPoint &ip = ir.IntPoint(j);
            T->SetIntPoint(&ip);
            real_t val = desi_density.GetValue(*T, ip);
            elem_max = max(elem_max, val);
        }

        if (elem_max > threshold)
        {
            pmesh.SetAttribute(i, sol_attr);
        }
    }
    pmesh.SetAttributes();

    Array<int> sol_mesh_attrs(1);
    sol_mesh_attrs[0] = sol_attr;

    ParSubMesh sol_submesh = ParSubMesh::CreateFromDomain(pmesh, sol_mesh_attrs);
    ParFiniteElementSpace filter_subfes(&sol_submesh, desi_density.ParFESpace()->FEColl());

    ParGridFunction desi_density_sub(&filter_subfes);
    ParGridFunction phys_density_sub(&filter_subfes);

    ParSubMesh::Transfer(desi_density, desi_density_sub);
    ParSubMesh::Transfer(phys_density, phys_density_sub);

    // save in separate paraview
    ParaViewDataCollection dc(run_tag, &sol_submesh);
    dc.SetPrefixPath("ParaView_fsol");
    dc.SetLevelsOfDetail(order);
    dc.SetDataFormat(VTKFormat::BINARY);
    dc.SetHighOrderOutput(true);
    dc.RegisterField("density", &phys_density_sub);
    dc.RegisterField("rho_filter", &desi_density_sub);
    dc.Save();
}

// 3x1x1 cantilever beam on a built-in hex grid, the default when no -m is given.
static MeshProblem SetupCartesianBeam(Mesh &mesh)
{
    mesh = Mesh::MakeCartesian3D(12, 4, 4, Element::HEXAHEDRON, 3.0, 1.0, 1.0);

    MeshProblem p;
    p.domain_attr = Array<int>({ 1 });          // MakeCartesian3D tags every element 1

    // attrs: 1 z=0, 2 y=0, 3 x=lx, 4 y=ly, 5 x=0 (clamped), 6 z=lz.
    p.outer_bdr_attrs = Array<int>({1, 2, 3, 4, 6});

    p.cases.resize(1);
    LoadCase &lc = p.cases[0];
    lc.clamp_attrs = Array<int>({ 5 });

    // body force pushing down (-Z) on a small patch near the bottom right corner.
    lc.vol_load = [](const Vector &x, Vector &f)
    {
        const int dim = x.Size();

        f.SetSize(dim);
        f = 0.0;

        real_t radius = 0.05;
        real_t center_x = 2.9;
        real_t center_z = 0.1;

        bool x_in_range = (x[0] < center_x + radius) && (x[0] > center_x - radius);
        bool z_in_range = (x[2] < center_z + radius) && (x[2] > center_z - radius);

        if (x_in_range && z_in_range) f(2) = -1.0;
    };

    return p;
}

// placeholder for the first 3D mesh
static MeshProblem SetupMesh3D(Mesh &mesh, const char *mesh_file)
{
    mesh = Mesh(mesh_file);

    MeshProblem p;
    // p.domain_attr     = Array<int>({ ... });   // designable subdomain(s)
    // p.outer_bdr_attrs = Array<int>({ ... });   // free surfaces: rho~ = 0

    // p.cases.resize(1);

    // LoadCase &lc = p.cases[0];
    // lc.clamp_attrs = Array<int>({ ... });
    // lc.load_attrs  = Array<int>({ ... });
    // lc.fx          = Array<real_t>({ ... });
    // lc.fx          = Array<real_t>({ ... });
    // lc.fx          = Array<real_t>({ ... });

    return p;
}

// select the per-mesh setup from the mesh file name
MeshProblem loadMesh(int myid, const char *mesh_file, Mesh &mesh)
{
    // no -m: fall back to the built-in Cartesian beam
    if (!mesh_file || mesh_file[0] == '\0')
    {
        return SetupCartesianBeam(mesh);
    }

    if (strstr(mesh_file, "placeholder.msh") != NULL)
    {
        return SetupMesh3D(mesh, mesh_file);
    }

    if (myid == 0) { mfem::out << "invalid mesh file" << endl; }
    return MeshProblem();
}
