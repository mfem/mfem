// Linear elasticity topology optimization with a thickness constraint.
//
// Sample run:  mpirun -np 8 ./ElastTopOpt_3d
//              mpirun -np 8 ./ElastTopOpt_3d -r 1 -vf 0.5
//

#include "mfem.hpp"
#include "ElastTopOpt.hpp"
#include "qoi.hpp"
#include "pseudo_transient_solver.hpp"
#include "../../diffusion_mass_solver.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../mtop_solvers.hpp"
#include "checkpoint.hpp"
#include <memory>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;
using namespace mfem;

Vector ray_vector(const int r, const int n_dir, const int dim);
void load(const Vector &x, Vector &f);

int main(int argc, char *argv[])
{
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // Timer
    double init_time = MPI_Wtime();

    // 1. Options.
    const char *mesh_file = "";
    int    ref_levels   = 0;
    int    order        = 2;
    real_t r_f          = 0.03;       // min filter length
    real_t vol_fraction = 0.4;
    int    max_it       = 300;
    real_t tol          = 1e-4;       // stopping tol on iteration error
    real_t move         = 0.1;        // MMA move limit
    real_t epsilon      = 1e-2;       // thickness residual tolerance
    const int seed      = 0;

    // Thickness constraint parameters
    real_t alpha_min    = 1e-6;        // minimum thickness bound
    real_t alpha_max    = 1.0;        // maximum thickness bound

    bool visualization = true;
    bool paraview      = false;

    const real_t E_min    = 1e-6;     // SIMP void stiffness
    const real_t E_max    = 1.0;      // SIMP E max
    const real_t exponent = 3.0;      // SIMP exponent

    real_t decay       = 0.7;
    real_t eps_floor   = 1e-5;
    int    decay_int   = 20;
    int    decay_start = 10;

    bool pa      = false;
    bool restart = false;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "mesh file to use");
    args.AddOption(&ref_levels, "-r", "--refine", "uniform refinement levels");
    args.AddOption(&order, "-o", "--order", "finite element order");
    args.AddOption(&vol_fraction, "-vf", "--volume-fraction", "volume fraction");
    args.AddOption(&r_f, "-rf", "--r_fwidth", "min filter width");
    args.AddOption(&alpha_min, "-amin", "--alpha_min", "minimum thickness bound");
    args.AddOption(&alpha_max, "-amax", "--alpha_max", "maximum thickness bound");
    args.AddOption(&epsilon, "-e", "--epsilon", "thickness residual tolerance (initial)");
    args.AddOption(&decay, "-d", "--decay", "decay rate of epsilon");
    args.AddOption(&decay_int, "-di", "--decay_int", "decay interval of epsilon");
    args.AddOption(&decay_start, "-ds", "--decay_start", "iteration count to start the decay");
    args.AddOption(&max_it, "-mi", "--max-it", "max optimization iterations");
    args.AddOption(&tol, "-tol", "--tol", "stopping tol on max design change");
    args.AddOption(&move, "-mv", "--move", "MMA move limit");
    args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                    "store solution in paraview");
    args.AddOption(&visualization, "-vis", "--visualization",
                    "-no-vis", "--no-visualization", "enable GLVis visualization");
    args.AddOption(&restart, "-restart", "--restart", "-no-restart", "--no-restart",
                    "restart from checkpoint");
    args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa", "--no-partial-assembly",
                    "enable partial assembly (recommended for large problems)");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0) { args.PrintUsage(cout); }
        return 1;
    }
    if (myid == 0) { args.PrintOptions(cout); }

    // 2. Load the mesh and construct corresponding attributes.
    Mesh mesh = (mesh_file[0] != '\0') ? Mesh(mesh_file) 
        : Mesh::MakeCartesian3D(3, 1, 1, Element::HEXAHEDRON, 3.0, 1.0, 1.0);

    // 3. Preprocess the mesh.
    const int dim = mesh.Dimension();
    const int clamp_attr = 5;  // X=0 face (left end)

    for (int l = 0; l < ref_levels; l++)
    {
        mesh.UniformRefinement();
    }
   
    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    // 3b. Build ray fields and outflow submeshes for thickness constraint
    Array<int> candidate_be;        // all boundary elements
    Array<int> candidate_attr;
    for (int i = 0; i < pmesh.GetNBE(); i++)
    {
        const int el_attr = pmesh.GetBdrAttribute(i);

        candidate_be.Append(i);
        candidate_attr.Append(el_attr);
    }

    // const int n_dir = 4;
    const int n_dir = 0;  // Set to 0 to test without thickness constraints
    vector<unique_ptr<VectorConstantCoefficient>> ray_cf(n_dir);
    vector<unique_ptr<ParSubMesh>> outflow(n_dir);

    for (int r = 0; r < n_dir; r++)
    {
        Vector v = ray_vector(r, n_dir, dim);
        ray_cf[r] = make_unique<VectorConstantCoefficient>(v);

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
    H1_FECollection control_fec(order-1, dim);
    ParFiniteElementSpace state_fes(&pmesh, &state_fec, dim, Ordering::byVDIM);
    ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);
    ParFiniteElementSpace control_fes(&pmesh, &control_fec);

    // 4b. Define the DG space for thickness evaluation
    DG_FECollection dgfec(order, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace dgfes(&pmesh, &dgfec);

    vector<unique_ptr<DG_FECollection>> sub_dg_fec(n_dir);
    vector<unique_ptr<ParFiniteElementSpace>> sub_dg_fes(n_dir);

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
    rho += vol_fraction;
    rho_filter = vol_fraction;

    // initilize alpha in the DG space
    vector<unique_ptr<ParGridFunction>> alpha(n_dir);
    for (int r = 0; r < n_dir; r++)
    {
        const int sub_dim = outflow[r]->Dimension();
        sub_dg_fec[r] = make_unique<DG_FECollection>(order, sub_dim, BasisType::GaussLobatto);
        sub_dg_fes[r] = make_unique<ParFiniteElementSpace>(outflow[r].get(), sub_dg_fec[r].get());

        alpha[r] = make_unique<ParGridFunction>(sub_dg_fes[r].get());
        *alpha[r] = alpha_max * vol_fraction;   // initialize thickness variables
    }

    // Lame constants and SIMP material coefficients
    ConstantCoefficient one_cf(1.0);
    ConstantCoefficient E_cf(3.0), nu_cf(0.3);
    IsoElasticyLambdaCoeff lambda_cf(&E_cf, &nu_cf);
    IsoElasticySchearCoeff mu_cf(&E_cf, &nu_cf);
    SIMPCoefficient simp_cf(&rho_filter, E_min, E_max, exponent);                // r(rho~)
    SIMPGradCoefficient simp_grad_cf(&rho_filter, E_min, E_max, exponent);       // r'(rho~)
    ProductCoefficient E_simp(simp_cf, E_cf);           // r(rho~) * E0
    GridFunctionCoefficient rho_cf(&rho);

    // 7. Construct the solvers.
    // 7a. Linear elasticity solver
    IsoLinElasticSolver elast(&pmesh, order, pa);
    VectorFunctionCoefficient vol_force(dim, load);
    elast.SetVolForce(vol_force);
    elast.AddDispBC(clamp_attr, -1, 0.0);
    elast.SetMaterial(E_simp, nu_cf);
    elast.SetLinearSolver(1e-10, 1e-14, 10000);

    StrainEnergyDensityCoefficient energy_cf(&lambda_cf, &mu_cf,
                                             &elast.GetDisplacements());
    ProductCoefficient prod(energy_cf, simp_grad_cf);   // r'(rho~) * psi0
    ProductCoefficient dcdrho_cf(-1.0, prod);           // dc/drho~ = -r'(rho~) * psi0
    Compliance comp(MPI_COMM_WORLD, &filter_fes, simp_cf, energy_cf);

    // 7b. Minimum length scale filter solver (diffusion-mass PDE filter).
    PDEFilter filter(control_fes, filter_fes);
    filter.SetFilterRadius(r_f);
    DiffusionMassSolver &filter_solver = filter.GetSolver();
    for (int i = 1; i <= pmesh.bdr_attributes.Max(); i++)
    {
        if(i == clamp_attr) continue;
        filter_solver.AddBoundaryID(i);
    }
    filter.Assemble();
    
    // 7c. Thickness constriant solver
    vector<unique_ptr<MaterialThicknessSolver>> advect(n_dir);
    vector<unique_ptr<AdvectThicknessResidual>> adv_res(n_dir);

    DGMassInverse minv(dgfes);

    for (int r = 0; r < n_dir; r++)
    {
        advect[r] = make_unique<MaterialThicknessSolver>(filter_fes, dgfes, *ray_cf[r], pa);
        advect[r]->SetMinv(minv);
        adv_res[r] = make_unique<AdvectThicknessResidual>(*outflow[r], advect[r]->GetRhoA(), *alpha[r]);
    }

    // set timestep according to the CFL condition
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
        advect[r]->GetSolver().SetTimeStep(dt);
        advect[r]->GetSolver().SetTerminalTime(3.0);
    }

    // 8. Volume constraint data:  g(rho) = (1, rho)/Vstar - 1.
    ParLinearForm vol_form(&control_fes);
    vol_form.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
    vol_form.Assemble();
    std::unique_ptr<HypreParVector> vol_w(vol_form.ParallelAssemble());

    real_t domain_volume;
    real_t loc = vol_w->Sum();
    MPI_Allreduce(&loc, &domain_volume, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
    const real_t Vstar = vol_fraction * domain_volume;

    // 9. MMA optimizer and per-iteration work vectors.
    // Design variables: x = [ rho ; alpha_0 ; alpha_1 ; ... ; alpha_{n_dir-1} ]
    const int n = control_fes.GetTrueVSize();       // local rho design variables
    vector<int> m(n_dir);
    for (int r = 0; r < n_dir; r++) { m[r] = sub_dg_fes[r]->GetTrueVSize(); }

    Array<int> toffsets(n_dir + 2);
    toffsets[0] = 0;
    toffsets[1] = n;
    for (int r = 0; r < n_dir; r++) { toffsets[2 + r] = m[r]; }
    toffsets.PartialSum();

    const int num_con = 1 + n_dir;                  // constraints: volume + one thickness per ray
    Vector dcdrho(n), fival(num_con);
    Vector rho_tv(n), rho_old(n);
    vector<Vector> alpha_tv(n_dir);

    // initialize for normalization
    real_t init_comp = 1.0;
    vector<real_t> init_thickness_res(n_dir, 1.0);

    rho.GetTrueDofs(rho_tv);
    for (int r = 0; r < n_dir; r++) { alpha[r]->GetTrueDofs(alpha_tv[r]); }
    
    // Initialize checkpoint system and restart from a saved design if requested.
    Checkpoint checkpoint("checkpoints", MPI_COMM_WORLD);
    int start_iteration = 0;
    const int cp_interval = 5;
    MFEM_VERIFY(checkpoint.RestartIfRequested(restart, ref_levels, order, n_dir,
                                        rho_tv, alpha_tv, start_iteration, epsilon,
                                        init_comp, init_thickness_res),
                                    "Failed to restart from checkpoint.");
    rho.SetFromTrueDofs(rho_tv);
    for (int r = 0; r < n_dir; r++) { alpha[r]->SetFromTrueDofs(alpha_tv[r]); }

    BlockVector tx_local(toffsets);
    tx_local.GetBlock(0) = rho_tv;
    for (int r = 0; r < n_dir; r++) { tx_local.GetBlock(1 + r) = alpha_tv[r]; }
    mfem_mma::MMAOptimizerParallel mma(MPI_COMM_WORLD, toffsets.Last(), num_con, tx_local);

    BlockVector tx_min(toffsets), tx_max(toffsets);
    BlockVector df0dx(toffsets);                     // objective gradient: df0/dx = [ dc/drho ; 0 ; ... ; 0 ]
    BlockVector dvol(toffsets);                       // volume constraint gradient (constant)
    dvol.GetBlock(0) = *vol_w;  dvol.GetBlock(0) /= Vstar;
    for (int r = 0; r < n_dir; r++) { dvol.GetBlock(1 + r) = 0.0; }

    // Thickness constraint gradients: one BlockVector per ray direction.
    // Only block(0) (∂/∂rho) and block(1+r) (∂/∂alpha_r) are nonzero.
    vector<BlockVector> dthick(n_dir, BlockVector(toffsets));

    vector<Vector> dfidx(num_con);
    dfidx[0] = dvol;                                  // dfidx[1..n_dir] updated each iteration

    // 10. Visualizations and data information
    ParGridFunction phys_density(&filter_fes);
    phys_density.ProjectCoefficient(simp_cf);

    // 10a. GLVis
    char vishost[] = "localhost";  int visport = 19916;
    socketstream sout;
    
    // Initialize GLVis display
    if (visualization) { 
        sout.open(vishost, visport);  
        sout.precision(8);

        sout << "parallel " << num_procs << " " << myid << "\n"
             << "solution\n" << pmesh << phys_density
             << "window_title 'Design density r(rho~)'\n"
             << "keys Rjlc\n" << flush;
    }

    // 10b. Paraview
    std::ostringstream run_tag;
    run_tag << "3dbeam_amax" << alpha_max << "_vf" << vol_fraction;
    ParaViewDataCollection paraview_dc(run_tag.str(), &pmesh);

    if (paraview) {
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.RegisterField("density", &phys_density);
    }

    // Initialization block runtime.
    double init_block_time = MPI_Wtime() - init_time;
    if (myid == 0)
    {
        mfem::out << "\nInitialization block runtime: " << fixed << setprecision(4)
                   << init_block_time << " s\n";
    }

    // 10c. CSV convergence log (rank 0 only).
    std::ofstream csv;
    if (myid == 0)
    {
        csv.open("convergence.csv");
        csv << "it,compliance,vol,res_max,eps,iterErr,iter_time\n";
    }

    // 11. Optimization loop.
    int k = start_iteration;
    real_t iterationError = 1.0;

    double opt_start_time = MPI_Wtime();  // Reference time for the optimization loop

    for (; k < max_it && iterationError > tol; k++)
    {
        double iter_start_time = MPI_Wtime() - opt_start_time;

        if (myid == 0)
        {
            mfem::out << "\niteration " << k + 1
                        << "\n=================================="
                        << "===================================="
                        << "\nelapsed time at start: " << fixed << setprecision(2)
                        << iter_start_time << " s\n" << endl;
        }

        if (k % decay_int == 0 && k > decay_start)
        {
            epsilon = std::max(epsilon * decay, eps_floor);
        }

        // (1) forward filter:  (r_f^2 K + M) ρ~ = M_fc ρ
        rho.GetTrueDofs(rho_tv);
        Vector rho_filter_tv(filter_fes.GetTrueVSize());
        filter.Mult(rho_tv, rho_filter_tv);
        rho_filter.SetFromTrueDofs(rho_filter_tv);

        // (2) State solve:  K(ρ~) u = f   (compliance objective is self-adjoint)
        elast.Assemble();
        elast.FSolve();
        elast.GetDisplacements();     // refresh fdisp from sol so energy_cf sees new u
        real_t compliance = comp.Eval();

        // (3) Adjoint filter + objective gradient:
        //     w~  = (r_f^2 K + M)^{-1} ∫ (-r'(ρ~) ψ_0) φ_i dΩ
        //     dc/drho = M_fc^T w~
        ParLinearForm adj_rhs(&filter_fes);
        adj_rhs.AddDomainIntegrator(new DomainLFIntegrator(dcdrho_cf));
        adj_rhs.Assemble();
        std::unique_ptr<HypreParVector> adj_rhs_tv(adj_rhs.ParallelAssemble());
        filter.MultTranspose(*adj_rhs_tv, dcdrho);

        // (4) Objective gradient:  df0/dx = [ dc/drho ; 0 ; ... ; 0 ]
        df0dx.GetBlock(0) = dcdrho;
        for (int r = 0; r < n_dir; r++) { df0dx.GetBlock(1 + r) = 0.0; }

        // (5) Thickness constraint evaluation and gradient (one per ray direction)
        real_t res_max = -infinity();
        real_t res_eps = 0.5 * epsilon * epsilon;
        for (int r = 0; r < n_dir; r++)
        {
            advect[r]->SetRhs(rho_filter_tv);
            advect[r]->FSolve();
            const real_t thickness_res = adv_res[r]->Eval();
            if (k == 0) { init_thickness_res[r] = thickness_res; }

            dthick[r] = 0.0;

            Vector dGdrhoa;
            adv_res[r]->GetGrad(dGdrhoa, dthick[r].GetBlock(1 + r));

            // Transfer dG/drho_a from submesh back to full-domain DG space
            ParGridFunction g_sub(sub_dg_fes[r].get());  g_sub.SetFromTrueDofs(dGdrhoa);
            ParGridFunction g_full(&dgfes);              g_full = 0.0;
            outflow[r]->Transfer(g_sub, g_full);
            Vector rhs_full;  g_full.GetTrueDofs(rhs_full);

            // Chain rule adjoint solve: dG/dρ = M_fc^T N^T g
            advect[r]->SetAdjointRhs(rhs_full);
            advect[r]->ASolve();
            filter.MultTranspose(advect[r]->GetSensitivity(), dthick[r].GetBlock(0));

            dthick[r] /= init_thickness_res[r];
            dfidx[1 + r] = dthick[r];

            fival(1 + r) = (thickness_res - res_eps) / init_thickness_res[r];
            res_max = std::max(res_max, fival(1 + r));
        }

        // (6) MMA update
        for (int r = 0; r < n_dir; r++) { alpha[r]->GetTrueDofs(alpha_tv[r]); }

        // Volume constraint:  ∫ρ dΩ / V* - 1 ≤ 0
        real_t vol = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / domain_volume;
        fival(0) = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / Vstar - 1.0;

        // Box constraints with move limits: ρ ∈ [0,1], α_i ∈ [alpha_min, alpha_max]
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

        // Stacked design variables: x = [ ρ ; α_0 ; ... ; α_{n_dir-1} ]
        tx_local.GetBlock(0) = rho_tv;
        for (int r = 0; r < n_dir; r++) { tx_local.GetBlock(1 + r) = alpha_tv[r]; }
        rho_old = rho_tv;

        // Normalize compliance and gradient by initial value
        if(k == 0) init_comp = compliance;
        compliance /= init_comp;
        df0dx /= init_comp;

        // print out advection solve info
        for (int r = 0; r < n_dir; r++)
        {
            real_t local_max = advect[r]->GetRhoA().Max();
            real_t global_max = local_max;
            MPI_Allreduce(&local_max, &global_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                        advect[r]->GetRhoA().ParFESpace()->GetComm());

            real_t local_alpha_max = alpha[r]->Max();
            real_t global_alpha_max = local_alpha_max;
            MPI_Allreduce(&local_alpha_max, &global_alpha_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX,
                        alpha[r]->ParFESpace()->GetComm());

            if (myid == 0)
            {
                mfem::out << "ray direction [" << r << "]"
                        << ":    max rho_a = " << fixed << setprecision(8) << global_max 
                        << ",    max alpha = " << fixed << setprecision(8) << global_alpha_max
                        << ",    residual = " << scientific << setprecision(8) << fival[r+1] << endl;
            }
        }
        mma.Update(tx_local, df0dx, compliance, fival, dfidx.data(), tx_min, tx_max);
        rho.SetFromTrueDofs(tx_local.GetBlock(0));
        for (int r = 0; r < n_dir; r++) { alpha[r]->SetFromTrueDofs(tx_local.GetBlock(1 + r)); }

        // Measure design change (iteration error)
        ParGridFunction rho_old_gf(&control_fes);
        rho_old_gf.SetFromTrueDofs(rho_old);
        iterationError = rho_old_gf.ComputeL1Error(rho_cf);

        double iter_end_time = MPI_Wtime() - opt_start_time;
        double iter_runtime = iter_end_time - iter_start_time;

        if (myid == 0)
        {
            mfem::out << "c = " << scientific << setprecision(6) << compliance
                      << "   vol = " << fixed << setprecision(4) << vol
                      << "   res_max = " << scientific << setprecision(3) << res_max
                      << "   eps = " << fixed << setprecision(4) << epsilon
                      << "   iterErr = " << setprecision(4) << iterationError
                      << "\nelapsed time at end: " << fixed << setprecision(2)
                      << iter_end_time << " s"
                      << ",   iteration runtime: " << setprecision(2) << iter_runtime << " s" << endl;

            csv << k + 1 << ','
                << scientific << setprecision(8) << compliance << ','
                << vol << ','
                << res_max << ','
                << epsilon << ','
                << iterationError << ','
                << iter_runtime << '\n';
            csv.flush();
        }

        // Checkpoint every n iterations
        if ((k + 1) % cp_interval == 0)
        {
            rho.GetTrueDofs(rho_tv);
            for (int r = 0; r < n_dir; r++) { alpha[r]->GetTrueDofs(alpha_tv[r]); }
            checkpoint.Save(rho_tv, alpha_tv, k + 1, n_dir, ref_levels, order, epsilon,
                            init_comp, init_thickness_res);
        }

        // Update physical density r(ρ~) for visualization
        phys_density.ProjectCoefficient(simp_cf);

        if (visualization)
        {
            sout << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n" << pmesh << phys_density
                << "window_title 'Design density r(rho~)'"  << flush;
        }

        if (paraview)
        {
            paraview_dc.SetCycle(k + 1);
            paraview_dc.SetTime(k + 1);
            paraview_dc.Save();
        }

        // cin.get();
    }

    double total_runtime = MPI_Wtime() - init_time;

    if (myid == 0)
    {
        csv.close();
        mfem::out << "\nfinished after " << k << " iterations"
                  << "\ntotal runtime is " << total_runtime << " s\n";
    }

    // Option: save only the final solution instead of all iterations
    // if (paraview)
    // {
    //     paraview_dc.SetCycle(k);
    //     paraview_dc.SetTime(k);
    //     paraview_dc.Save();
    // }

    return 0;
}

Vector ray_vector(const int r, const int n_dir, const int dim)
{
    const real_t phi = r * (M_PI / n_dir);
    Vector v(dim);

    v(0) = 0;
    v(1) = cos(phi);
    v(2) = sin(phi);

    return v;
}

// load applies downward (-Z direction) near the right end, bottom corner
void load(const Vector &x, Vector &f)
{
    const int dim = x.Size();

    f.SetSize(dim);
    f = 0.0;

    real_t radius = 0.05;
    real_t center_x = 2.9;
    real_t center_z = 0.1;

    bool x_in_range = (x[0] < center_x + radius) && (x[0] > center_x - radius);
    bool z_in_range = (x[2] < center_z + radius) && (x[2] > center_z - radius);

    if (x_in_range && z_in_range)
        f(2) = -1.0;
}