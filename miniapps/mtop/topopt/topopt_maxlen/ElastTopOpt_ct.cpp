// Linear elasticity topology optimization with a thickness constraint.
//
// Sample run:  mpirun -np 4 ./ElastTopOpt_ct -m "../data/a_circular_5_holes.msh"
//              mpirun -np 4 ./ElastTopOpt_ct -m "../data/b_circular_9_holes.msh"

#include "mfem.hpp"
#include "ElastTopOpt.hpp"
#include "qoi.hpp"
#include "../../diffusion_mass_solver.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../mtop_solvers.hpp"
#include <memory>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;
using namespace mfem;

void loadMesh(int myid, const char *mesh_file,
                Mesh &mesh,
                Array<int> &domain_attr,
                vector<Array<int>> &clamp_attrs,
                vector<Array<int>> &load_attrs,
                vector<Array<int>> &load_fx,
                vector<Array<int>> &load_fy,
                int &n_elast_solve,
                int &outer_bdr_attr);


int main(int argc, char *argv[])
{
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 1. Options.
    const char *mesh_file     = nullptr;
    int    ref_levels   = 0;
    int    order        = 2;
    int    n_dir        = 8;          // number of ray directions around the domain
    real_t r_f          = 0.03;       // min filter length
    real_t vol_fraction = 0.5;
    int    max_it       = 300;
    real_t tol          = 1e-3;       // stopping tol on iteration error
    real_t move         = 0.2;        // MMA move limit
    real_t epsilon      = 1e-2;       // thickness residual tolerance
    real_t domain_init  = 0.1;
    const int seed      = 0;

    // Thickness constraint parameters
    real_t alpha_min    = 1e-6;        // minimum thickness bound
    real_t alpha_max    = 0.5;        // maximum thickness bound

    bool visualization = true;
    bool paraview      = false;

    const real_t E_min    = 1e-6;     // SIMP void stiffness
    const real_t E_max    = 1.0;      // SIMP E max
    const real_t exponent = 3.0;      // SIMP exponent

    real_t decay = 0.7;
    real_t eps_floor = 1e-10;
    int decay_int = 20;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "mesh file to use", true);
    args.AddOption(&ref_levels, "-r", "--refine", "uniform refinement levels");
    args.AddOption(&order, "-o", "--order", "finite element order");
    args.AddOption(&n_dir, "-nr", "--n-rays", "number of ray directions around the domain");
    args.AddOption(&vol_fraction, "-vf", "--volume-fraction", "volume fraction");
    args.AddOption(&r_f, "-rf", "--r_fwidth", "min filter width");
    args.AddOption(&alpha_min, "-amin", "--alpha_min", "minimum thickness bound");
    args.AddOption(&alpha_max, "-amax", "--alpha_max", "maximum thickness bound");
    args.AddOption(&epsilon, "-e", "--epsilon", "thickness residual tolerance (initial)");
    args.AddOption(&decay, "-d", "--decay", "decay rate of epsilon");
    args.AddOption(&decay_int, "-di", "--decay_int", "decay interval of epsilon");
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
    if (!mesh_file)
    {
        if (myid == 0) { cout << "Error: -m <mesh file> is required." << endl; }
        return 1;
    }
    if (myid == 0) { args.PrintOptions(cout); }

    // 2. Load the mesh and construct corresponding attributes.
    Mesh mesh;
    Array<int> domain_attr;
    vector<Array<int>> clamp_attrs, load_attrs, load_fx, load_fy;
    int n_elast_solve = 0;
    int outer_bdr_attr = 0;

    loadMesh(myid, mesh_file, mesh, domain_attr, clamp_attrs, 
        load_attrs, load_fx, load_fy,
        n_elast_solve, outer_bdr_attr);
    
    // 3. Preprocess the mesh.
    const int dim = mesh.Dimension();
    for (int l = 0; l < ref_levels; l++)
    {
        mesh.UniformRefinement();
    }
   
    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    ParSubMesh design_domain = ParSubMesh::CreateFromDomain(pmesh, domain_attr);

    // 3b. Build ray fields and outflow submeshes
    constexpr real_t pi = M_PI;
    
    Array<int> candidate_be;        // extract only the outer boundary elements
    for (int i = 0; i < pmesh.GetNBE(); i++)
    {
        if (pmesh.GetBdrAttribute(i) != outer_bdr_attr) continue;
        candidate_be.Append(i);
    }

    vector<unique_ptr<VectorConstantCoefficient>> ray_cf(n_dir);
    vector<unique_ptr<ParSubMesh>> outflow(n_dir);

    for (int r = 0; r < n_dir; r++)
    {
        const real_t theta = 2.0 * pi * r / n_dir;
        Vector v(dim);  v(0) = cos(theta);  v(1) = sin(theta);
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
            pmesh.SetBdrAttribute(candidate_be[k], outer_bdr_attr);
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
             << ",  state dofs = "  << filter_size
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

    // Set up the thickness design variables, one set per ray direction.
    // solving rho_a requires DG, so alpha also lives in DG
    DG_FECollection dgfec(order, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace dgfes(&pmesh, &dgfec);

    vector<unique_ptr<DG_FECollection>> sub_dg_fec(n_dir);
    vector<unique_ptr<ParFiniteElementSpace>> sub_dg_fes(n_dir);
    vector<unique_ptr<ParGridFunction>> rho_a(n_dir);
    vector<unique_ptr<ParGridFunction>> alpha(n_dir);
    vector<unique_ptr<PseudoTransientSolver>> advect(n_dir);
    vector<unique_ptr<AdvectThicknessResidual>> adv_res(n_dir);

    for (int r = 0; r < n_dir; r++)
    {
        const int sub_dim = outflow[r]->Dimension();
        sub_dg_fec[r] = make_unique<DG_FECollection>(order, sub_dim, BasisType::GaussLobatto);
        sub_dg_fes[r] = make_unique<ParFiniteElementSpace>(outflow[r].get(), sub_dg_fec[r].get());

        rho_a[r] = make_unique<ParGridFunction>(&dgfes);
        alpha[r] = make_unique<ParGridFunction>(sub_dg_fes[r].get());
        *rho_a[r] = domain_init;
        *alpha[r] = domain_init;   // initialize to mid-range

        advect[r] = make_unique<PseudoTransientSolver>(*rho_a[r], rho_filter, *ray_cf[r]);
        adv_res[r] = make_unique<AdvectThicknessResidual>(*outflow[r], advect[r]->GetRhoA(), *alpha[r]);
    }

    // 6a. Set timestep according to the CFL condition (shared across rays).
    real_t cfl = 0.5;
    real_t hmin = infinity();
    for (int i = 0; i < pmesh.GetNE(); i++)
    {
        hmin = min(pmesh.GetElementSize(i, 1), hmin);
    }
    MPI_Allreduce(MPI_IN_PLACE, &hmin, 1,  MPITypeMap<real_t>::mpi_type, MPI_MIN,
                    pmesh.GetComm());

    real_t dt = cfl * hmin / (2 * order + 1);
    for (int r = 0; r < n_dir; r++) { advect[r]->SetTimeStep(dt); advect[r]->SetTerminalTime(3.0); }

    // Lame constants and SIMP material coefficients
    ConstantCoefficient one_cf(1.0);
    ConstantCoefficient E_cf(3.0), nu_cf(0.3);
    IsoElasticyLambdaCoeff lambda_cf(&E_cf, &nu_cf);
    IsoElasticySchearCoeff mu_cf(&E_cf, &nu_cf);
    SIMPCoefficient simp_cf(&rho_filter, E_min, E_max, exponent);                // r(rho~)
    SIMPGradCoefficient simp_grad_cf(&rho_filter, E_min, E_max, exponent);       // r'(rho~)
    ProductCoefficient E_simp(simp_cf, E_cf);           // r(rho~) * E0

    // 7. Construct the solvers.
    // 7a. Linear elasticity solver, 2 solves with different load
    vector<unique_ptr<IsoLinElasticSolver>> elast(n_elast_solve);
    for (int i = 0; i < n_elast_solve; i++) 
    {
        elast[i] = make_unique<IsoLinElasticSolver>(&design_domain, order);
        for (int j = 0; j < load_attrs[i].Size(); j++)
        {
            elast[i]->AddSurfLoad(load_attrs[i][j], load_fx[i][j], load_fy[i][j]);
        }
        for (int j = 0; j < clamp_attrs[i].Size(); j++)
        {
            elast[i]->AddDispBC(clamp_attrs[i][j], -1, real_t(0));
        }
        elast[i]->SetMaterial(E_simp, nu_cf);
        elast[i]->SetLinearSolver(1e-10, 1e-14, 10000);
    }

    ParGridFunction u(&state_fes);
    StrainEnergyDensityCoefficient energy_cf(&lambda_cf, &mu_cf, &u);
    ProductCoefficient prod(energy_cf, simp_grad_cf);   // r'(rho~) * psi0
    ProductCoefficient dcdrho_cf(-1.0, prod);           // dc/drho~ = -r'(rho~) * psi0
    Compliance comp(MPI_COMM_WORLD, &filter_fes, simp_cf, energy_cf);

    // 7b. Min length scale filter solver (diffusion-mass PDE filter).
    PDEFilter filter(control_fes, filter_fes);
    filter.SetFilterRadius(r_f);
    DiffusionMassSolver &filter_solver = filter.GetSolver();
    for (int a = 1; a <= design_domain.bdr_attributes.Max(); a++)
    {
        bool is_load = false;
        for (int i = 0; i < n_elast_solve; i++)
        {
            for (int j = 0; j < load_attrs[i].Size(); j++)
            {
                if (load_attrs[i][j] == a) { is_load = true; }
            }
        }
        if (!is_load) { filter_solver.AddBoundaryID(a); }
    }
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
    // stacked design  x = [ rho ; alpha_0 ; alpha_1 ; ... ; alpha_{nrays-1} ]
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

    rho.GetTrueDofs(rho_tv);
    for (int r = 0; r < n_dir; r++) { alpha[r]->GetTrueDofs(alpha_tv[r]); }

    BlockVector tx_local(toffsets);
    tx_local.GetBlock(0) = rho_tv;
    for (int r = 0; r < n_dir; r++) { tx_local.GetBlock(1 + r) = alpha_tv[r]; }
    mfem_mma::MMAOptimizerParallel mma(MPI_COMM_WORLD, toffsets.Last(), num_con, tx_local);

    BlockVector tx_min(toffsets), tx_max(toffsets);
    BlockVector df0dx(toffsets);                     // objective gradient  df0/dx = [ dc/drho ; 0 ; ... ; 0 ]
    BlockVector dvol(toffsets);                       // volume constraint gradient is constant
    dvol.GetBlock(0) = *vol_w;  dvol.GetBlock(0) /= Vstar;
    for (int r = 0; r < n_dir; r++) { dvol.GetBlock(1 + r) = 0.0; }

    // one full-size gradient BlockVector per ray-thickness constraint; only
    // block(0) (drho) and block(1+r) (dalpha_r) are ever nonzero.
    vector<BlockVector> dthick(n_dir, BlockVector(toffsets));

    vector<Vector> dfidx(num_con);
    dfidx[0] = dvol;                                  // dfidx[1..n_dir] set each iteration

    // 10. Visualizations
    // 10a. GLVis
    char vishost[] = "localhost";  int visport = 19916;
    socketstream sout;
    if (visualization) { sout.open(vishost, visport);  sout.precision(8); }

    // 10b. Paraview
    ParGridFunction phys_density(&filter_fes);
    std::ostringstream run_tag;
    run_tag << "ct_amax" << alpha_max << "_vf" << vol_fraction;
    ParaViewDataCollection paraview_dc(run_tag.str(), &pmesh);

    if (paraview) {
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.RegisterField("density", &phys_density);
        paraview_dc.RegisterField("rho_filter", &rho_filter);
    }

    // 10c. CSV convergence log (rank 0 only).
    std::ofstream csv;
    if (myid == 0)
    {
        csv.open("convergence.csv");
        csv << "it,compliance,vol,res_max,eps,iterErr\n";
    }

    // 11. Optimization loop.
    int k = 0;
    real_t init_comp = 1.0;
    real_t iterationError = 1.0;
    for (; k < max_it && iterationError > tol; k++)
    {
        if (k % decay_int == 0 && k > 0)
        {
            epsilon = std::max(epsilon * decay, eps_floor);
        }

        // (1) forward filter:  (r_f^2 K + M) ρ~ = M_fc ρ
        rho.GetTrueDofs(rho_tv);
        Vector rho_filter_tv(filter_fes.GetTrueVSize());
        filter.Mult(rho_tv, rho_filter_tv);
        rho_filter.SetFromTrueDofs(rho_filter_tv);

        // (2) state solve:  K(ρ~) u = f   (self-adjoint compliance)

        real_t compliance = 0.0;
        Vector adj_rhs_tv(filter_fes.GetTrueVSize());
        adj_rhs_tv = 0.0;
        for (int i = 0; i < n_elast_solve; i++)
        {
            elast[i]->Assemble();
            elast[i]->FSolve();
            u = elast[i]->GetDisplacements();
            compliance += comp.Eval();

            ParLinearForm adj_rhs(&filter_fes);
            adj_rhs.AddDomainIntegrator(new DomainLFIntegrator(dcdrho_cf));
            adj_rhs.Assemble();
            std::unique_ptr<HypreParVector> adj_rhs_e_tv(adj_rhs.ParallelAssemble());
            adj_rhs_tv += *adj_rhs_e_tv;
        }
        compliance /= n_elast_solve;
        adj_rhs_tv /= n_elast_solve;
        
        // (3) adjoint filter + objective gradient:
        //     w~  = (r_f^2 K + M)^{-1} ∫ (-r'(ρ~) psi_0) φ_i
        //     dc/drho = M_fc^T w~
        filter.MultTranspose(adj_rhs_tv, dcdrho);

        // (4) objective gradient:  df0/dx = [ dc/drho ; 0 ; ... ; 0 ]
        df0dx.GetBlock(0) = dcdrho;
        for (int r = 0; r < n_dir; r++) { df0dx.GetBlock(1 + r) = 0.0; }

        // (5) thickness constraint evaluation and gradient, one per ray
        real_t res_max = -infinity();
        for (int r = 0; r < n_dir; r++)
        {
            advect[r]->FSolve();
            const real_t thickness_res = adv_res[r]->Eval();

            dthick[r] = 0.0;

            Vector dGdrhoa;
            adv_res[r]->GetGrad(dGdrhoa, dthick[r].GetBlock(1 + r));

            // transfer dGdrhoa back to the full-domain dgfes
            ParGridFunction g_sub(sub_dg_fes[r].get());  g_sub.SetFromTrueDofs(dGdrhoa);
            ParGridFunction g_full(&dgfes);               g_full = 0.0;
            outflow[r]->Transfer(g_sub, g_full);
            Vector rhs_full;  g_full.GetTrueDofs(rhs_full);

            // chain rule adjoint solve: dG/drho = M_fc^T N^T g
            advect[r]->SetAdjointRHS(rhs_full);
            advect[r]->ASolve();
            filter.MultTranspose(advect[r]->GetFilterSensitivity(), dthick[r].GetBlock(0));

            dfidx[1 + r] = dthick[r];
            fival(1 + r) = thickness_res - epsilon;
            res_max = std::max(res_max, fival(1 + r));
        }

        // (6) MMA update (rho_tv already set in step (1))
        for (int r = 0; r < n_dir; r++) { alpha[r]->GetTrueDofs(alpha_tv[r]); }

        // volume constraint:  (1,rho)/Vstar - 1 <= 0
        real_t vol = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / domain_volume;
        fival(0) = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / Vstar - 1.0;

        // box constraints:  rho ∈ [0,1],  α_i ∈ [alpha_min, alpha_max]  (move limits)
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

        // stacked update  x = [ ρ ; α_0 ; ... ; α_{nrays-1} ]
        tx_local.GetBlock(0) = rho_tv;
        for (int r = 0; r < n_dir; r++) { tx_local.GetBlock(1 + r) = alpha_tv[r]; }
        rho_old = rho_tv;

        // normalize compliance and its gradient
        if(k == 0) init_comp = compliance;
        compliance /= init_comp;
        df0dx /= init_comp;

        mma.Update(tx_local, df0dx, compliance, fival, dfidx.data(), tx_min, tx_max);
        rho.SetFromTrueDofs(tx_local.GetBlock(0));
        for (int r = 0; r < n_dir; r++) { alpha[r]->SetFromTrueDofs(tx_local.GetBlock(1 + r)); }

        // measure iteration error
        ParGridFunction rho_old_gf(&control_fes);
        rho_old_gf.SetFromTrueDofs(rho_old);
        iterationError = rho_old_gf.ComputeL1Error(rho_cf);

        if (myid == 0)
        {
            mfem::out << "it " << setw(3) << k + 1
                    << "   c = " << scientific << setprecision(6) << compliance
                    << "   vol = " << fixed << setprecision(4) << vol
                    << "   res_max = " << scientific << setprecision(3) << res_max
                    << "   eps = " << fixed << setprecision(4) << epsilon
                    << "   iterErr = " << setprecision(4) << iterationError << endl;

            csv << k + 1 << ','
                << scientific << setprecision(8) << compliance << ','
                << vol << ','
                << res_max << ','
                << epsilon << ','
                << iterationError << '\n';
            csv.flush();
        }

        // physical density r(rho~) for both GLVis and the ParaView archive
        phys_density.ProjectCoefficient(simp_cf);

        if (visualization)
        {
            sout << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n" << design_domain << phys_density
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

    return 0;
}

void loadMesh(int myid, const char *mesh_file,
                Mesh &mesh,
                Array<int> &domain_attr,
                vector<Array<int>> &clamp_attrs,
                vector<Array<int>> &load_attrs,
                vector<Array<int>> &load_fx,
                vector<Array<int>> &load_fy,
                int &n_elast_solve,
                int &outer_bdr_attr)
{
    if (strcmp(mesh_file, "../../data/a_circular_5_holes.msh") == 0)
    {

    }
    else if (strcmp(mesh_file, "../../data/b_circular_9_holes.msh") == 0)
    {

    }
    else if(strcmp(mesh_file, "../../data/d_square_4_holes.msh") == 0)
    {
        n_elast_solve = 2;
        outer_bdr_attr = 1;
        mesh = Mesh(mesh_file);
        // mesh = Mesh::MakeCartesian2D(20, 20, Element::QUADRILATERAL, true, 1.0, 1.0);

        domain_attr.Append(1);

        {
            clamp_attrs.resize(n_elast_solve);
            load_attrs.resize(n_elast_solve);
            load_fx.resize(n_elast_solve);
            load_fy.resize(n_elast_solve);

            // first elast solve
            clamp_attrs[0] = Array<int>({ 6, 7});
            load_attrs[0]  = Array<int>({ 5, 8});
            load_fx[0]     = Array<int>({ 1,-1});
            load_fy[0]     = Array<int>({-1, 1});

            // second elast solve
            clamp_attrs[1] = Array<int>({ 5, 8});
            load_attrs[1]  = Array<int>({ 6, 7});
            load_fx[1]     = Array<int>({-1, 1});
            load_fy[1]     = Array<int>({-1, 1});
        }
    }
    else
        if(myid == 0) mfem::out << "invalid mesh files" << endl;
}