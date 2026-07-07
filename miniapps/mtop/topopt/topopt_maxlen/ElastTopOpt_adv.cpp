// Linear elasticity topology optimization with a thickness constraint
// Thickness measure is calculated by solving an advection pde
//
// Sample run:  mpirun -np 4 ./ElastTopOpt_adv
//              mpirun -np 4 ./ElastTopOpt_adv -r 5 -amax 0.4 -tol 1e-4 -mi 500

#include "mfem.hpp"
#include "ElastTopOpt.hpp"
#include "qoi.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../pde_filter.hpp"
#include "../../mtop_solvers.hpp"
#include <memory>
#include <fstream>
#include <sstream>

using namespace std;
using namespace mfem;

void bodyload(const Vector &x, Vector &f);
void rayfield(const Vector &x, Vector &v);

int main(int argc, char *argv[])
{
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 1. Options.
    int    dim          = 2;          // problem dimension (2 or 3)
    int    ref_levels   = 5;
    int    order        = 1;
    real_t r_f          = 0.03;       // min filter length
    real_t vol_fraction = 0.5;
    int    max_it       = 300;
    real_t tol          = 1e-3;       // stopping tol on iteration error
    real_t move         = 0.2;        // MMA move limit
    real_t epsilon      = 1e-2;       // thickness residual tolerance
    real_t domain_init  = 0.1;

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
    args.AddOption(&dim, "-dim", "--dimension", "problem dimension (2 or 3)");
    args.AddOption(&ref_levels, "-r", "--refine", "uniform refinement levels");
    args.AddOption(&order, "-o", "--order", "finite element order");
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
    if (myid == 0) { args.PrintOptions(cout); }

    // 2. Build the mesh (3 x 1 box in 2D, 3 x 1 x 1 prism in 3D).
    //    Clamp the x = 0 edge or face in 3D
    Mesh mesh = (dim == 2)
        ? Mesh::MakeCartesian2D(3, 1, Element::QUADRILATERAL, true, 3.0, 1.0)
        : Mesh::MakeCartesian3D(3, 1, 1, Element::HEXAHEDRON, 3.0, 1.0, 1.0);
    const int clamp_attr = (dim == 2) ? 4 : 5;       // x = 0 face

    // 3. Refined the mesh and construct pmesh
    for (int l = 0; l < ref_levels; l++) 
    { 
        mesh.UniformRefinement();
    }

    // 3b. Construct ray field and mark the outflow boundaries
    VectorFunctionCoefficient ray_cf(dim, rayfield);
    {
        for (int i = 0; i < mesh.GetNBE(); i++)
        {
            // skip the clamped boundary
            if (mesh.GetBdrAttribute(i) == clamp_attr) { continue; }

            ElementTransformation *trans = mesh.GetBdrElementTransformation(i);
            const IntegrationPoint &ip = Geometries.GetCenter(
                                            mesh.GetBdrElementGeometry(i));
            trans->SetIntPoint(&ip);

            Vector v(dim);
            ray_cf.Eval(v, *trans, ip);

            Vector normal(dim);
            CalcOrtho(trans->Jacobian(), normal);

            real_t dot = v * normal;

            Vector phys_pt(dim);
            trans->Transform(ip, phys_pt);
            real_t x = phys_pt(0);

            // bool in_x_range = true;
            bool in_x_range = (x >= 1.5 && x <= 2.5);
            bool is_outflow = dot > 0;

            int attr = in_x_range && is_outflow;
            mesh.SetBdrAttribute(i, attr+1);
        }
        mesh.SetAttributes();
    }

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    Array<int> submesh_attr; submesh_attr.Append(2);
    ParSubMesh outflow = ParSubMesh::CreateFromBoundary(pmesh, submesh_attr);

    // 4. Define finite element collections and spaces
    H1_FECollection H1_fec(order, dim);
    L2_FECollection L2_fec(order-1, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace H1_fes(&pmesh, &H1_fec);
    ParFiniteElementSpace L2_fes(&pmesh, &L2_fec);

    // Printing all true dofs.
    HYPRE_BigInt state_size  = H1_fes.GlobalTrueVSize();
    HYPRE_BigInt design_size = L2_fes.GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "\nstate dofs = "  << state_size
            << ",  design dofs = " << design_size << endl;
    }

    // 5. Initialize all the grid functions and coefficients
    ParGridFunction rho(&L2_fes);
    ParGridFunction rho_filter(&H1_fes);
    rho = domain_init;
    rho_filter = domain_init;

    GridFunctionCoefficient rho_cf(&rho);

    // Set up the thickness design variables on the submesh.
    // solving rho_a requires DG, so alpha also lives in DG
    const int sub_dim = outflow.Dimension();

    DG_FECollection dgfec(order, dim, BasisType::GaussLobatto);
    DG_FECollection sub_dg_fec(order, sub_dim, BasisType::GaussLobatto);
    ParFiniteElementSpace dgfes(&pmesh, &dgfec);
    ParFiniteElementSpace sub_dg_fes(&outflow, &sub_dg_fec);

    ParGridFunction rho_a(&dgfes);
    ParGridFunction alpha(&sub_dg_fes);

    rho_a = domain_init;
    alpha = domain_init;  // initialize to mid-range

    // 6. Advection thickness-constraint and its solver
    PseudoTransientSolver advect(rho_a, rho_filter, ray_cf);
    AdvectThicknessResidual adv_res(outflow, advect.GetRhoA(), alpha);

    // advect.SetTimeStep(1 / std::pow(2, ref_levels+1));  // pseudo-transient time step

    // Lame constants and SIMP material coefficients
    ConstantCoefficient one_cf(1.0);
    ConstantCoefficient E_cf(3.0), nu_cf(0.3);
    IsoElasticyLambdaCoeff lambda_cf(&E_cf, &nu_cf);
    IsoElasticySchearCoeff mu_cf(&E_cf, &nu_cf);
    SIMPCoefficient simp_cf(&rho_filter, E_min, E_max, exponent);                // r(rho~)
    SIMPGradCoefficient simp_grad_cf(&rho_filter, E_min, E_max, exponent);       // r'(rho~)

    // 7. Construct the solvers.
    // 7a. Linear elasticity solver.
    VectorFunctionCoefficient force(dim, bodyload);     // body force f
    ProductCoefficient E_simp(simp_cf, E_cf);           // r(rho~) * E0

    IsoLinElasticSolver elast(&pmesh, order);
    elast.SetVolForce(force);
    elast.SetMaterial(E_simp, nu_cf);
    elast.AddDispBC(clamp_attr, -1, 0);
    elast.SetLinearSolver(1e-10, 1e-14, 10000);

    StrainEnergyDensityCoefficient energy_cf(&lambda_cf, &mu_cf,
                                             &elast.GetDisplacements());
    ProductCoefficient prod(energy_cf, simp_grad_cf);   // r'(rho~) * psi0
    ProductCoefficient dcdrho_cf(-1.0, prod);           // dc/drho~ = -r'(rho~) * psi0
    Compliance comp(MPI_COMM_WORLD, &H1_fes, simp_cf, energy_cf);

    // 7b. Min length scale filter solver
    toopt::PDEFilterOptions filter_opts;
    filter_opts.filter_radius = r_f;
    toopt::PDEFilter filter(H1_fes, L2_fes, filter_opts);
    filter.Assemble();

    // 8. Volume constraint data:  g(rho) = (1, rho)/Vstar - 1.
    ParLinearForm vol_form(&L2_fes);
    vol_form.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
    vol_form.Assemble();
    std::unique_ptr<HypreParVector> vol_w(vol_form.ParallelAssemble());

    real_t domain_volume;
    real_t loc = vol_w->Sum();
    MPI_Allreduce(&loc, &domain_volume, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
    const real_t Vstar = vol_fraction * domain_volume;

    // 9. MMA optimizer and its per-iteration work vectors.
    const int n = L2_fes.GetTrueVSize();       // local rho design variables
    const int m = sub_dg_fes.GetTrueVSize();           // local alpha design variables
    Array<int> toffsets(3); toffsets[0] = 0; toffsets[1] = n; toffsets[2] = m; toffsets.PartialSum();

    const int num_con = 2;                          // constraints: volume + thickness
    Vector dcdrho(n), fival(num_con);
    Vector rho_tv(n), rho_old(n);
    Vector alpha_tv(m);

    rho.GetTrueDofs(rho_tv);
    alpha.GetTrueDofs(alpha_tv);

    // stacked design  x = [ rho ; alpha ] 
    BlockVector tx_local(toffsets);
    tx_local.GetBlock(0) = rho_tv;
    tx_local.GetBlock(1) = alpha_tv;
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
    ParGridFunction phys_density(&H1_fes);
    ParaViewDataCollection paraview_dc("ElasticityTopOpt", &pmesh);

    if (paraview) {
        std::ostringstream run_tag;
        run_tag << "ParaView/adv_amax_" << alpha_max;
        paraview_dc.SetPrefixPath(run_tag.str());
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
    real_t iterationError = 1.0;
    for (; k < max_it && iterationError > tol; k++)
    {
        if (k % decay_int == 0 && k > 0)
        {
            epsilon = std::max(epsilon * decay, eps_floor);
        }

        // (1) forward filter:  (r_f^2 K + M) ρ~ = M_fc ρ
        filter.Mult(rho, rho_filter);

        // (2) state solve:  K(ρ~) u = f   (self-adjoint compliance)
        elast.Assemble();
        elast.FSolve();
        elast.GetDisplacements();     // refresh fdisp from sol so energy_cf sees new u

        // (3) adjoint filter + objective gradient:
        //     w~  = (r_f^2 K + M)^{-1} ∫ (-r'(ρ~) psi_0) φ_i
        //     dc/drho = M_fc^T w~
        ParLinearForm adj_rhs(&H1_fes);
        adj_rhs.AddDomainIntegrator(new DomainLFIntegrator(dcdrho_cf));
        adj_rhs.Assemble();
        std::unique_ptr<HypreParVector> adj_rhs_tv(adj_rhs.ParallelAssemble());
        filter.MultTranspose(*adj_rhs_tv, dcdrho);

        // (4) objective gradient:  df0/dx = [ dc/drho ; 0 ]
        df0dx.GetBlock(0) = dcdrho;
        df0dx.GetBlock(1) = 0.0;

        // (5) thickness constraint evaluation and gradient
        advect.FSolve();
        real_t thickness_res = adv_res.Eval();

        Vector dGdrhoa;
        adv_res.GetGrad(dGdrhoa, dthick.GetBlock(1));

        // transfer dGdrhoa back to the parent fes
        ParGridFunction g_sub(&sub_dg_fes);   g_sub.SetFromTrueDofs(dGdrhoa);
        ParGridFunction g_full(&dgfes);       g_full = 0.0;
        outflow.Transfer(g_sub, g_full);
        Vector rhs_full;  g_full.GetTrueDofs(rhs_full);

        // chain rule adjoint solve: dG/drho = M_fc^T N^T g
        advect.SetAdjointRHS(rhs_full);
        advect.ASolve();
        Vector dGdrf;
        advect.GetFilterGrad(dGdrf);
        filter.MultTranspose(dGdrf, dthick.GetBlock(0));
        dfidx[1] = dthick;

        // (6) MMA update
        rho.GetTrueDofs(rho_tv);
        alpha.GetTrueDofs(alpha_tv);

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
            tx_min[n+i] = std::max(alpha_min, alpha_tv[i] - move);
            tx_max[n+i] = std::min(alpha_max, alpha_tv[i] + move);
        }

        // stacked update  x = [ ρ ; α ]
        tx_local.GetBlock(0) = rho_tv;
        tx_local.GetBlock(1) = alpha_tv;
        rho_old = rho_tv;
        real_t compliance = comp.Eval();
        mma.Update(tx_local, df0dx, compliance, fival, dfidx, tx_min, tx_max);
        rho.SetFromTrueDofs(tx_local.GetBlock(0));
        alpha.SetFromTrueDofs(tx_local.GetBlock(1));

        // measure iteration error
        ParGridFunction rho_old_gf(&L2_fes);
        rho_old_gf.SetFromTrueDofs(rho_old);
        iterationError = rho_old_gf.ComputeL1Error(rho_cf);

        if (myid == 0)
        {
            mfem::out << "it " << setw(3) << k + 1
                    << "   c = " << scientific << setprecision(6) << compliance
                    << "   vol = " << fixed << setprecision(4) << vol
                    << "   res_max = " << scientific << setprecision(3) << fival(1)
                    << "   eps = " << fixed << setprecision(4) << epsilon
                    << "   iterErr = " << setprecision(4) << iterationError << endl;

            csv << k + 1 << ','
                << scientific << setprecision(8) << compliance << ','
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

    return 0;
}

void bodyload(const Vector &x, Vector &f)
{
    const int dim = x.Size();
    const real_t xcenter = 2.85;
    const real_t ycenter = 0.5;
    const real_t zcenter = 0.5;
    const real_t radius = 0.1;

    f = 0.0;

    real_t xdiff = x[0] - xcenter;
    real_t ydiff = x[1] - ycenter;
    real_t zdiff = (dim == 3) ? (x[2] - zcenter) : 0.0;
    if (sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff) < radius) 
    { 
        f[dim-1] = -1.0; 
    }
}

void rayfield(const Vector &x, Vector &v)
{
    const int dim = x.Size();
    v.SetSize(dim);
    v = 1.0 / sqrt(dim);
    // v = 0.0; v[dim-1] = 1.0;
}