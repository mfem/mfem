// Linear elasticity topology optimization with a thickness constraint.
// Thickness is measured by sampling ray-cast segments through the design
// with FindPointsGSLIB.
//
// Sample run:  mpirun -np 4 ./ElastTopOpt_maxthick
//              mpirun -np 4 ./ElastTopOpt_maxthick -r 5 -nr 8 -amax 0.4 -mi 300

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
void initialize_rays(int dim, int nrays, Vector *ray_starts, Vector *ray_ends);

int main(int argc, char *argv[])
{
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 1. Options.
    int    dim          = 2;          // problem dimension (2 or 3)
    int    ref_levels   = 1;
    int    order        = 1;
    real_t r_f          = 0.03;       // min filter length
    real_t vol_fraction = 0.5;
    int    max_it       = 100;
    real_t tol          = 1e-3;       // stopping tol on iteration error
    real_t move         = 0.2;        // MMA move limit
    real_t epsilon      = 1e-2;       // thickness residual tolerance
    real_t domain_init  = 0.1;       // initial design value

    // Thickness constraint parameters
    int    nrays        = 5;          // number of thickness measurement rays
    int    nsamples     = 100;        // samples per ray for integration
    real_t alpha_min    = 1e-6;        // minimum thickness bound
    real_t alpha_max    = 0.6;        // maximum thickness bound
    
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
    args.AddOption(&nrays, "-nr", "--nrays", "number of thickness measurement rays");
    args.AddOption(&nsamples, "-ns", "--nsamples", "samples per ray");
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
    //    The Cartesian generators auto-assign per-face boundary attributes, so no
    //    manual marking is needed: the elasticity solver clamps by attribute id.
    //    Native attribute of the x = 0 face:  4 in 2D, 5 in 3D.
    Mesh mesh = (dim == 2)
        ? Mesh::MakeCartesian2D(3, 1, Element::QUADRILATERAL, true, 3.0, 1.0)
        : Mesh::MakeCartesian3D(3, 1, 1, Element::HEXAHEDRON, 3.0, 1.0, 1.0);
    const int clamp_attr = (dim == 2) ? 4 : 5;       // x = 0 face

    // 3. Refined the mesh and construct pmesh
    for (int l = 0; l < ref_levels; l++) 
    { 
        mesh.UniformRefinement(); 
    }

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
    HYPRE_BigInt state_size  = filter_fes.GlobalTrueVSize();
    HYPRE_BigInt design_size = control_fes.GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "\nstate dofs = "  << state_size
            << ",  design dofs = " << design_size << endl;
    }

    // 5. Initialize all the grid functions and coefficients
    ParGridFunction rho(&control_fes);
    ParGridFunction rho_filter(&filter_fes);
    rho = domain_init;
    rho_filter = domain_init;

    GridFunctionCoefficient rho_cf(&rho);

    // 5b. Setup thickness measurement rays (vertical rays across the domain)
    Vector *ray_starts = new Vector[nrays];
    Vector *ray_ends = new Vector[nrays];
    initialize_rays(dim, nrays, ray_starts, ray_ends);

    // Thickness design variables (one per ray)
    Vector alpha(nrays);
    alpha = domain_init;  // initialize to mid-range

    // 6. Thickness constraint residual (ray-sampled):  ∑_i 1/2 (A_i - α_i)²
    ThicknessResidual thickness_residual(pmesh, rho_filter, ray_starts, ray_ends, nrays, alpha, nsamples);

    // Lame constants and SIMP material coefficients
    ConstantCoefficient one_cf(1.0);
    ConstantCoefficient E_cf(3.0), nu_cf(0.3);
    IsoElasticyLambdaCoeff lambda_cf(&E_cf, &nu_cf);
    IsoElasticySchearCoeff mu_cf(&E_cf, &nu_cf);
    SIMPCoefficient simp_cf(&rho_filter, E_min, E_max, exponent);                // r(rho~)
    SIMPGradCoefficient simp_grad_cf(&rho_filter, E_min, E_max, exponent);       // r'(rho~)

    // 7. Construct the solvers.
    // 7a. Linear elasticity solver (clamp the x = 0 face, all components).
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
    Compliance comp(MPI_COMM_WORLD, &filter_fes, simp_cf, energy_cf);

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
    ParaViewDataCollection paraview_dc("ElasticityTopOpt", &pmesh);

    if (paraview) {
        std::ostringstream run_tag;
        run_tag << "ParaView/maxthick_amax_" << alpha_max;
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
        ParLinearForm adj_rhs(&filter_fes);
        adj_rhs.AddDomainIntegrator(new DomainLFIntegrator(dcdrho_cf));
        adj_rhs.Assemble();
        std::unique_ptr<HypreParVector> adj_rhs_tv(adj_rhs.ParallelAssemble());
        filter.MultTranspose(*adj_rhs_tv, dcdrho);

        // (4) objective gradient:  df0/dx = [ dc/drho ; 0 ]
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
        real_t compliance = comp.Eval();
        mma.Update(tx_local, df0dx, compliance, fival, dfidx, tx_min, tx_max);
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

    // Cleanup
    delete[] ray_starts;
    delete[] ray_ends;

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

    // Localized load region (disk in 2D, sphere in 3D); force in the last
    // component: -y in 2D, -z in 3D.
    real_t xdiff = x[0] - xcenter;
    real_t ydiff = x[1] - ycenter;
    real_t zdiff = (dim == 3) ? (x[2] - zcenter) : 0.0;
    if (sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff) < radius) { f[dim-1] = -1.0; }
}

void initialize_rays(int dim, int nrays, Vector *ray_starts, Vector *ray_ends)
{
    for (int r = 0; r < nrays; r++)
    {
        ray_starts[r].SetSize(dim);
        ray_ends[r].SetSize(dim);

        // x_0 ∈ [0.5, 1.5]
        real_t x_pos = 3.0 * r / (nrays - 1);
        // real_t x_pos = 0.5 + 1.0 * r / (nrays - 1);

        ray_starts[r](0) = x_pos;
        ray_ends[r](0)   = x_pos;  
        // ray_ends[r](0)   = x_pos + 1.0;  
        
        ray_starts[r](1) = 0.0;
        ray_ends[r](1)   = 1.0;
        
        if (dim == 3) { 
            ray_starts[r](2) = 0.0;  
            ray_ends[r](2)   = 1.0; 
        }
    }
}