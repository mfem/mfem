// Linear elasticity topology optimization: standard SIMP compliance
// minimization with a volume constraint and a minimum length-scale
// (density) filter only.
//
// Sample run:  mpirun -np 4 ./ElastTopOpt_std
//              mpirun -np 4 ./ElastTopOpt_std -r 6 -rf 0.03 -vf 0.5 -mi 300 -no-vis -pv

#include "mfem.hpp"
#include "ElastTopOpt.hpp"
#include "qoi.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../mtop_solvers.hpp"
#include "../../diffusion_mass_solver.hpp"
#include <memory>
#include <fstream>
#include <sstream>

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
    int    dim          = 2;          // problem dimension (2 or 3)
    int    ref_levels   = 5;
    int    order        = 2;
    real_t r_f          = 0.01;       // min filter length
    real_t vol_fraction = 0.5;
    int    max_it       = 300;
    real_t tol          = 1e-3;       // stopping tol on iteration error
    real_t move         = 0.2;        // MMA move limit

    bool visualization = true;
    bool paraview      = false;

    const real_t E_min    = 1e-6;     // SIMP void stiffness
    const real_t E_max    = 1.0;      // SIMP E max
    const real_t exponent = 3.0;      // SIMP exponent

    OptionsParser args(argc, argv);
    args.AddOption(&dim, "-dim", "--dimension", "problem dimension (2 or 3)");
    args.AddOption(&ref_levels, "-r", "--refine", "uniform refinement levels");
    args.AddOption(&order, "-o", "--order", "finite element order");
    args.AddOption(&vol_fraction, "-vf", "--volume-fraction", "volume fraction");
    args.AddOption(&r_f, "-rf", "--r_fwidth", "min filter width");
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

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    // 4. Define finite element collections and spaces
    H1_FECollection filter_fec(order, dim);
    H1_FECollection control_fec(order-1, dim, BasisType::GaussLobatto);

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

    // 5. Initialize all the grid functions and coefficients
    ParGridFunction rho(&control_fes);
    ParGridFunction rho_filter(&filter_fes);
    rho = vol_fraction;
    rho_filter = vol_fraction;

    GridFunctionCoefficient rho_cf(&rho);

    // Lame constants and SIMP material coefficients
    ConstantCoefficient one_cf(1.0);
    ConstantCoefficient E_cf(3.0), nu_cf(0.3);
    IsoElasticyLambdaCoeff lambda_cf(&E_cf, &nu_cf);
    IsoElasticySchearCoeff mu_cf(&E_cf, &nu_cf);
    SIMPCoefficient simp_cf(&rho_filter, E_min, E_max, exponent);                // r(rho~)
    SIMPGradCoefficient simp_grad_cf(&rho_filter, E_min, E_max, exponent);       // r'(rho~)

    // 6. Construct the solvers.
    // 6a. Linear elasticity solver (clamp the x = 0 face, all components).
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

    // 6b. Minimum length scale filter solver
    PDEFilter filter(control_fes, filter_fes);
    filter.SetFilterRadius(r_f);
    DiffusionMassSolver &filter_solver = filter.GetSolver();
    for(int a = 1; a <= pmesh.bdr_attributes.Max(); a++){
        if (a != clamp_attr) filter_solver.AddBoundaryID(a);
    }
    filter.Assemble();

    // 7. Volume constraint data:  g(rho) = (1, rho)/Vstar - 1.
    ParLinearForm vol_form(&control_fes);
    vol_form.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
    vol_form.Assemble();
    std::unique_ptr<HypreParVector> vol_w(vol_form.ParallelAssemble());

    real_t domain_volume;
    real_t loc = vol_w->Sum();
    MPI_Allreduce(&loc, &domain_volume, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
    const real_t Vstar = vol_fraction * domain_volume;

    // 8. MMA optimizer and its per-iteration work vectors.
    const int n = control_fes.GetTrueVSize();

    const int num_con = 1;                          // constraints: volume only
    Vector dcdrho(n), fival(num_con);
    Vector rho_tv(n), rho_old(n);

    rho.GetTrueDofs(rho_tv);

    mfem_mma::MMAOptimizerParallel mma(MPI_COMM_WORLD, n, num_con, rho_tv);

    Vector tx_min(n), tx_max(n);
    Vector df0dx(n);                                // objective gradient  df0/dx = dc/drho
    Vector dvol(n);                                 // volume constraint gradient (constant)
    dvol = *vol_w;
    dvol /= Vstar;

    Vector dfidx[num_con];  dfidx[0] = dvol;

    // 9. Visualizations
    // 9a. GLVis
    char vishost[] = "localhost";  int visport = 19916;
    socketstream sout;
    if (visualization) { sout.open(vishost, visport);  sout.precision(8); }

    // 9b. Paraview
    ParGridFunction phys_density(&filter_fes);
    std::ostringstream run_tag;
    run_tag << "std_rf" << r_f << "_vf" << vol_fraction;
    ParaViewDataCollection paraview_dc(run_tag.str(), &pmesh);

    if (paraview) {
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.RegisterField("density", &phys_density);
        paraview_dc.RegisterField("rho_filter", &rho_filter);
    }

    // 9c. CSV convergence log (rank 0 only).
    std::ofstream csv;
    if (myid == 0)
    {
        csv.open("convergence.csv");
        csv << "it,compliance,vol,iterErr\n";
    }

    // 10. Optimization loop.
    int k = 0;
    real_t iterationError = 1.0;
    real_t init_comp = 1.0;
    for (; k < max_it && iterationError > tol; k++)
    {
        // (1) forward filter:  (r_f^2 K + M) ρ~ = M_fc ρ
        rho.GetTrueDofs(rho_tv);
        Vector rho_filter_tv(filter_fes.GetTrueVSize());
        filter.Mult(rho_tv, rho_filter_tv);
        rho_filter.SetFromTrueDofs(rho_filter_tv);

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

        // (4) objective gradient:  df0/dx = dc/drho  (from step 3)
        df0dx = dcdrho;

        // (5) MMA update
        rho.GetTrueDofs(rho_tv);

        // volume constraint:  (1, rho)/Vstar - 1 ≤ 0
        real_t vol = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / domain_volume;
        fival(0) = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / Vstar - 1.0;

        // box constraints:  rho ∈ [0,1]  (move limits)
        for (int i = 0; i < n; i++)
        {
            tx_min[i] = std::max(real_t(0), rho_tv[i] - move);
            tx_max[i] = std::min(real_t(1), rho_tv[i] + move);
        }

        rho_old = rho_tv;
        real_t compliance = comp.Eval();

        // Normalize compliance and gradient by initial value
        if (k == 0) { init_comp = compliance; }
        compliance /= init_comp;
        df0dx /= init_comp;

        mma.Update(rho_tv, df0dx, compliance, fival, dfidx, tx_min, tx_max);
        rho.SetFromTrueDofs(rho_tv);

        // measure iteration error
        ParGridFunction rho_old_gf(&control_fes);
        rho_old_gf.SetFromTrueDofs(rho_old);
        iterationError = rho_old_gf.ComputeL1Error(rho_cf);

        if (myid == 0)
        {
            mfem::out << "it " << setw(3) << k + 1
                    << "   c = " << scientific << setprecision(6) << compliance
                    << "   vol = " << fixed << setprecision(4) << vol
                    << "   iterErr = " << setprecision(4) << iterationError << endl;

            csv << k + 1 << ','
                << scientific << setprecision(8) << compliance << ','
                << vol << ','
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
    }

    if (myid == 0)
    {
        csv.close();
        mfem::out << "\nfinished after " << k << " iterations\n";
    }

    // Option: save only the final solution instead of all iterations
    if (paraview)
    {
        paraview_dc.SetCycle(k);
        paraview_dc.SetTime(k);
        paraview_dc.Save();
    }

    return 0;
}

void bodyload(const Vector &x, Vector &f)
{
    const int dim = x.Size();
    const real_t xcenter = 2.85;
    const real_t ycenter = 0.5;
    const real_t zcenter = 0.5;
    const real_t radius = 0.05;

    f = 0.0;

    // Localized load region (disk in 2D, sphere in 3D); force in the last
    // component: -y in 2D, -z in 3D.
    real_t xdiff = x[0] - xcenter;
    real_t ydiff = x[1] - ycenter;
    real_t zdiff = (dim == 3) ? (x[2] - zcenter) : 0.0;
    if (sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff) < radius) { f[dim-1] = -1.0; }
}
