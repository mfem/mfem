// Parallel compliance-minimization topology optimization for linear elasticity.
//
// Sample run:  mpirun -np 4 ./ElasticityTopOpt -r 5 -o 2 -e 0.01 -vf 0.5 -mi 80

#include "mfem.hpp"
#include "ElastTopOpt.hpp"
#include "../MMA_MFEM.hpp"
#include <memory>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 1. Options.
    int    ref_levels  = 5;
    int    order       = 2;
    real_t epsilon     = 0.01;       // filter length
    real_t vol_fraction = 0.5;
    int    max_it      = 100;
    real_t tol         = 1e-2;       // stopping tol on max design change
    real_t move        = 0.2;        // MMA move limit
    bool   visualization = true;
    bool   paraview      = false;

    const real_t rho_min = 1e-6;     // SIMP void stiffness
    const real_t penal = 3.0;        // SIMP exponent
    const real_t Lx = 3.0;           // cantilever length (height 1)
    const real_t load_half = 0.05;   // half-height of the traction patch

    OptionsParser args(argc, argv);
    args.AddOption(&ref_levels, "-r", "--refine", "uniform refinement levels");
    args.AddOption(&order, "-o", "--order", "finite element order");
    args.AddOption(&epsilon, "-e", "--epsilon", "filter length scale");
    args.AddOption(&vol_fraction, "-vf", "--volume-fraction", "volume fraction");
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

    // 2. Label boundary attributes.
    Mesh mesh = Mesh::MakeCartesian2D(3, 1, Element::QUADRILATERAL, true, Lx, 1.0);
    int dim = mesh.Dimension();

    for (int i = 0; i < mesh.GetNBE(); i++)
    {
        Element *be = mesh.GetBdrElement(i);
        Array<int> v;  be->GetVertices(v);
        real_t *c1 = mesh.GetVertex(v[0]);
        real_t *c2 = mesh.GetVertex(v[1]);
        real_t cx = 0.5 * (c1[0] + c2[0]);
        real_t cy = 0.5 * (c1[1] + c2[1]);
        int attr = 2;                                // free
        if (std::abs(cx) < 1e-10) { 
            attr = 1; 
        }                  // clamp (x = 0)
        else if (cx > Lx - 1e-10 &&  std::abs(cy - 0.5) <= load_half) { 
            attr = 3; 
        }                                            // load patch
        be->SetAttribute(attr);
    }
    mesh.SetAttributes();

    // 3. Refined the mesh and construct pmesh
    for (int l = 0; l < ref_levels; l++) 
    { 
        mesh.UniformRefinement(); 
    }

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    // 4. Define finite element collections and spaces
    H1_FECollection state_fec(order, dim);
    H1_FECollection filter_fec(order, dim);
    L2_FECollection control_fec(order - 1, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace state_fes(&pmesh, &state_fec, dim);
    ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);
    ParFiniteElementSpace control_fes(&pmesh, &control_fec);

    // Printing all true dofs.
    HYPRE_BigInt state_size  = state_fes.GlobalTrueVSize();
    HYPRE_BigInt filter_size = filter_fes.GlobalTrueVSize();
    HYPRE_BigInt design_size = control_fes.GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "\nstate dofs = "  << state_size
            << ",  filter dofs = " << filter_size
            << ",  design dofs = " << design_size << endl;
    }

    // 5. Initialized all the grid funcitons and coefficients
    ParGridFunction rho(&control_fes);          
    ParGridFunction rho_filter(&filter_fes);    

    rho = vol_fraction;
    rho_filter = vol_fraction;

    GridFunctionCoefficient rho_cf(&rho);

    // Lame constants and SIMP material coefficients
    ConstantCoefficient lambda_cf(1.0), mu_cf(1.0);
    SIMPCoefficient simp_cf(&rho_filter, rho_min, 1.0, penal);  // r(rho~)
    ProductCoefficient lambda_simp(lambda_cf, simp_cf);         // lambda = r(rho~) lambda0
    ProductCoefficient mu_simp(mu_cf, simp_cf);                 // mu     = r(rho~) mu0

    // 6. Mark essential and load boundaries 
    Array<int> ess_bdr(pmesh.bdr_attributes.Max());
    ess_bdr = 0;  ess_bdr[0] = 1;               // clamp = attribute 1
    Array<int> load_bdr(pmesh.bdr_attributes.Max());
    load_bdr = 0; load_bdr[2] = 1;              // load  = attribute 3
    Vector traction(dim);  traction = 0.0;  traction(dim - 1) = -1.0;
    VectorConstantCoefficient traction_cf(traction);

     // 7. Define elasticity and filter solvers.
    LinearElasticitySolver elast;
    elast.SetMesh(&pmesh);
    elast.SetOrder(order);
    elast.SetLameCoefficients(&lambda_simp, &mu_simp);
    elast.SetRHSbdrCoefficient(&traction_cf);
    elast.SetEssentialBoundary(ess_bdr);
    elast.SetLoadBoundary(load_bdr);
    elast.SetupFEM();
   
    ConstantCoefficient eps2_cf(epsilon * epsilon), one_cf(1.0);
    StrainEnergyDensityCoefficient energy_cf(&lambda_cf, &mu_cf,
                                                elast.GetFEMSolution(), &rho_filter,
                                                rho_min, penal);
    FilterSolver filter;
    filter.SetMesh(&pmesh);
    filter.SetOrder(order);
    filter.SetDiffusionCoefficient(&eps2_cf);
    filter.SetMassCoefficient(&one_cf);
    filter.SetRHSCoefficient(&rho_cf);
    filter.SetAdjointRHSCoefficient(&energy_cf);
    filter.SetupFEM();

    // 8. Volume constraint data:  g(rho) = (1, rho)/Vstar - 1.
    ParLinearForm vol_form(&control_fes);
    vol_form.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
    vol_form.Assemble();
    std::unique_ptr<HypreParVector> vol_w(vol_form.ParallelAssemble());
    real_t loc = vol_w->Sum(), domain_volume;
    MPI_Allreduce(&loc, &domain_volume, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
    const real_t Vstar = vol_fraction * domain_volume;

    // 9. MMA optimizer and its per-iteration work vectors.
    const int n = control_fes.GetTrueVSize();
    Vector rho_tv(n);  rho.GetTrueDofs(rho_tv);
    Vector rho_old(n), dcdrho(n), dgdx(*vol_w), fival(1), xmin(n), xmax(n);

    dgdx /= Vstar;                              // constant volume gradient
    mfem_mma::MMAOptimizerParallel mma(MPI_COMM_WORLD, n, 1, rho_tv);

    // 10. GLVis.
    char vishost[] = "localhost";  int visport = 19916;
    socketstream sout;
    if (visualization) { sout.open(vishost, visport);  sout.precision(8); }

    // 10b. Paraview
    ParGridFunction phys_density(&filter_fes);
    ParaViewDataCollection paraview_dc("ElasticityTopOpt", &pmesh);

    if (paraview) {
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.RegisterField("density", &phys_density);
        paraview_dc.RegisterField("rho_filter", &rho_filter);
    }

    // 11. Optimization loop.
    real_t change = 1.0;
    int k = 0;
    for (; k < max_it && change > tol; k++)
    {
        // (1) forward filter:  (ε^2 K + M) rho~ = N rho
        filter.FSolve();
        rho_filter = *filter.GetFEMSolution();

        // (2) state solve:  K(rho~) u = f   (self-adjoint compliance)
        elast.FSolve();
        real_t compliance = elast.GetCompliance();

        // (3) adjoint filter:  (ε^2 K + M) w~ = -r'(rho~) psi0(u)
        filter.ASolve();
        ParGridFunction &w_filter = *filter.GetFEMSolution();

        // (4) gradient on the design space:  dc/drho = (w~, .)_L2
        GridFunctionCoefficient w_cf(&w_filter);
        ParLinearForm grad_form(&control_fes);
        grad_form.AddDomainIntegrator(new DomainLFIntegrator(w_cf));
        grad_form.Assemble();
        std::unique_ptr<HypreParVector> g(grad_form.ParallelAssemble());
        dcdrho = *g;

        // (5) MMA update (one inequality: volume), with [0,1] + move-limit box
        rho.GetTrueDofs(rho_tv);
        // Volume of the design
        real_t vol = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / domain_volume;
        fival(0) = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / Vstar - 1.0;
        for (int i = 0; i < n; i++)
        {
            xmin[i] = std::max(real_t(0), rho_tv[i] - move);
            xmax[i] = std::min(real_t(1), rho_tv[i] + move);
        }
        rho_old = rho_tv;
        mma.Update(rho_tv, dcdrho, compliance, fival, &dgdx, xmin, xmax);
        rho.SetFromTrueDofs(rho_tv);

        // convergence: global max design change
        change = 0.0;
        for (int i = 0; i < n; i++)
        { change = std::max(change, std::abs(rho_tv[i] - rho_old[i])); }
        MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPITypeMap<real_t>::mpi_type,
                        MPI_MAX, MPI_COMM_WORLD);

        if (myid == 0)
        {
            mfem::out << "it " << setw(3) << k + 1
                    << "   c = " << scientific << setprecision(6) << compliance
                    << "   vol = " << fixed << setprecision(4) << vol
                    << "   change = " << change << endl;
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
        mfem::out << "\nfinished after " << k << " iterations\n";
    }
    return 0;
}
