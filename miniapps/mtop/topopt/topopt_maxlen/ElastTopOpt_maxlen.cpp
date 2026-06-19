// Parallel compliance-minimization topology optimization for linear elasticity.
//
// Sample run:  mpirun -np 4 ./ElastTopOpt_maxlen
//              mpirun -np 4 ./ElastTopOpt_maxlen -gv 0.2 -gs 0.8 -mi 300

#include "mfem.hpp"
#include "ElastTopOpt.hpp"
#include "../MMA_MFEM.hpp"
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
    real_t r_f          = 0.01;       // min filter length
    real_t r_s          = 0.1;        // max filter length
    real_t gamma_v      = 1e-6;       // max filter lower bound
    real_t gamma_s      = 0.7;        // max filter upper bound
    real_t vol_fraction = 0.5;
    int    max_it       = 100;
    real_t tol          = 1e-3;       // stopping tol on iteration error
    real_t move         = 0.2;        // MMA move limit
    real_t epsilon      = 1e-2;       // gamma - alpha tolerance
    bool visualization = true;
    bool paraview      = false;

    const real_t E_min    = 1e-6;     // SIMP void stiffness
    const real_t E_max    = 1.0;      // SIMP E max
    const real_t exponent = 3.0;      // SIMP exponent

    OptionsParser args(argc, argv);
    args.AddOption(&ref_levels, "-r", "--refine", "uniform refinement levels");
    args.AddOption(&order, "-o", "--order", "finite element order");
    args.AddOption(&vol_fraction, "-vf", "--volume-fraction", "volume fraction");
    args.AddOption(&r_f, "-rf", "--r_fwidth", "min filter width");
    args.AddOption(&r_s, "-rs", "--r_swidth", "max filter width");
    args.AddOption(&gamma_v, "-gv", "--gamma_v", "lower bound for max filtered density");
    args.AddOption(&gamma_s, "-gs", "--gamma_s", "upper bound for max filtered density");
    args.AddOption(&epsilon, "-e", "--epsilon", "alpha tolerance (initial)");
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
    Mesh mesh = Mesh::MakeCartesian2D(3, 1, Element::QUADRILATERAL, true, 3.0, 1.0);
    int dim = mesh.Dimension();

    for (int i = 0; i < mesh.GetNBE(); i++)
    {
        Element *be = mesh.GetBdrElement(i);
        Array<int> v;  be->GetVertices(v);
        real_t *c1 = mesh.GetVertex(v[0]);
        real_t *c2 = mesh.GetVertex(v[1]);
        real_t cx = 0.5 * (c1[0] + c2[0]);
        real_t cy = 0.5 * (c1[1] + c2[1]);
        int attr = 3;                                // free

        // clamp (x = 0)
        if (std::abs(cx) < 1e-10) { 
            attr = 1; 
        }
        else if (abs(cy - 1) < 1e-10 || abs(cy) < 1e-10 ) {
            attr = 2;
        }                  
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
    // min filter varaibles
    ParGridFunction rho(&control_fes);          
    ParGridFunction rho_filter(&filter_fes); 
    rho = vol_fraction;
    rho_filter = vol_fraction;

    // max filter variables
    ParGridFunction alpha(&control_fes);
    ParGridFunction gamma(&filter_fes);
    alpha = vol_fraction;                           // initialize ⍺ design variable
    gamma = vol_fraction;                           // initialize ɣ filtered maxlen

    GridFunctionCoefficient rho_cf(&rho);
    GridFunctionCoefficient alpha_cf(&alpha);

    // Max-length constraint residual:  1/2 ∫(γ−α)²
    MaxFilterResidual max_residual(MPI_COMM_WORLD, gamma, alpha);

    // Lame constants and SIMP material coefficients
    ConstantCoefficient one_cf(1.0);
    ConstantCoefficient E_cf(3.0), nu_cf(0.3);
    IsoElasticyLambdaCoeff lambda_cf(&E_cf, &nu_cf);
    IsoElasticySchearCoeff mu_cf(&E_cf, &nu_cf);
    SIMPCoefficient simp_cf(&rho_filter, E_min, E_max, exponent);                // r(rho~)
    SIMPGradCoefficient simp_grad_cf(&rho_filter, E_min, E_max, exponent);       // r'(rho~)

    // 6. Mark essential boundaries 
    Array<int> ess_bdr(pmesh.bdr_attributes.Max()); 
    ess_bdr = 0;  
    // ess_bdr[0] = 1;              
    // ess_bdr[1] = 1;

    // 7. Construct the solvers.
    // 7a. Linear elasticity solver
    VectorFunctionCoefficient force(dim, bodyload);     // body force f
    ProductCoefficient E_simp(simp_cf, E_cf);           // r(rho~) * E0

    IsoLinElasticSolver elast(&pmesh, order);
    elast.SetVolForce(force);
    elast.SetMaterial(E_simp, nu_cf);
    elast.AddDispBC(1, -1, 0);
    elast.SetLinearSolver(1e-10, 1e-14, 10000);

    StrainEnergyDensityCoefficient energy_cf(&lambda_cf, &mu_cf,
                                             &elast.GetDisplacements());
    ProductCoefficient prod(energy_cf, simp_grad_cf);   // r'(rho~) * psi0
    ProductCoefficient dcdrho_cf(-1.0, prod);           // dc/drho~ = -r'(rho~) * psi0
    Compliance comp(MPI_COMM_WORLD, &filter_fes, simp_cf, energy_cf);

    // 7b. Min length scale filter solver
    ConstantCoefficient minlen2_cf(r_f * r_f);

    FilterSolver filter;
    filter.SetMesh(&pmesh);
    filter.SetOrder(order);
    filter.SetDiffusionCoefficient(&minlen2_cf);
    filter.SetMassCoefficient(&one_cf);
    filter.SetRHSCoefficient(&rho_cf);
    filter.SetAdjointRHSCoefficient(&dcdrho_cf);
    filter.SetupFEM();

    // 7c. Max length scale filter solver
    ConstantCoefficient maxlen2_cf(r_s * r_s);

    FilterSolver maxfilter;
    maxfilter.SetMesh(&pmesh);
    maxfilter.SetOrder(order);
    maxfilter.SetDiffusionCoefficient(&maxlen2_cf);
    maxfilter.SetMassCoefficient(&one_cf);
    maxfilter.SetRHSCoefficient(&rho_cf);
    maxfilter.SetAdjointRHSCoefficient(max_residual.GetResidualCoefficient());  // γ − α
    maxfilter.SetupFEM();

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
    const int n = control_fes.GetTrueVSize();
    const int m = control_fes.GetTrueVSize();
    // Array<int>  offsets(3);  offsets[0] = 0;  offsets[1] = n;  offsets[2] = m;  offsets.PartialSum();
    Array<int> toffsets(3); toffsets[0] = 0; toffsets[1] = n; toffsets[2] = m; toffsets.PartialSum();

    const int num_con = 2;                       // constraints: volume + max-length
    Vector dcdrho(n), fival(num_con);
    Vector rho_tv(n), rho_old(n);    
    Vector alpha_tv(m);  

    rho.GetTrueDofs(rho_tv);
    alpha.GetTrueDofs(alpha_tv);

    // stacked design  x = [ rho ; alpha ]
    BlockVector tx_local(toffsets); tx_local.GetBlock(0) = rho_tv; tx_local.GetBlock(1) = alpha_tv;
    mfem_mma::MMAOptimizerParallel mma(MPI_COMM_WORLD, n+m, 2, tx_local);

    BlockVector tx_min(toffsets), tx_max(toffsets);
    BlockVector df0dx(toffsets);                    // objective gradient  df0/dx = [ dc/drho ; 0 ]   
    BlockVector dgmax(toffsets);                    // max-length constraint gradient  [ dG/drho ; dG/dalpha ]
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
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.RegisterField("density", &phys_density);
        paraview_dc.RegisterField("rho_filter", &rho_filter);
    }

    // 11. Optimization loop.
    int k = 0;
    real_t iterationError = 1.0;
    real_t eps_ceiling  = epsilon;
    real_t eps_floor    = 1e-10;
    int    infeas_count = 0;         // consecutive iterations with maxres > ε

    for (; k < max_it && iterationError > tol; k++)
    {
        // (1) forward filter:  (r_f^2 K + M) ρ~ = N ρ
        filter.FSolve();
        rho_filter = *filter.GetFEMSolution();

        // (2) state solve:  K(ρ~) u = f   (self-adjoint compliance)
        elast.Assemble();
        elast.FSolve();
        elast.GetDisplacements();     // refresh fdisp from sol so energy_cf sees new u

        // (3) adjoint filter:  (r_f^2 K + M) w~ = -r'(ρ~) psi_0(u)
        filter.ASolve();
        ParGridFunction &w_filter = *filter.GetFEMSolution();

        // (4) max filter: (r_s^2 K + M) γ = N ρ
        maxfilter.FSolve();
        gamma = *maxfilter.GetFEMSolution(); 

        // (5) max adjoint filter: (r_s^2 K + M) s~ = (γ − α)
        maxfilter.ASolve();
        ParGridFunction s_filter = *maxfilter.GetFEMSolution();

        // (6) objective gradient:  df0/dx = [ dc/drho ; 0 ],  dc/drho = (w~, ·)_L2
        GridFunctionCoefficient w_cf(&w_filter);
        ParLinearForm grad_form(&control_fes);
        grad_form.AddDomainIntegrator(new DomainLFIntegrator(w_cf));
        grad_form.Assemble();
        std::unique_ptr<HypreParVector> g(grad_form.ParallelAssemble());
        dcdrho = *g;
        df0dx.GetBlock(0) = dcdrho;
        df0dx.GetBlock(1) = 0.0;

        // max-length constraint gradient:  [ dG/drho ; dG/dalpha ]
        max_residual.GetGrad(s_filter, dgmax.GetBlock(0), dgmax.GetBlock(1));
        dfidx[1] = dgmax;

        // (7) MMA update
        rho.GetTrueDofs(rho_tv);
        alpha.GetTrueDofs(alpha_tv);

        // max-length residual at the current design
        real_t maxres = max_residual.Eval();

        // ========================================================================
        // Adaptively adjust ε every 10 iterations.
        // If constraint is enforce in 10 iterations, then relax the eps tolerance.
        // Otherwise decay it by 0.8. We also set a hard decaying rate of 0.9.
        // =====================================================================
        // infeas_count = (maxres > epsilon) ? infeas_count + 1 : 0;

        // if (k % 10 == 0 && k > 0)
        // {
        //     eps_ceiling *= 0.95;                                        // hard decaying
        //     if (infeas_count >= 10)                                     // relaxation
        //         epsilon = std::min(eps_ceiling, 1.05 * epsilon);
        //     else                                                        // tighten
        //         epsilon *= 0.8;
        //     epsilon = std::min(epsilon, eps_ceiling);                   // cap the ceiling
        //     epsilon = std::max(epsilon, eps_floor);                     // numerical floor
        //     infeas_count = 0;
        // }

        // fixed step restriction for eps
        if (k % 20 == 0 && k > 0)
        {
            epsilon = std::max(epsilon * 0.9, eps_floor);
        }

        // constraints:  volume constraint  and  1/2 ∫(γ−α)² − ε ≤ 0
        real_t vol = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / domain_volume;
        fival(0) = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / Vstar - 1.0;
        fival(1) = maxres - epsilon;

        // box constraints:  rho ∈ [0,1],  ɑ ∈ [γ_v, γ_s]  (move limits)
        for (int i = 0; i < n; i++)
        {
            tx_min[i] = std::max(real_t(0), rho_tv[i] - move);
            tx_max[i] = std::min(real_t(1), rho_tv[i] + move);
        }
        for (int i = 0; i < m; i++)
        {
            tx_min[n+i] = std::max(real_t(gamma_v), alpha_tv[i] - move);
            tx_max[n+i] = std::min(real_t(gamma_s), alpha_tv[i] + move);
        }

        // stacked update  x = [ ρ ; ɑ ]
        tx_local.GetBlock(0) = rho_tv;  tx_local.GetBlock(1) = alpha_tv;
        rho_old = rho_tv;
        real_t compliance = comp.Eval();
        mma.Update(tx_local, df0dx, compliance, fival, dfidx, tx_min, tx_max);
        rho.SetFromTrueDofs(tx_local.GetBlock(0));
        alpha.SetFromTrueDofs(tx_local.GetBlock(1));

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

void bodyload(const Vector &x, Vector &f)
{
    const int dim = x.Size();
    const real_t xcenter = 2.85;
    const real_t ycenter = 0.5;
    const real_t radius = 0.05;

    f = 0.0;

    real_t xdiff = x[0] - xcenter;
    real_t ydiff = x[1] - ycenter;

    if (sqrt(xdiff*xdiff + ydiff*ydiff) < radius) { f[dim-1] = -1.0; }
}