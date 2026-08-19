// Linear elasticity topology optimization with a maximum length scale imposed
// through a p-norm aggregate of the max-filtered density gamma:
//
//   min_rho  c(rho)   s.t.   V_d/V* - 1 <= 0   and   P(gamma)/gamma_s - 1 <= 0
//
//   P(gamma) = ( int_Omega gamma^p dOmega / |Omega| )^(1/p)
//
// P is the power mean of gamma, so it approaches max(gamma) from below as
// p -> infinity and equals gamma exactly for a uniform field.
//
// Sample run:  mpirun -np 4 ./ElastTopOpt_pnorm
//              mpirun -np 4 ./ElastTopOpt_pnorm -r 6 -rf 0.05 -rs 0.2 -gs 0.7 -p 8

#include "mfem.hpp"
#include "ElastTopOpt.hpp"
#include "qoi.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../mtop_solvers.hpp"
#include "../../linear_elasticity_solver.hpp"
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
    real_t r_f          = 0.03;       // min filter length
    real_t r_s          = 0.2;        // max filter length
    real_t beta         = 2.0;        // Heaviside beta
    real_t eta          = 0.2;        // Heaviside eta
    real_t vol_fraction = 0.5;
    int    max_it       = 300;
    real_t tol          = 1e-3;       // stopping tol on iteration error
    real_t move         = 0.2;        // MMA move limit
    int    p            = 8;          // p-norm exponent for the max constraint
    real_t gamma_s      = 0.7;        // max filter upper bound

    int  pc_type     = 2;       // 0 = Jacobi, 1 = LOR diagonal AMG, 2 = LOR monolithic AMG
    bool lor_by_vdim = true;    // monolithic LOR (-pc 2) ordering: byVDIM vs byNODES

    bool visualization = true;
    bool paraview      = false;

    const real_t E_min    = 1e-6;     // SIMP void stiffness
    const real_t E_max    = 1.0;      // SIMP E max
    const real_t exponent = 1.0;      // SIMP exponent (applied to the projection)

    int    init_it    = 20;           // initial iterations before continuation
    int    beta_steps = 50;           // Heaviside beta continuation steps
    real_t beta_max   = 2.0;          // Heaviside beta max value

    OptionsParser args(argc, argv);
    args.AddOption(&dim, "-dim", "--dimension", "problem dimension (2 or 3)");
    args.AddOption(&ref_levels, "-r", "--refine", "uniform refinement levels");
    args.AddOption(&order, "-o", "--order", "finite element order");
    args.AddOption(&vol_fraction, "-vf", "--volume-fraction", "volume fraction");
    args.AddOption(&r_f, "-rf", "--r_fwidth", "min filter width");
    args.AddOption(&r_s, "-rs", "--r_swidth", "max filter width");
    args.AddOption(&beta, "-b", "--beta", "Heaviside beta");
    args.AddOption(&beta_steps, "-bs", "--beta-steps", "Heaviside beta continuation steps");
    args.AddOption(&beta_max, "-bm", "--beta-max", "Heaviside beta max value");
    args.AddOption(&eta, "-e", "--eta", "Heaviside eta");
    args.AddOption(&init_it, "-ii", "--init_it", "initial iterations before continuation");
    args.AddOption(&max_it, "-mi", "--max-it", "max optimization iterations");
    args.AddOption(&tol, "-tol", "--tol", "stopping tol on max design change");
    args.AddOption(&move, "-mv", "--move", "MMA move limit");
    args.AddOption(&p, "-p", "--pnorm", "p-norm exponent for the max constraint");
    args.AddOption(&gamma_s, "-gs", "--gamma_s", "upper bound for max filtered density");
    args.AddOption(&pc_type, "-pc", "--elast-precond", "elasticity preconditioner: "
                    "0 = Jacobi, 1 = LOR diagonal AMG,  2 = LOR monolithic AMG");
    args.AddOption(&lor_by_vdim, "-vdim", "--by-vdim", "-nodes", "--by-nodes",
                    "monolithic LOR ordering: byVDIM / byNODES");
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
    H1_FECollection state_fec(order, dim);
    H1_FECollection filter_fec(order, dim);
    H1_FECollection control_fec(order-1, dim, BasisType::GaussLobatto);

    ParFiniteElementSpace state_fes(&pmesh, &state_fec, dim, Ordering::byNODES);
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
    ParGridFunction gamma(&filter_fes);
    rho = vol_fraction;
    rho_filter = vol_fraction;
    gamma = vol_fraction;

    GridFunctionCoefficient rho_cf(&rho);

    GridFunctionCoefficient gamma_cf(&gamma);
    PowerCoefficient gamma_p_cf(gamma_cf, p);         // γ^p
    PowerCoefficient gamma_pm1_cf(gamma_cf, p - 1);   // γ^(p-1)

    // Lame constants and SIMP material coefficients
    ConstantCoefficient E_cf(3.0), nu_cf(0.3);
    IsoElasticyLambdaCoeff lambda_cf(&E_cf, &nu_cf);
    IsoElasticySchearCoeff mu_cf(&E_cf, &nu_cf);

    HeavisideCoefficient rho_erod_cf(&rho_filter, beta, 1-eta);
    HeavisideGradCoefficient rho_erod_grad_cf(&rho_filter, beta, 1-eta);

    HeavisideCoefficient rho_dila_cf(&rho_filter, beta, eta);
    HeavisideGradCoefficient rho_dila_grad_cf(&rho_filter, beta, eta);

    HeavisideCoefficient rho_inter_cf(&rho_filter, beta, 0.5);

    // SIMP on the eroded projection: r(rho_e) = E_min + rho_e^exponent (E_max - E_min)
    SIMPCoefficient simp_cf(rho_erod_cf, E_min, E_max, exponent);            // r(rho_e)
    SIMPGradCoefficient simp_grad_cf(rho_erod_cf, E_min, E_max, exponent);   // r'(rho_e)

    // 6. Construct the solvers.
    // 6a. Linear elasticity solver (clamp the x = 0 face, all components).
    VectorFunctionCoefficient force(dim, bodyload);          // body force f
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

    LinearElasticitySolver elast(state_fes);
    for (int a = 0; a < pmesh.attributes.Size(); a++)
    {
        elast.AddVolumeLoad(pmesh.attributes[a], force);
    }
    elast.SetLambda(lambda_simp_cf);
    elast.SetMu(mu_simp_cf);
    elast.AddBoundaryID(clamp_attr);
    elast.SetPreconditionerType(elast_pc);
    elast.SetMonolithicLOROrdering(
        lor_by_vdim ? Ordering::byVDIM : Ordering::byNODES);
    elast.SetPrintLevel(0);
    elast.SetRelTol(1e-10);
    elast.SetAbsTol(1e-14);
    elast.SetMaxIter(2000);

    ParGridFunction u(&state_fes);
    StrainEnergyDensityCoefficient energy_cf(&lambda_cf, &mu_cf, &u);

    // dc/drho~ = - (r'(rho_e) * H_e'(rho~)) * psi0(u)
    ProductCoefficient drdrho_cf(simp_grad_cf, rho_erod_grad_cf);
    ProductCoefficient prod(energy_cf, drdrho_cf);
    ProductCoefficient dcdrho_tilde_cf(-1.0, prod);

    // 6b. Minimum length scale filter solver
    PDEFilter filter(control_fes, filter_fes);
    filter.SetFilterRadius(r_f);
    DiffusionMassSolver &filter_solver = filter.GetSolver();
    for(int a = 1; a <= pmesh.bdr_attributes.Max(); a++){
        if (a != clamp_attr) filter_solver.AddBoundaryID(a);
    }
    filter.Assemble();

    // 6c. Max length scale filter solver
    PDEFilter maxfilter(control_fes, filter_fes);
    maxfilter.SetFilterRadius(r_s);
    DiffusionMassSolver &maxfilter_solver = maxfilter.GetSolver();
    for(int a = 1; a <= pmesh.bdr_attributes.Max(); a++){
        if (a != clamp_attr) maxfilter_solver.AddBoundaryID(a);
    }
    maxfilter.Assemble();

    // 7. Construct the quantity of interest objects
    Compliance comp(MPI_COMM_WORLD, &filter_fes, simp_cf, energy_cf);

    // volume of the dilated field, measured against V* = vol_fraction |Omega|
    VolumeResidual volume(MPI_COMM_WORLD, &filter_fes, &rho_dila_cf,
                           &rho_dila_grad_cf, vol_fraction);

    // |Omega|, so the p-norm is a power mean on the same scale as gamma and
    // gamma_s is directly a density bound, independent of domain size and p
    real_t domain_volume = 0.0;
    {
        ConstantCoefficient one_cf(1.0);
        ParLinearForm vol_lf(&filter_fes);
        vol_lf.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
        vol_lf.Assemble();
        std::unique_ptr<HypreParVector> vol_tv(vol_lf.ParallelAssemble());

        real_t loc = vol_tv->Sum();
        MPI_Allreduce(&loc, &domain_volume, 1, MPITypeMap<real_t>::mpi_type,
                      MPI_SUM, MPI_COMM_WORLD);
    }

    // 8. MMA optimizer and its per-iteration work vectors.
    const int n = control_fes.GetTrueVSize();
    const int num_con = 2;                          // constraints: volume, max-length p-norm

    Vector rho_tv(n), rho_old(n);
    rho.GetTrueDofs(rho_tv);

    Vector a(num_con), c(num_con), d(num_con);
    a = 0.0; c = 1000.0; d = 0.0;
    mfem_mma::MMAOptimizerParallel mma(MPI_COMM_WORLD, n, num_con, rho_tv, a, c, d);
    
    Vector fival(num_con);
    Vector df0dx(n);                                 // objective gradient  df0/dx
    Vector dfidx[num_con];                           // local gradient of constraint  dg/dx
    for (int i = 0; i < num_con; i++) { dfidx[i].SetSize(n); }
    Vector tx_min(n), tx_max(n);

    Vector dcdrho(n);                                // compliance gradient  dc/drho
    Vector dvol_tilde(filter_fes.GetTrueVSize());    // dV/drho~

    // 9. Visualizations
    // 9a. GLVis
    char vishost[] = "localhost";  int visport = 19916;
    socketstream sout;
    if (visualization) {
        sout.open(vishost, visport);
        sout.precision(8);

        sout << "parallel " << num_procs << " " << myid << "\n"
            << "solution\n" << pmesh << rho_filter
            << "window_title 'Design density r(rho~)'\n"
            << "window_geometry 0 0 800 600\n"
            << "colorbar_numberformat '%.2f'\n"
            << "keys Rjlc*****\n" << flush;
    }

    // 9b. Paraview
    ParGridFunction phys_density(&filter_fes);
    std::ostringstream run_tag;
    run_tag << "pnorm_rs" << r_s << "_gs" << gamma_s << "_p" << p
            << "_vf" << vol_fraction;
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
        csv << "it,compliance,volume,pnorm,fival,iterErr\n";
    }

    // 10. Optimization loop.
    real_t iterationError = 1.0;
    real_t init_comp = 1.0;

    // Track next iteration for beta doubling
    int next_beta_double = init_it + beta_steps;

    int it = 1;
    for (; (it <= init_it) || (it <= max_it && iterationError > tol); it++)
    {
        // (1) forward filter:  (r_f^2 K + M) ρ~ = M_fc ρ
        rho.GetTrueDofs(rho_tv);
        Vector rho_filter_tv(filter_fes.GetTrueVSize());
        filter.Mult(rho_tv, rho_filter_tv);
        rho_filter.SetFromTrueDofs(rho_filter_tv);

        // (2) state solve:  K(ρ~) u = f   (self-adjoint compliance)
        elast.SetNeedsAssembly();   // reset assembly flag
        elast.Solve(u);

        // evaluate compliance 
        real_t compliance = comp.Eval();

        // (3) adjoint filter + objective gradient:
        //     w~  = (r_f^2 K + M)^{-1} ∫ (-r'(ρ~) psi_0) φ_i
        //     dc/drho = M_fc^T w~
        ParLinearForm adj_rhs(&filter_fes);
        adj_rhs.AddDomainIntegrator(new DomainLFIntegrator(dcdrho_tilde_cf));
        adj_rhs.Assemble();
        std::unique_ptr<HypreParVector> adj_rhs_tv(adj_rhs.ParallelAssemble());
        filter.MultTranspose(*adj_rhs_tv, dcdrho);
        df0dx = dcdrho;                                 // objective gradient

        // (4) volume constraint and gradient on the dilated field:
        //       g        = V_d / V* - 1
        //       dg/drho~ = (H_d'(ρ~), φ_i) / V*

        fival(0) = volume.Eval() - 1.0;                 // update constraint value
        volume.GetGrad(dvol_tilde);
        filter.MultTranspose(dvol_tilde, dfidx[0]);     // update constraint gradient

        // (5) max filter: (r_s^2 K + M) γ = M_fc ρ
        Vector gamma_tv(filter_fes.GetTrueVSize());
        maxfilter.Mult(rho_tv, gamma_tv);
        gamma.SetFromTrueDofs(gamma_tv);

        // (6) p-norm constraint and gradient on the max filtered field:
        //       P        = ( ∫ γ^p dΩ / |Ω| )^(1/p)  ->  max γ  as p -> ∞
        //       g        = P / γ_s - 1
        //       dP/dγ    = P^(1-p) (γ^(p-1), φ_i) / |Ω|
        ParLinearForm gp_lf(&filter_fes);
        gp_lf.AddDomainIntegrator(new DomainLFIntegrator(gamma_p_cf));
        gp_lf.Assemble();
        std::unique_ptr<HypreParVector> gp_tv(gp_lf.ParallelAssemble());

        real_t loc_int = gp_tv->Sum(), glob_int;
        MPI_Allreduce(&loc_int, &glob_int, 1, MPITypeMap<real_t>::mpi_type,
                      MPI_SUM, MPI_COMM_WORLD);

        const real_t pnorm_val = pow(glob_int / domain_volume, 1.0 / p);
        fival(1) = pnorm_val / gamma_s - 1.0;           // update constraint value

        ParLinearForm dP_lf(&filter_fes);
        dP_lf.AddDomainIntegrator(new DomainLFIntegrator(gamma_pm1_cf));
        dP_lf.Assemble();
        std::unique_ptr<HypreParVector> dP_tv(dP_lf.ParallelAssemble());
        *dP_tv *= pow(pnorm_val, 1.0 - p) / (domain_volume * gamma_s);

        maxfilter.MultTranspose(*dP_tv, dfidx[1]);      // update constraint gradient

        // (7) box constraints:  rho ∈ [0,1]  (move limits)
        for (int i = 0; i < n; i++)
        {
            tx_min[i] = std::max(real_t(0), rho_tv[i] - move);
            tx_max[i] = std::min(real_t(1), rho_tv[i] + move);
        }

        // (8) MMA update:
        rho_old = rho_tv;

        // Normalize compliance and gradient by initial value
        if (it == 1) { init_comp = compliance; }
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
            const int w = 14;               // column width
            mfem::out << "\niteration " << it << '\n' << left
                    << setw(w) << "c"
                    << setw(w) << "volume"
                    << setw(w) << "pnorm"
                    << setw(w) << "fival"
                    << setw(w) << "beta"
                    << setw(w) << "iterErr" << '\n'
                    << string(6*w, '=') << '\n'
                    << fixed      << setprecision(6) << setw(w) << compliance
                    <<               setprecision(4) << setw(w) << fival(0) + 1.0
                    <<               setprecision(4) << setw(w) << pnorm_val
                    << scientific << setprecision(3) << setw(w) << fival(1)
                    << fixed      << setprecision(0) << setw(w) << beta
                    << scientific << setprecision(4) << setw(w) << iterationError << endl;

            csv << it << ','
                << scientific << setprecision(8) << compliance << ','
                << fival(0) + 1.0 << ','
                << pnorm_val << ','
                << fival(1) << ','
                << iterationError << '\n';
            csv.flush();
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

        // physical density r(rho~) for both GLVis and the ParaView archive
        phys_density.ProjectCoefficient(rho_inter_cf);

        if (visualization)
        {
            sout << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n" << pmesh << phys_density << flush;
        }
    }

    if (myid == 0)
    {
        csv.close();
        mfem::out << "\nfinished after " << (it - 1) << " iterations\n";
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
