// Linear elasticity topology optimization with a max thickness constraint.
// Thickness is measured by sampling ray-cast segments through the design
// with FindPointsGSLIB.
//
// Sample run:  mpirun -np 8 ./ElastTopOpt_maxthick
//              mpirun -np 8 ./ElastTopOpt_maxthick -r 6 -nr 8 -amax 0.3 -rt 2 -mi 500

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
void initialize_rays(int dim, int nrays, int ray_type, Vector *ray_starts, Vector *ray_ends);

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
    real_t alpha_min    = 1e-6;       // thickness variable lower bound
    real_t alpha_max    = 0.5;        // thickness variable upper bound
    real_t beta         = 2.0;        // Heaviside beta
    real_t eta          = 0.2;        // Heaviside eta
    real_t vol_fraction = 0.5;
    int    max_it       = 300;
    real_t tol          = 1e-3;       // stopping tol on iteration error
    real_t move         = 0.1;        // MMA move limit
    real_t epsilon      = 1e-2;       // thickness residual tolerance

    // Ray sampling of the thickness
    int    nrays        = 50;         // number of thickness measurement rays
    int    nsamples     = 100;        // samples per ray for integration
    int    ray_type     = 1;          // 1: vertical, 2: diagonal

    real_t domain_init  = 0.1;        // initial (uniform) design density

    bool visualization = true;
    bool paraview      = false;

    const real_t E_min    = 1e-6;     // SIMP void stiffness
    const real_t E_max    = 1.0;      // SIMP E max
    const real_t exponent = 1.0;      // SIMP exponent (applied to the projection)
    // --- PLAIN SIMP ---  use p = 3 when SIMP acts directly on rho~
    // const real_t exponent = 3.0;

    int init_it  = 20;
    real_t decay     = 0.8;
    real_t eps_floor = 1e-6;
    int decay_int    = 10;

    int beta_steps   = 50;          // Heaviside beta continuation steps
    real_t beta_max  = 2.0;         // Heaviside beta max value

    OptionsParser args(argc, argv);
    args.AddOption(&dim, "-dim", "--dimension", "problem dimension (2 or 3)");
    args.AddOption(&ref_levels, "-r", "--refine", "uniform refinement levels");
    args.AddOption(&order, "-o", "--order", "finite element order");
    args.AddOption(&vol_fraction, "-vf", "--volume-fraction", "volume fraction");
    args.AddOption(&r_f, "-rf", "--r_fwidth", "min filter width");
    args.AddOption(&nrays, "-nr", "--nrays", "number of thickness measurement rays");
    args.AddOption(&nsamples, "-ns", "--nsamples", "samples per ray");
    args.AddOption(&ray_type, "-rt", "--raytype", "ray field type: 1=vertical, 2=diagonal");
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
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0) { args.PrintUsage(cout); }
        return 1;
    }
    if (ray_type != 1 && ray_type != 2)
    {
        if (myid == 0) {
            cerr << "Invalid ray type: " << ray_type
                 << ". Must be 1 (vertical) or 2 (diagonal)." << endl;
        }
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

    // FindPointsGSLIB (used by ThicknessResidual) needs a nodal grid function.
    pmesh.SetCurvature(order);

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
    rho = domain_init;
    rho_filter = domain_init;

    GridFunctionCoefficient rho_cf(&rho);

    // 5b. Thickness measurement rays and their per-ray design variables.
    Vector *ray_starts = new Vector[nrays];
    Vector *ray_ends = new Vector[nrays];
    initialize_rays(dim, nrays, ray_type, ray_starts, ray_ends);

    Vector alpha(nrays);
    alpha = domain_init;

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
    // 6a. Linear elasticity solver (clamp the x = 0 face, all components).
    VectorFunctionCoefficient force(dim, bodyload);     // body force f
    ProductCoefficient E_simp(simp_cf, E_cf);           // r(rho_e) * E0

    IsoLinElasticSolver elast(&pmesh, order);
    elast.SetVolForce(force);
    elast.SetMaterial(E_simp, nu_cf);
    elast.AddDispBC(clamp_attr, -1, 0);
    elast.SetLinearSolver(1e-10, 1e-14, 10000);

    StrainEnergyDensityCoefficient energy_cf(&lambda_cf, &mu_cf,
                                             &elast.GetDisplacements());
    // dc/drho~ = - (r'(rho_e) * H_e'(rho~)) * psi0(u)
    ProductCoefficient drdrho_cf(simp_grad_cf, rho_erod_grad_cf);
    ProductCoefficient prod(energy_cf, drdrho_cf);
    ProductCoefficient dcdrho_tilde_cf(-1.0, prod);

    // --- PLAIN SIMP ---  dc/drho~ = -r'(rho~) * psi0
    // ProductCoefficient prod(energy_cf, simp_grad_cf);
    // ProductCoefficient dcdrho_tilde_cf(-1.0, prod);

    // 6b. Min length scale filter solver
    PDEFilter filter(control_fes, filter_fes);
    filter.SetFilterRadius(r_f);
    DiffusionMassSolver &filter_solver = filter.GetSolver();
    for(int a = 1; a <= pmesh.bdr_attributes.Max(); a++){
        if (a != clamp_attr) filter_solver.AddBoundaryID(a);
    }
    filter.Assemble();

    // 7. Construct the quantity of interest objects
    Compliance comp(MPI_COMM_WORLD, &filter_fes, simp_cf, energy_cf);

    // volume of the dilated field, measured against V* = vol_fraction |Omega|
    VolumeResidual vol_qoi(MPI_COMM_WORLD, &filter_fes, &rho_dila_cf,
                           &rho_dila_grad_cf, vol_fraction);

    // Max-thickness constraint residual:  1/(2 N) ∑_i (A_i − α_i)²
    ThicknessResidual thickness_qoi(pmesh, rho_filter, ray_starts, ray_ends,
                                    nrays, alpha, nsamples);

    // --- PLAIN SIMP ---  linear volume constraint  g(rho) = (1, rho)/Vstar - 1
    // ParLinearForm vol_form(&control_fes);
    // vol_form.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
    // vol_form.Assemble();
    // std::unique_ptr<HypreParVector> vol_w(vol_form.ParallelAssemble());
    //
    // real_t domain_volume;
    // real_t loc = vol_w->Sum();
    // MPI_Allreduce(&loc, &domain_volume, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
    // const real_t Vstar = vol_fraction * domain_volume;

    // 8. MMA optimizer and its per-iteration work vectors.
    const int n  = control_fes.GetTrueVSize();      // local rho design variables
    const int m  = (myid == 0) ? nrays : 0;         // alpha block lives on rank 0 only
    const int nf = filter_fes.GetTrueVSize();
    Array<int> toffsets(3); toffsets[0] = 0; toffsets[1] = n; toffsets[2] = m; toffsets.PartialSum();

    const int num_con = 2;                          // constraints: volume + max-thickness

    // 8a. stacked design  x = [ rho ; alpha ]
    Vector rho_tv(n), rho_old(n);
    rho.GetTrueDofs(rho_tv);

    BlockVector tx_local(toffsets);
    tx_local.GetBlock(0) = rho_tv;
    if (myid == 0) { tx_local.GetBlock(1) = alpha; }

    Vector a(num_con), c(num_con), d(num_con);
    a = 0.0; c = 1000.0; d = 0.0;
    mfem_mma::MMAOptimizerParallel mma(MPI_COMM_WORLD, n+m, num_con, tx_local, a, c, d);

    // 8b. objective initialization
    BlockVector df0dx(toffsets);                    // objective gradient  df0/dx = [ dc/drho ; 0 ]
    Vector dcdrho(n);                               // compliance gradient  dc/drho

    // 8c. local constraints
    Vector fival(num_con);
    Vector dfidx[num_con];

    BlockVector dvol(toffsets);                     // volume gradient     [ dg/drho ; 0 ]
    BlockVector dthick(toffsets);                   // thickness gradient  [ dR/drho ; dR/dalpha ]
    dvol = 0.0; dfidx[0] = dvol;
    Vector dvol_tilde(nf);                          // dV_d/drho~
    Vector dthick_tilde;                            // dR/drho~

    // --- PLAIN SIMP ---  the linear volume gradient is constant:  [ vol_w/Vstar ; 0 ]
    // dvol.GetBlock(0) = *vol_w;
    // dvol.GetBlock(0) /= Vstar;
    // dvol.GetBlock(1) = 0.0;

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
            << "solution\n" << pmesh << rho_filter
            << "window_title 'Design density r(rho~)'\n"
            << "window_geometry 0 0 800 600\n"
            << "colorbar_numberformat '%.2f'\n"
            << "keys Rjlc*****\n" << flush;

        sout_filter.open(vishost, visport);
        sout_filter.precision(8);

        sout_filter << "parallel " << num_procs << " " << myid << "\n"
            << "solution\n" << pmesh << rho_filter
            << "window_title 'Filtered density rho~'\n"
            << "window_geometry 810 0 800 600\n"
            << "colorbar_numberformat '%.2f'\n"
            << "keys Rjlc*****\n" << flush;
    }

    // 9b. Paraview
    ParGridFunction phys_density(&filter_fes);
    std::ostringstream run_tag;
    run_tag << "maxthick_amax" << alpha_max << "_rt" << ray_type << "_vf" << vol_fraction;
    ParaViewDataCollection paraview_dc(run_tag.str(), &pmesh);

    if (paraview) {
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.RegisterField("density", &phys_density);
        paraview_dc.RegisterField("rho_filter", &rho_filter);
    }

    // 9c. CSV convergence log.
    std::ofstream csv;
    if (myid == 0)
    {
        csv.open("convergence.csv");
        csv << "it,compliance,volume,res_thick,eps,iterErr\n";
    }

    // 10. Optimization loop.
    real_t iterationError = 1.0;
    real_t init_comp = 1.0;

    // Track next iteration for epsilon decay and beta doubling
    int next_epsilon_decay = init_it;
    int next_beta_double = init_it + beta_steps;

    int it = 1;
    for (; (it <= init_it) || (it <= max_it && iterationError > tol); it++)
    {
        // (1) forward filter:  (r_f^2 K + M) ρ~ = M_fc ρ
        rho.GetTrueDofs(rho_tv);
        Vector rho_filter_tv(nf);
        filter.Mult(rho_tv, rho_filter_tv);
        rho_filter.SetFromTrueDofs(rho_filter_tv);

        // (2) state solve:  K(ρ~) u = f   (self-adjoint compliance)
        elast.Assemble();
        elast.FSolve();
        elast.GetDisplacements();     // refresh fdisp from sol so energy_cf sees new u

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
        df0dx.GetBlock(0) = dcdrho;                     // objective gradient
        df0dx.GetBlock(1) = 0.0;                        // c does not depend on alpha

        // (4) volume constraint and gradient on the dilated field:
        //       g        = V_d / V* - 1
        //       dg/drho~ = (H_d'(ρ~), φ_i) / V*
        fival(0) = vol_qoi.Eval() - 1.0;                // update constraint value
        real_t vol = (fival(0) + 1.0) * vol_fraction;   // current volume fraction
        vol_qoi.GetGrad(dvol_tilde);
        filter.MultTranspose(dvol_tilde, dvol.GetBlock(0));
        dfidx[0] = dvol;                                // update constraint gradient

        // --- PLAIN SIMP ---  linear volume constraint on rho (gradient is
        //                     constant, set once outside the loop)
        // fival(0) = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / Vstar - 1.0;

        // (5) ray-sampled thickness  A_i = ∫_ray ρ~ ds.  Eval fills the per-ray
        //     residuals that the gradients below reuse, so it must come first.
        real_t thickres = thickness_qoi.Eval();

        // (6) max-thickness constraint and gradient:  1/(2N) ∑(A_i−α_i)² − ε ≤ 0
        //     dR/drho   = M_fc^T (r_f^2 K + M)^{-1} ℓ,  ℓ = Σ_i (A_i−α_i) ds_i Σ_k φ(x_ik)
        //     dR/dalpha = -(A_i − α_i)/N
        thickness_qoi.GetGradRHS(dthick_tilde);
        filter.MultTranspose(dthick_tilde, dthick.GetBlock(0));
        if (myid == 0) { thickness_qoi.GetGrad(dthick.GetBlock(1)); }

        fival(1) = thickres / epsilon - 1.0;             // update constraint value
        dthick /= epsilon;
        dfidx[1] = dthick;                               // update constraint gradient

        // (7) box constraints:  rho ∈ [0,1],  ɑ ∈ [alpha_min, alpha_max]  (move limits)
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

        // (8) MMA update on the stacked design  x = [ ρ ; α ]
        tx_local.GetBlock(0) = rho_tv;
        if (myid == 0) { tx_local.GetBlock(1) = alpha; }
        rho_old = rho_tv;

        // Normalize compliance and gradient by initial value
        if (it == 1) { init_comp = compliance; }
        compliance /= init_comp;
        df0dx /= init_comp;

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
            const int w = 14;               // column width
            mfem::out << "\niteration " << it << '\n' << left
                    << setw(w) << "c"
                    << setw(w) << "volume"
                    << setw(w) << "res_thick"
                    << setw(w) << "eps"
                    << setw(w) << "fival"
                    << setw(w) << "beta"
                    << setw(w) << "iterErr" << '\n'
                    << string(7*w, '=') << '\n'
                    << fixed      << setprecision(6) << setw(w) << compliance
                    <<               setprecision(4) << setw(w) << vol
                    << scientific << setprecision(3) << setw(w) << thickres
                    <<               setprecision(3) << setw(w) << epsilon
                    <<               setprecision(3) << setw(w) << fival(1)
                    << fixed      << setprecision(0) << setw(w) << beta
                    << scientific << setprecision(4) << setw(w) << iterationError << endl;

            csv << it << ','
                << scientific << setprecision(8) << compliance << ','
                << vol << ','
                << thickres << ','
                << epsilon << ','
                << iterationError << '\n';
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
                << "solution\n" << pmesh << phys_density << flush;

            sout_filter << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n" << pmesh << rho_filter << flush;
        }

        // if (paraview)
        // {
        //     paraview_dc.SetCycle(it);
        //     paraview_dc.SetTime(it);
        //     paraview_dc.Save();
        // }
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
    const real_t radius = 0.05;

    f = 0.0;

    // Localized load region (disk in 2D, sphere in 3D)
    // force in the last component: -y in 2D, -z in 3D.
    real_t xdiff = x[0] - xcenter;
    real_t ydiff = x[1] - ycenter;
    real_t zdiff = (dim == 3) ? (x[2] - zcenter) : 0.0;
    if (sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff) < radius) { f[dim-1] = -1.0; }
}

void initialize_rays(int dim, int nrays, int ray_type, Vector *ray_starts, Vector *ray_ends)
{
    real_t x_pos;
    for (int r = 0; r < nrays; r++)
    {
        ray_starts[r].SetSize(dim);
        ray_ends[r].SetSize(dim);

        if (ray_type == 1)  // vertical rays
        {
            x_pos = 3.0 * r / (nrays - 1);

            ray_starts[r](0) = x_pos;
            ray_ends[r](0)   = x_pos;

            ray_starts[r](1) = 0.0;
            ray_ends[r](1)   = 1.0;

            if (dim == 3) {
                ray_starts[r](2) = 0.0;
                ray_ends[r](2)   = 1.0;
            }
        }
        else if (ray_type == 2)  // diagonal rays
        {
            // Diagonal rays from bottom to top-right
            x_pos = 0.5 + 1.0 * r / (nrays - 1);

            ray_starts[r](0) = x_pos;
            ray_ends[r](0)   = x_pos + 1.0;

            ray_starts[r](1) = 0.0;
            ray_ends[r](1)   = 1.0;

            if (dim == 3) {
                ray_starts[r](2) = 0.0;
                ray_ends[r](2)   = 1.0;
            }
        }
    }
}
