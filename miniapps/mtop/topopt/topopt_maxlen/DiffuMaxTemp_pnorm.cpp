// Heat conduction topology optimization: minimize volume subject to a
// maximum-temperature constraint, approximated by a p-norm of T.
//
//    min_ρ   V(ρ~) = ∫ ρ~ dx / |Ω|
//    s.t.    (r_f^2 K + M) ρ~ = M_fc ρ                  (PDE filter)
//            -div(k(ρ~) grad T) = q,   T = 0 on Γ_s     (heat sink)
//            P(T) = ( 1/|Ω| ∫ T^p dx )^{1/p} ≤ T_max
//            0 ≤ ρ ≤ 1
//
// P ≤ max T for every p (P → max T as p → ∞), so P ≤ T_max does not
// guarantee max T ≤ T_max: both are reported every iteration.
//
// Sample run:  mpirun -np 8 ./DiffuMaxTemp_pnorm
//              mpirun -np 8 ./DiffuMaxTemp_pnorm -p 16 -tmax 12 -mi 300

#include "mfem.hpp"
#include "DiffuTopOpt.hpp"
#include "qoi.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../diffusion_mass_solver.hpp"
#include <memory>
#include <fstream>
#include <sstream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 1. Options.
    int    ref_levels   = 5;          // >= 5 needed to resolve the heat sink
    int    order        = 2;
    real_t r_f          = 0.03;       // min filter length
    real_t T_max        = 20.0;       // max temperature
    real_t p            = 8.0;        // p-norm exponent
    int    p_ir_order   = 0;          // quadrature order for T^p; 0 = ceil(p) * order
    real_t rho0         = 0.5;        // initial design (mean, +/- 0.05 random perturbation)
    const int seed      = 0;          // random perturbation of the initial design
    int    max_it       = 200;
    real_t tol          = 1e-3;       // stopping tol on iteration error
    real_t move         = 0.1;        // MMA move limit

    bool visualization = false;
    bool paraview      = false;
    int  paraview_interval = 0;       // intermediate output interval; 0 disables it

    const real_t k_min    = 1e-3;     // SIMP void conductivity
    const real_t k_max    = 1.0;      // SIMP max conductivity
    const real_t exponent = 3.0;      // SIMP exponent

    OptionsParser args(argc, argv);
    args.AddOption(&ref_levels, "-r", "--refine", "uniform refinement levels");
    args.AddOption(&order, "-o", "--order", "finite element order");
    args.AddOption(&r_f, "-rf", "--r_fwidth", "min filter width");
    args.AddOption(&T_max, "-tmax", "--t-max", "maximum allowed temperature");
    args.AddOption(&p, "-p", "--pnorm", "p-norm exponent for the max-temperature constraint");
    args.AddOption(&p_ir_order, "-pq", "--pnorm-quad-order",
                    "quadrature order for the p-norm; 0 = ceil(p) * order");
    args.AddOption(&rho0, "-rho0", "--rho0", "initial design (mean)");
    args.AddOption(&max_it, "-mi", "--max-it", "max optimization iterations");
    args.AddOption(&tol, "-tol", "--tol", "stopping tol on max design change");
    args.AddOption(&move, "-mv", "--move", "MMA move limit");
    args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                    "store solution in paraview");
    args.AddOption(&paraview_interval, "-pvi", "--paraview-interval",
                    "write ParaView fields every N optimization iterations; 0 disables intermediate output");
    args.AddOption(&visualization, "-vis", "--visualization",
                    "-no-vis", "--no-visualization", "enable GLVis visualization");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0) { args.PrintUsage(cout); }
        return 1;
    }
    if (myid == 0) { args.PrintOptions(cout); }
    MFEM_VERIFY(paraview_interval >= 0,
                "ParaView output interval must be nonnegative.");

    // 2. Build the mesh: 2 x 2 square.
    //    Heat sink (T = 0) on the bottom edge, x ∈ [0.95, 1.05].
    Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, true, 2.0, 2.0);
    const int dim = mesh.Dimension();

    // 3. Refine the mesh, mark the heat sink, and construct pmesh
    for (int l = 0; l < ref_levels; l++)
    {
        mesh.UniformRefinement();
    }

    const int sink_attr = mesh.bdr_attributes.Max() + 1;
    int n_sink = 0;
    for (int i = 0; i < mesh.GetNBE(); i++)
    {
        Element *be = mesh.GetBdrElement(i);
        Array<int> v;
        be->GetVertices(v);
        const real_t *v0 = mesh.GetVertex(v[0]), *v1 = mesh.GetVertex(v[1]);

        const real_t y_avg = 0.5 * (v0[1] + v1[1]);
        const real_t x_min = std::min(v0[0], v1[0]);
        const real_t x_max = std::max(v0[0], v1[0]);
        if (std::abs(y_avg) < 1e-8 && x_min >= 0.95 - 1e-8 && x_max <= 1.05 + 1e-8)
        {
            be->SetAttribute(sink_attr);
            n_sink++;
        }
    }
    // without a resolved sink the problem is pure Neumann with q > 0: no solution
    MFEM_VERIFY(n_sink > 0, "heat sink is not resolved by the mesh; increase -r");
    mesh.SetAttributes();

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    // 4. Define finite element collections and spaces
    H1_FECollection state_fec(order, dim);       // T, λ
    H1_FECollection filter_fec(order, dim);
    H1_FECollection control_fec(order-1, dim, BasisType::GaussLobatto);

    ParFiniteElementSpace state_fes(&pmesh, &state_fec);
    ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);
    ParFiniteElementSpace control_fes(&pmesh, &control_fec);

    // Printing all true dofs.
    HYPRE_BigInt state_size  = state_fes.GlobalTrueVSize();
    HYPRE_BigInt filter_size = filter_fes.GlobalTrueVSize();
    HYPRE_BigInt design_size = control_fes.GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "\nstate (T) dofs = " << state_size
             << ",  filter dofs = " << filter_size
             << ",  design dofs = " << design_size << endl;
    }

    // 5. Initialize all the grid functions and coefficients
    ParGridFunction rho(&control_fes);
    ParGridFunction rho_filter(&filter_fes);

    rho.Randomize(seed);
    rho -= 0.5;
    rho *= 0.1;
    rho += rho0;
    rho_filter = rho0;

    ConstantCoefficient one_cf(1.0);
    GridFunctionCoefficient rho_cf(&rho);
    GridFunctionCoefficient rho_filter_cf(&rho_filter);

    // SIMP conductivity: k(rho~) = k_min + rho~^p (k_max - k_min)
    SIMPCoefficient simp_cf(&rho_filter, k_min, k_max, exponent);                // k(rho~)
    SIMPGradCoefficient simp_grad_cf(&rho_filter, k_min, k_max, exponent);       // k'(rho~)

    // 6. Construct the solvers.
    // 6a. Heat diffusion solver (T = 0 on the heat sink, uniform source q = 1).
    ConstantCoefficient heat_source_cf(1.0);

    DiffusionSolver heat(state_fes, simp_cf);
    heat.SetEssentialBC(sink_attr, 0.0);
    heat.SetHeatSource(heat_source_cf);

    ParGridFunction &T      = heat.GetTemperature();
    ParGridFunction &lambda = heat.GetAdjoint();

    // dP/drho~ = k'(rho~) gradT . gradλ,  integrated with the DiffusionIntegrator
    // rule so it is the exact derivative of the assembled K(rho~)
    GradTGradLambdaCoefficient gTgL_cf(&T, &lambda);
    ProductCoefficient dPdrho_tilde_cf(simp_grad_cf, gTgL_cf);
    const FiniteElement &state_fe = *state_fes.GetFE(0);
    const IntegrationRule &sens_ir = DiffusionIntegrator::GetRule(state_fe, state_fe);

    // 6b. Min length scale filter solver
    PDEFilter filter(control_fes, filter_fes);
    filter.SetFilterRadius(r_f);
    DiffusionMassSolver &filter_solver = filter.GetSolver();
    for(int a = 1; a <= pmesh.bdr_attributes.Max(); a++){
        if (a != sink_attr) filter_solver.AddBoundaryID(a);
    }
    filter.Assemble();

    // 7. Construct the quantity of interest objects
    // volume of rho~ relative to |Omega|  (vol_fraction = 1 so V* = |Omega|)
    VolumeResidual vol_qoi(MPI_COMM_WORLD, &filter_fes, &rho_filter_cf, &one_cf, 1.0);

    // temperature p-norm:  P(T) = ( 1/|Ω| ∫ T^p )^{1/p}
    TemperaturePNorm temp_pnorm(T, p, p_ir_order);
    if (myid == 0)
    {
        cout << "p-norm quadrature order = " << temp_pnorm.GetIntRuleOrder() << endl;
    }

    // 8. MMA optimizer and its per-iteration work vectors.
    const int n = control_fes.GetTrueVSize();
    const int m = state_fes.GetTrueVSize();
    const int num_con = 1;                          // constraints: max temperature

    Vector rho_tv(n), rho_old(n), T_tv(m);
    rho.GetTrueDofs(rho_tv);

    Vector a(num_con), c(num_con), d(num_con);
    a = 0.0; c = 1000.0; d = 0.0;
    mfem_mma::MMAOptimizerParallel mma(MPI_COMM_WORLD, n, num_con, rho_tv, a, c, d);

    // 8a. objective initialization
    Vector df0dx(n);                                // dV/drho
    Vector dvol_tilde(filter_fes.GetTrueVSize());   // dV/drho~

    // 8b. local constraints
    Vector fival(num_con);
    Vector dfidx[num_con];

    Vector dPdrho(n);                               // dP/drho
    Vector dPdT(m);                                 // dP/dT  (adjoint rhs = -dP/dT)

    // 8c. mma upper and lower bounds
    Vector tx_min(n), tx_max(n);

    // 9. Visualizations
    // 9a. GLVis
    char vishost[] = "localhost";  int visport = 19916;
    socketstream sout_filter, sout_temp;

    if (visualization) {
        sout_filter.open(vishost, visport);
        sout_filter.precision(8);

        sout_filter << "parallel " << num_procs << " " << myid << "\n"
            << "solution\n" << pmesh << rho_filter
            << "window_title 'Filtered density rho~'\n"
            << "window_geometry 0 0 800 600\n"
            << "colorbar_numberformat '%.2f'\n"
            << "keys Rjlc\n" << flush;

        sout_temp.open(vishost, visport);
        sout_temp.precision(8);

        sout_temp << "parallel " << num_procs << " " << myid << "\n"
            << "solution\n" << pmesh << T
            << "window_title 'Temperature T'\n"
            << "window_geometry 810 0 800 600\n"
            << "colorbar_numberformat '%.2f'\n"
            << "keys Rjlc\n" << flush;
    }

    // 9b. Paraview
    std::ostringstream run_tag;
    run_tag << "maxtemp_pnorm_p" << p << "_tmax" << T_max;
    ParaViewDataCollection paraview_dc(run_tag.str(), &pmesh);

    if (paraview) {
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.RegisterField("rho", &rho);
        paraview_dc.RegisterField("rho_filter", &rho_filter);
        paraview_dc.RegisterField("temperature", &T);
    }

    // 9c. CSV convergence log.
    std::ofstream csv;
    if (myid == 0)
    {
        csv.open("convergence_pnorm.csv");
        csv << "it,volume,pnorm,maxT,fival,iterErr\n";
    }

    // 10. Optimization loop.
    real_t iterationError = 1.0;

    int it = 1;
    for (; it <= max_it && iterationError > tol; it++)
    {
        // (1) forward filter:  (r_f^2 K + M) ρ~ = M_fc ρ
        rho.GetTrueDofs(rho_tv);
        Vector rho_filter_tv(filter_fes.GetTrueVSize());
        filter.Mult(rho_tv, rho_filter_tv);
        rho_filter.SetFromTrueDofs(rho_filter_tv);

        // (2) objective and gradient:  volume, backward filter
        //       V       = ∫ ρ~ / |Ω|
        //       dV/drho = M_fc^T (r_f^2 K + M)^{-1} (1, φ_i) / |Ω|
        real_t vol = vol_qoi.Eval();
        vol_qoi.GetGrad(dvol_tilde);
        filter.MultTranspose(dvol_tilde, df0dx);

        // (3) max-temperature constraint and gradient:
        //     heat forward, constraint value, heat backward, backward filter
        //     forward:   K(ρ~) T = q,                     T = 0 on Γ_s
        //     P = ( 1/|Ω| ∫ T^p )^{1/p} ≤ T_max
        //     dP/dT_i   = P^{1−p} / |Ω| ∫ T^{p−1} φ_i
        //     adjoint:  K(ρ~) λ = −dP/dT,   λ = 0 on Γ_s   (same K as the forward solve)
        //     dP/drho~  = ∫ k'(ρ~) ∇T·∇λ φ_i
        //     dP/drho   = M_fc^T (r_f^2 K + M)^{-1} dP/drho~
        heat.Assemble();            // K(ρ~) changes every iteration
        heat.FSolve();

        T.GetTrueDofs(T_tv);
        real_t maxT_loc = T_tv.Max(), maxT;
        MPI_Allreduce(&maxT_loc, &maxT, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, MPI_COMM_WORLD);

        real_t pnorm = temp_pnorm.Eval();
        temp_pnorm.GetGrad(dPdT);

        Vector adj_rhs(dPdT);
        adj_rhs.Neg();
        heat.ASolve(adj_rhs);

        ParLinearForm sens_rhs(&filter_fes);
        sens_rhs.AddDomainIntegrator(new DomainLFIntegrator(dPdrho_tilde_cf, &sens_ir));
        sens_rhs.Assemble();
        std::unique_ptr<HypreParVector> sens_rhs_tv(sens_rhs.ParallelAssemble());
        filter.MultTranspose(*sens_rhs_tv, dPdrho);

        // constraint normalized by T_max:  P/T_max − 1 ≤ 0
        fival(0) = pnorm / T_max - 1.0;                 // update constraint value
        dPdrho /= T_max;
        dfidx[0] = dPdrho;                              // update constraint gradient

        // intermediate ParaView output, written before the MMA update so that
        // rho, rho~, T all belong to the same (current) design
        if (paraview && paraview_interval > 0 && it % paraview_interval == 0)
        {
            paraview_dc.SetCycle(it);
            paraview_dc.SetTime(it);
            paraview_dc.Save();
        }

        // (4) box constraints:  rho ∈ [0,1]  (move limits)
        for (int i = 0; i < n; i++)
        {
            tx_min[i] = std::max(real_t(0), rho_tv[i] - move);
            tx_max[i] = std::min(real_t(1), rho_tv[i] + move);
        }

        // (5) MMA update on rho
        rho_old = rho_tv;

        // objective: volume fraction V = ∫ρ~/|Ω| ∈ [0,1], already O(1), unscaled
        mma.Update(rho_tv, df0dx, vol, fival, dfidx, tx_min, tx_max);
        rho.SetFromTrueDofs(rho_tv);

        // measure iteration error
        ParGridFunction rho_old_gf(&control_fes);
        rho_old_gf.SetFromTrueDofs(rho_old);
        iterationError = rho_old_gf.ComputeL1Error(rho_cf);

        if (myid == 0)
        {
            const int w = 14;               // column width
            mfem::out << "\niteration " << it << '\n' << left
                    << setw(w) << "volume"
                    << setw(w) << "pnorm"
                    << setw(w) << "maxT"
                    << setw(w) << "fival"
                    << setw(w) << "iterErr" << '\n'
                    << string(5*w, '=') << '\n'
                    << fixed      << setprecision(4) << setw(w) << vol
                    <<               setprecision(4) << setw(w) << pnorm
                    <<               setprecision(4) << setw(w) << maxT
                    << scientific << setprecision(3) << setw(w) << fival(0)
                    <<               setprecision(4) << setw(w) << iterationError << endl;

            csv << it << ','
                << scientific << setprecision(8) << vol << ','
                << pnorm << ','
                << maxT << ','
                << fival(0) << ','
                << iterationError << '\n';
            csv.flush();
        }

        if (visualization)
        {
            sout_filter << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n" << pmesh << rho_filter << flush;

            sout_temp << "parallel " << num_procs << " " << myid << "\n"
                << "solution\n" << pmesh << T << flush;
        }
    }

    if (myid == 0)
    {
        csv.close();
        mfem::out << "\nfinished after " << (it - 1) << " iterations\n";
    }

    // Post-process the final design. The last MMA update produced a rho that
    // was never evaluated: refilter and re-solve so rho, rho~, T in the final
    // output all belong to the same design.
    {
        rho.GetTrueDofs(rho_tv);
        Vector rho_filter_tv(filter_fes.GetTrueVSize());
        filter.Mult(rho_tv, rho_filter_tv);
        rho_filter.SetFromTrueDofs(rho_filter_tv);

        heat.Assemble();
        heat.FSolve();

        T.GetTrueDofs(T_tv);
        real_t maxT_loc = T_tv.Max(), maxT;
        MPI_Allreduce(&maxT_loc, &maxT, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, MPI_COMM_WORLD);

        const real_t vol   = vol_qoi.Eval();
        const real_t pnorm = temp_pnorm.Eval();

        if (myid == 0)
        {
            mfem::out << "\nfinal design (after last MMA update)\n"
                      << "  volume = " << fixed      << setprecision(4) << vol   << '\n'
                      << "  pnorm  = " <<                                 pnorm << '\n'
                      << "  fival  = " << scientific << setprecision(3) << pnorm / T_max - 1.0 << '\n'
                      << "  maxT   = " << fixed      << setprecision(4) << maxT
                      << "  (T_max = " << T_max << ", p = " << p << ")" << endl;
        }

        if (paraview)
        {
            paraview_dc.SetCycle(it);
            paraview_dc.SetTime(it);
            paraview_dc.Save();
        }
    }

    return 0;
}
