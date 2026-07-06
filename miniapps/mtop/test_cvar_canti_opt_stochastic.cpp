#include "mfem.hpp"
#include "mtop_solvers.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <utility>
#include <bitset>
#include <random>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <ctime>
#include <algorithm>
#include <tuple>
#include <stdexcept>
#include <cstddef>   // size_t
#include <numeric>   // std::accumulate
#include <cmath>     // std::abs, std::isfinite

#include "test_cvar_canti_opt_stochastic_easiest_helpers.hpp"
#include "test_cvar_canti_opt_stochastic_simpl_helpers.hpp"

using namespace std;
using namespace mfem;


// Return true if we are in a deterministic phase.
// recall we use 1-indexing for the outer_iteration.
bool is_deterministic_step(
    unsigned int outer_iteration,
    const int RUN_DETERMINISTIC_UNTIL,
    const int RUN_DETERMINISTIC_AFTER
) {
    const bool deterministic_until =
        (RUN_DETERMINISTIC_UNTIL < 0) ||
        (RUN_DETERMINISTIC_UNTIL >= 0 && outer_iteration <= RUN_DETERMINISTIC_UNTIL);
    const bool deterministic_after =
        (RUN_DETERMINISTIC_AFTER >= 0 && outer_iteration >= RUN_DETERMINISTIC_AFTER);
    return deterministic_until || deterministic_after;
}

// Return true if Heaviside projection should be active at this outer iteration.
bool is_heaviside_step(
    unsigned int outer_iteration,
    const bool ENABLE_HEAVISIDE_PROJECTION,
    const int RUN_HEAVISIDE_AFTER
) {
    if (!ENABLE_HEAVISIDE_PROJECTION)
    {
        return false;
    }
    return (RUN_HEAVISIDE_AFTER < 0) || (outer_iteration >= static_cast<unsigned int>(RUN_HEAVISIDE_AFTER));
}

// Cosine schedule for SIMP power during non-Heaviside phases.
real_t simp_power_schedule(
    unsigned int outer_iteration,
    const real_t simp_power_mean,
    const real_t simp_power_amplitude,
    const real_t simp_power_period)
{
    return simp_power_mean - simp_power_amplitude * std::cos(M_PI * static_cast<real_t>(outer_iteration) / simp_power_period);
}

int main(int argc, char *argv[])
{
    // initialization. {} scoping helps wih navigating code.
    // 0. Generate Randomness
    std::mt19937_64 rng(123456); // seed however you like

    // 1. Initialize MPI and HYPRE.
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();
    // 2. Parse command-line options.
    // const char *mesh_file = "canti_2D_6.msh"; // SYMMETRIC MESH
    const char *mesh_file = "canti_n_2D_6.msh"; // ASYMMETRIC MESH

    // canti_2D_6.msh
    // const int fixed_node_bc_index = 1;
    // const int outer_boundary_bc_index = 8;
    // const std::array<int, N> bit_index_to_mesh_bc_index = {2, 3, 4, 5, 6, 7};
    

    // canti_n_2D_6.msh
    const int fixed_node_bc_index = 5;
    const int outer_boundary_bc_index = 8;
    const std::array<int, N> bit_index_to_mesh_bc_index = {2, 3, 4, 6, 7, 1};


    int order = 2;
    bool static_cond = false;
    bool pa = false;
    bool fa = false;
    const char *device_config = "cpu";
    bool visualization = true;
    bool algebraic_ceed = false;
    real_t filter_radius = real_t(0.07); // original 0.01

    // original 0.005 for symmetric.
    const real_t starting_gamma = 0.005; // gamma is the stepsize for updating the prox for the latent probabilities.
    real_t gamma = starting_gamma;

    // starting for armijo backtracking line search. Nothing to do with cvar.
    // There are more intelligent ways for pure-primal algorithms to update this, since, in our algorithm
    // we update q using the stepsize before updating x, this is a simpler initialization. 

    const real_t starting_alpha_stepsize = 1.0; // no opinion here. But, hey, let's be aggresive since we are doing backtracking line search. Original 1.0 
    const real_t ARMIJO_C1 = 1e-4; // armijo backtracking line search parameter. Original 1e-4
    const real_t ARMIJO_BETA = 0.5; // armijo backtracking line search parameter. Original 0.5
    const real_t MIN_ALPHA = 1e-2;
    const real_t MAX_ALPHA = 1.0;
    const real_t ALPHA_SCALING_FACTOR_OVERRIDE = 1.0; // if alpha goes out of bounds, we scale it back by this factor. Original 0.5

    const real_t CLAMP_VAL = 20;

    // originally 0.01 for symmetric. 0.005 for asymmetric. This is the rho in the augmented Lagrangian for the CVaR constraint. It is a penalty parameter for the CVaR constraint. It is updated in the outer loop.
    const real_t starting_rho = 0.1; 
    real_t rho = starting_rho;

    real_t vol_fraction = 0.3;
    // int max_it = 100;
    real_t itol = 1e-1;
    real_t ntol = 1e-4;
    real_t rho_min = 1e-3;
    real_t E_min = 1e-6; 
    real_t E_max = 1.0;
    real_t poisson_ratio = 0.02;
    bool glvis_visualization = true;
    bool paraview_output = true;

    // Initialization options for the design domain
    bool init_with_hole = false;             // Initialize with a symmetric hole in left-center
    real_t hole_radius = 0.1;                // Radius of hole as fraction of domain
    real_t hole_strength = 0.6;              // Strength of hole (0-1, where 1 removes material)
    real_t hole_size_x = 0.15;              // Position of hole center in x-direction as fraction from left edge

    // Promote black-white designs without changing the filter radius.
    bool use_heaviside_projection = false;
    real_t heaviside_eta_d = 0.45; // Small is 0.475
    real_t heaviside_eta_e = 0.55; // Small is 0.525
    real_t heaviside_beta = 8.0;
    real_t simp_power_mean = 3.5;                          // Mean value in cosine SIMP schedule when Heaviside is OFF
    real_t simp_power_amplitude = 0.5;                     // Amplitude in cosine SIMP schedule when Heaviside is OFF
    real_t simp_power_period = 2;                       // Period in cosine SIMP schedule when Heaviside is OFF
    const real_t SIMP_POWER_DEFAULT_FOR_HEAVISIDE = 1.0; // SIMP power used when Heaviside is ON (projection handles binarization)

    const real_t tau = 1e-10; // threshold for probability truncation.

    const real_t p1 = 0.01; 
    const real_t p2 = 0.01; 
    const real_t p3 = 0.01; 
    const real_t p4 = 0.005; 

    const real_t cvar_alpha = 0.95;
    const int outer_loop_iterations = 100;
    const int inner_loop_iterations = 15;
    const int MAX_BACKTRACKING_ATTEMPTS = 20; // maximum number of backtracking attempts in each inner loop iteration.

    const real_t block_size_ratio = 0.35; // trying full blocks // change back to 0.35
    const int STOCHASTIC_GRADIENT_MINIBATCH_SIZE = 5;

    // Deterministic for outer <= RUN_DETERMINISTIC_UNTIL.
    // If RUN_DETERMINISTIC_AFTER >= 0, deterministic again for outer >= RUN_DETERMINISTIC_AFTER.
    // RUN_DETERMINISTIC_AFTER = -1 means this second phase never activates.
    // Purely deterministic configuration: deterministic window always activates.
    const int RUN_DETERMINISTIC_UNTIL = -1;
    const int RUN_DETERMINISTIC_AFTER = -1;
    const bool RUN_SYMMETRIC = false;

    // Enable Heaviside projection starting from this outer iteration.
    // -1 means active from the start (when use_heaviside_projection is true).
    const int RUN_HEAVISIDE_AFTER = -1;

    if (RUN_DETERMINISTIC_UNTIL >= 0 && RUN_DETERMINISTIC_AFTER >= 0 &&
        RUN_DETERMINISTIC_AFTER < RUN_DETERMINISTIC_UNTIL)
    {
        throw std::invalid_argument(
            "RUN_DETERMINISTIC_AFTER must be >= RUN_DETERMINISTIC_UNTIL when both are non-negative.");
    }

    if (RUN_HEAVISIDE_AFTER == 0)
    {
        throw std::invalid_argument("RUN_HEAVISIDE_AFTER must be -1 or >= 1.");
    }

    if (simp_power_period <= 0.0)
    {
        throw std::invalid_argument("simp_power_period must be > 0.");
    }

    const bool heaviside_active_at_start =
        is_heaviside_step(1, use_heaviside_projection, RUN_HEAVISIDE_AFTER);
    const real_t simp_power_at_start = 3.0;
    // simp_power_schedule(
        // 1, simp_power_mean, simp_power_amplitude, simp_power_period);
    bool heaviside_active = heaviside_active_at_start;

    // "const" or "gradient"
    const float epsilon_TV = 1e-3;
    const float epsilon_g = 1e-3;
    const float epsilon_q = 1e-6;

    std::vector<std::pair<std::bitset<N>, real_t>> probability_space = getProbabilitySpace(p1, p2, p3, p4);
    // std::vector<std::pair<std::bitset<N>, real_t>> probability_space = nonProbabilitySpace();

    std::vector<size_t> symmetric_index_vector(probability_space.size());
    generate_symmetric_index_vector(probability_space, symmetric_index_vector);

    real_t total_probability = 0.0;
    for (const auto& [bits, p] : probability_space) {  
        total_probability += p; 
    }
    if (std::abs(total_probability - 1.0) > 1e-6) {
        throw std::runtime_error("Probabilities do not sum to 1.");
    }

    if (myid == 0) {
        std::cout << "Size of probability space: " << probability_space.size() << "\n";
    }

    const int BLOCK_SIZE = int(block_size_ratio * probability_space.size());

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                   "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                   "--no-partial-assembly", "Enable Partial Assembly.");
    args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                   "--no-full-assembly", "Enable Full Assembly.");
    args.AddOption(&device_config, "-d", "--device",
                   "Device configuration string, see Device::Configure().");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&filter_radius, "-fr", "--filter",
                   "Set the filter radius.");
    args.AddOption(&init_with_hole, "-hole", "--init-hole", "-no-hole", "--no-init-hole",
                   "Initialize with a symmetric hole in left-center of domain.");
    args.AddOption(&hole_radius, "-hr", "--hole-radius",
                   "Radius of the hole as fraction of domain (default 0.1).");
    args.AddOption(&hole_strength, "-hs", "--hole-strength",
                   "Strength of hole: 0-1 where 1 removes material (default 0.8).");
    args.AddOption(&use_heaviside_projection, "-hproj", "--heaviside-projection", "-no-hproj", "--no-heaviside-projection",
                   "Enable Heaviside projection of filtered densities to promote 0-1 designs.");
    args.AddOption(&heaviside_eta_d, "-heta-d", "--heaviside-eta-d",
                   "Threshold eta for Heaviside projection in volume/projection computations (default 0.3).");
    args.AddOption(&heaviside_eta_e, "-heta-e", "--heaviside-eta-e",
                   "Threshold eta for Heaviside projection in compliance and gradient computations (default 0.7).");
    args.AddOption(&heaviside_beta, "-hb", "--heaviside-beta",
                   "Beta for Heaviside projection continuation (default 1.0).");
    args.AddOption(&simp_power_mean, "-sp", "--simp-power",
                   "Mean SIMP power used in cosine schedule during non-Heaviside phases.");
    args.AddOption(&simp_power_amplitude, "-spa", "--simp-power-amplitude",
                   "Amplitude for cosine SIMP schedule during non-Heaviside phases.");
    args.AddOption(&simp_power_period, "-spp", "--simp-power-period",
                   "Period for cosine SIMP schedule during non-Heaviside phases.");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(cout);
    }

    // 3. Enable hardware devices such as GPUs, and programming models such as
    //    CUDA, OCCA, RAJA and OpenMP based on command line options.
    Device device(device_config);
    if (myid == 0)
    {
        device.Print();
    }

    // 4. Read the (serial) mesh from the given mesh file on all processors.  We
    //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
    //    and volume meshes with the same code.
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    // 5. Refine the serial mesh on all processors to increase the resolution. In
    //    this example we do 'ref_levels' of uniform refinement. We choose
    //    'ref_levels' to be the largest number that gives a final mesh with no
    //    more than 10,000 elements.
    {
        int ref_levels =
            (int)floor(log(1000. / mesh.GetNE()) / log(2.) / dim) + 1;
        for (int l = 0; l < ref_levels; l++)
        {
            mesh.UniformRefinement();
        }
    }

    // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.
    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();
    {
        int par_ref_levels = 1;
        for (int l = 0; l < par_ref_levels; l++)
        {
            pmesh.UniformRefinement();
        }
    }

    // allocate the fiter
    FilterOperator *filt = new FilterOperator(filter_radius, &pmesh);
    // set the boundary conditions

    // @TODO handle this within the initialization code. Too manual.
    filt->AddBC(fixed_node_bc_index, 1.0); // free hold

    for (int i = 0; i < N; i++)
    {
        filt->AddBC(bit_index_to_mesh_bc_index[i], 1.0);
    };

    // filt->AddBC(2, 1.0);
    // filt->AddBC(3, 1.0);
    // filt->AddBC(4, 1.0);
    // filt->AddBC(5, 1.0);
    // filt->AddBC(6, 1.0);
    // filt->AddBC(7, 1.0);

    filt->AddBC(outer_boundary_bc_index, 0.0); // outer boundary

    // allocate the slover after setting the BC and before applying the filter
    filt->Assemble();

    // define the control latent variable. Get an "old" version for saving later
    // However, we note we don't need to be as careful as with q because we only need the latest version of the density
    ParGridFunction odens_latent(filt->GetDesignFES());
    ParGridFunction odens_latent_old(filt->GetDesignFES());
    // ParGridFunction odens_latent_old_old(filt->GetDesignFES()); // for GBB test

    mfem::Vector CLAMP_MIN_VECTOR(filt->GetDesignFES()->TrueVSize()); CLAMP_MIN_VECTOR = -CLAMP_VAL;
    mfem::Vector CLAMP_MAX_VECTOR(filt->GetDesignFES()->TrueVSize()); CLAMP_MAX_VECTOR = CLAMP_VAL;

    // define the control field
    MappedGridFunctionCoefficient odens(&odens_latent, sigmoid);
    MappedGridFunctionCoefficient odens_old(&odens_latent_old, sigmoid);
    // MappedGridFunctionCoefficient odens_old_old(&odens_latent_old_old, sigmoid); // for GBB test
    // define the filtered field
    ParGridFunction fdens(filt->GetFilterFES());
    ParGridFunction hbeta_fdens_e(filt->GetFilterFES());
    ParGridFunction hbeta_fdens_d(filt->GetFilterFES());

    PostHeavisideCoefficient post_heaviside_coeff_e(
        &fdens, &heaviside_active, &heaviside_eta_e, &heaviside_beta);
    PostHeavisideCoefficient post_heaviside_coeff_d(
        &fdens, &heaviside_active, &heaviside_eta_d, &heaviside_beta);

    // hold the gradient of the functional. Used for GBB step size estimation.
    ParGridFunction odens_weighted_gradient(filt->GetDesignFES());
    ParGridFunction odens_weighted_gradient_old(filt->GetDesignFES());

    odens_weighted_gradient = 0.0;
    odens_weighted_gradient_old = 0.0;

    // set the elasticity solver
    IsoLinElasticSolver *elsolver = new IsoLinElasticSolver(&pmesh, order);
    // set up variables

    // 9. Define some tools for later.
    ConstantCoefficient zero(0.0);
    ConstantCoefficient one(1.0);
    ParGridFunction onegf(filt->GetFilterFES());
    onegf = 1.0;
    //ParGridFunction zerogf(filt->GetDesignFES());
    //zerogf = 0.0;
    ParLinearForm vol_form(filt->GetDesignFES());
    vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
    vol_form.Assemble();
    real_t domain_volume;
    real_t target_volume;
    {
        ParGridFunction donegf(filt->GetDesignFES());
        donegf = 1.0;
        domain_volume = vol_form(donegf);
        target_volume = domain_volume * vol_fraction;
        if (0 == myid)
        {
            std::cout << "domain volume=" << domain_volume << " target_volume=" << target_volume << std::endl;
        }
    }


    Vector odens_latent_vector(filt->GetDesignFES()->TrueVSize());
    
    // Apply selected initialization strategy
    if (init_with_hole)
    {
        if (myid == 0)
        {
            std::cout << "Initializing with symmetric hole in left-center." << std::endl
                      << "  Hole radius: " << hole_radius << std::endl
                      << "  Hole strength: " << hole_strength << std::endl;
        }
        initialize_with_hole(odens_latent, target_volume, domain_volume, myid,
                             hole_radius, hole_strength, hole_size_x,
                             heaviside_active_at_start, heaviside_eta_d, heaviside_beta);
    }
    else
    {
        // Default uniform initialization
        odens_latent = inv_sigmoid(vol_fraction);
    }
    
    odens_latent.SetTrueVector();
    odens_latent.GetTrueDofs(odens_latent_vector);

    // odens_gradient_holder = 0.0;

    // Vector odens_vector(filt->GetDesignFES()->TrueVSize());
    // The primal design density filed
    ParGridFunction odens_gf = ParGridFunction(filt->GetDesignFES());
    odens_gf.ProjectCoefficient(odens);
    // odens_gf.SetTrueVector();
    // odens_gf.GetTrueDofs(odens_vector);

    // gradient in design space
    // true gradient vector on
    // Vector ograd(odens_gf.GetTrueVector().Size());
    // gradient grid function
    ParGridFunction odens_gradient_single(filt->GetDesignFES());
    odens_gradient_single = 0.0;
    // define the control latent variable gradient. To be determined
    // ParGridFunction odens_latent_cumulative(filt->GetDesignFES());
    // odens_latent_cumulative = 0.0;


    // allocate these now. Used for testing later down.
    ParGridFunction projected_gradient(odens_latent.ParFESpace());
    ParGridFunction odens_difference(odens_latent.ParFESpace());
    ParGridFunction odens_latent_difference(odens_latent.ParFESpace());
    ParGridFunction odens_gradient_difference(odens_latent.ParFESpace());

    Vector fdv(filt->GetFilterFES()->TrueVSize());
    fdv = 0.0; // define the true vector of the filtered field

    // filter the initial density
    filt->FFilter(&odens,fdens);
    hbeta_fdens_e.ProjectCoefficient(post_heaviside_coeff_e);
    hbeta_fdens_d.ProjectCoefficient(post_heaviside_coeff_d);

    ParGridFunction& sol=elsolver->GetDisplacements();

    char timestamp_buffer[32] = {0};
    if (myid == 0)
    {
        const std::time_t now = std::time(nullptr);
        std::tm local_tm;
        localtime_r(&now, &local_tm);
        std::strftime(timestamp_buffer, sizeof(timestamp_buffer), "%Y%m%d_%H%M%S", &local_tm);
    }
    MPI_Bcast(timestamp_buffer, static_cast<int>(sizeof(timestamp_buffer)), MPI_CHAR, 0, MPI_COMM_WORLD);

    const std::string run_name =
        "cvar_optimization_asymmetric_18Jun2026_DET_RHO=0.1_" + std::string(timestamp_buffer);

    // set up the paraview
    mfem::ParaViewDataCollection paraview_dc(run_name, &pmesh);
    // rho_gf.ProjectCoefficient(rho);
    paraview_dc.SetPrefixPath("ParaView");
    paraview_dc.SetLevelsOfDetail(order);
    paraview_dc.SetDataFormat(VTKFormat::BINARY);
    paraview_dc.SetHighOrderOutput(true);
    paraview_dc.SetCycle(0);
    paraview_dc.SetTime(0.0);
    paraview_dc.RegisterField("density", &odens_gf);
    paraview_dc.RegisterField("filtered_density", &fdens);
    paraview_dc.RegisterField("post_heaviside_density_e", &hbeta_fdens_e);
    paraview_dc.RegisterField("post_heaviside_density_d", &hbeta_fdens_d);
    paraview_dc.RegisterField("control_gradient", &odens_gradient_single);
    paraview_dc.RegisterField("latent_density", &odens_latent);
    paraview_dc.RegisterField("disp",&sol);
    // paraview_dc.RegisterField("displacements", &displacements);
    paraview_dc.Save();

    // define material and interpolation parameters coefficient factory
    IsoComplCoef icc;
    icc.SetGridFunctions(&fdens, &(elsolver->GetDisplacements())); // (1) density (2) displacements. Deferred calculation.
    icc.SetMaterial(E_min, E_max, poisson_ratio);
    icc.SetSIMP(heaviside_active_at_start ? SIMP_POWER_DEFAULT_FOR_HEAVISIDE : simp_power_at_start);
    if (heaviside_active_at_start)
    {
        icc.SetProj(heaviside_eta_e, heaviside_beta);
    }

    // set the material to the elasticity solver
    elsolver->SetMaterial(*(icc.GetE()), *(icc.GetNu())); // take parameters from ICC. It's a coefficient factory. GetE() is a function of density, GetNu() is a constant function (might be a density)

    // set surface load
    elsolver->AddSurfLoad(fixed_node_bc_index, 0.0, 1.0); // set the load on the free hole. 0.0 == x direction, 1.0 == y direction.

    ParBilinearForm mass(filt->GetDesignFES());
    mass.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator(one)));
    mass.Assemble();
    HypreParMatrix Minv;
    Array<int> empty;
    mass.FormSystemMatrix(empty, Minv);
    Vector tmp_grad;
    tmp_grad.SetSize(filt->GetDesignFES()->GetTrueVSize());

    // vector of [Bitset, Original Probability, Latent Probability]
    // ensures that $q$ is initialized as $p$.
    std::vector<std::tuple<std::bitset<N>, real_t, real_t>> latent_probabilities_k_0;
    {
        real_t latent_probability_i = log((1 - cvar_alpha) / cvar_alpha);
        for (auto &[bits, value] : probability_space)
        {
            latent_probabilities_k_0.emplace_back(bits, value, latent_probability_i);
        }
    }

    std::vector<std::tuple<std::bitset<N>, real_t, real_t>> latent_probabilities_k_jp1_unnormalized;
    std::vector<std::tuple<std::bitset<N>, real_t, real_t>> latent_probabilities_k_jp1;
    std::vector<std::tuple<std::bitset<N>, real_t, real_t>> latent_probabilities_k_j;

    latent_probabilities_k_j = latent_probabilities_k_0; // initialize the current latents to the initial latents. We will update this in the inner loop, and use it to calculate the new latents.

    using Index = decltype(latent_probabilities_k_jp1)::size_type;
    std::discrete_distribution<Index> q_distribution;

    // we initialize via the current value of "latent_probabilities_k_0." In the future, will use the current latents.
    // But we need blocks.
    // TODO: remove this while deterministic. Small cost but still. 
    generate_distribution_from_latents(
        latent_probabilities_k_0,
        cvar_alpha,
        q_distribution
    );


    int total_iterations = 0;
    long long total_gradient_evaluations = 0;
    long long total_function_evaluations = 0;

    std::ofstream cvar_csv;
    std::ofstream dual_probabilities_csv;
    const std::string cvar_csv_dir = "cvar_csv";
    const std::string cvar_csv_path = cvar_csv_dir + "/" + run_name + ".csv";
    const std::string dual_probabilities_csv_path =
        cvar_csv_dir + "/" + run_name + "_dual_probabilities.csv";
    if (myid == 0)
    {
        std::error_code ec;
        std::filesystem::create_directories(cvar_csv_dir, ec);
        if (ec)
        {
            throw std::runtime_error("Unable to create CSV directory '" + cvar_csv_dir + "': " + ec.message());
        }

        cvar_csv.open(cvar_csv_path);
        if (!cvar_csv)
        {
            throw std::runtime_error("Unable to open " + cvar_csv_path + " for writing.");
        }
        cvar_csv << std::setprecision(17);
        cvar_csv << "outer_iteration,inner_iteration,total_steps,total_gradient_evaluations,total_function_evaluations,armijo_descents,current_cvar,current_cvar_estimation\n";

        dual_probabilities_csv.open(dual_probabilities_csv_path);
        if (!dual_probabilities_csv)
        {
            throw std::runtime_error("Unable to open " + dual_probabilities_csv_path + " for writing.");
        }
        dual_probabilities_csv << std::setprecision(17);
        dual_probabilities_csv
            << "outer_iteration,inner_iteration,total_steps,scenario_index,scenario_bits,original_probability,latent_probability,dual_probability\n";
    }

    auto evaluate_compliance_for_scenario =
        [&](const std::bitset<N> &bits,
            const std::string &inner_iter_tag,
            const std::string &eval_reason,
            const bool verbose_bits_logging) -> real_t
    {
        if (myid == 0)
        {
            std::cout << inner_iter_tag
                      << " [compliance-eval] reason=" << eval_reason
                      << " scenario=" << bits.to_string()
                      << std::endl;
        }

        elsolver->DelDispBC();
        for (int bit_index = 0; bit_index < N; bit_index++)
        {
            if (!(bits.test(bit_index)))
            {
                elsolver->AddDispBC(bit_index_to_mesh_bc_index[bit_index], 4, 0.0);
            }
            else if (myid == 0 && verbose_bits_logging)
            {
                std::cout << inner_iter_tag << " [compliance-eval:" << eval_reason
                          << "] Skipping bit " << (bit_index - 2)
                          << " (mesh BC " << bit_index_to_mesh_bc_index[bit_index] << ")\n"
                          << " in scenario " << bits << ".\n";
            }
        }

        elsolver->Assemble();
        elsolver->FSolve();

        ParGridFunction &sol_eval = elsolver->GetDisplacements();
        icc.SetGridFunctions(&fdens, &sol_eval);

        ParLinearForm compl_form_eval(filt->GetFilterFES());
        compl_form_eval.AddDomainIntegrator(new DomainLFIntegrator(icc));
        compl_form_eval.Assemble();
        const real_t compliance_value = compl_form_eval(onegf);

        if (myid == 0)
        {
            std::cout << inner_iter_tag
                      << " [compliance-eval] reason=" << eval_reason
                      << " value=" << compliance_value
                      << std::endl;
        }

        return compliance_value;
    };

    auto dual_probability_from_latent =
        [&](const real_t original_probability, const real_t latent_probability) -> real_t
    {
        return sigmoid(latent_probability) * original_probability / (1 - cvar_alpha);
    };

    /**
     * Set the initial stepsize.
     */
    real_t alpha_stepsize = -1.0;

    // loop
    for (unsigned int outer = 1; outer <= outer_loop_iterations; outer++)
    {
        heaviside_active = is_heaviside_step(outer, use_heaviside_projection, RUN_HEAVISIDE_AFTER);
        const real_t simp_power_current = simp_power_at_start;
        // simp_power_schedule(
            // outer, simp_power_mean, simp_power_amplitude, simp_power_period);

        icc.SetSIMP(heaviside_active ? SIMP_POWER_DEFAULT_FOR_HEAVISIDE : simp_power_current);
        if (heaviside_active)
        {
            icc.SetProj(heaviside_eta_e, heaviside_beta);
        }

        if (myid == 0)
        {
            std::cout << "[projection] outer=" << outer
                        << " heaviside_active=" << heaviside_active
                        << " eta_d=" << heaviside_eta_d
                        << " eta_e=" << heaviside_eta_e
                        << " beta=" << heaviside_beta
                        << " simp_power_current=" << simp_power_current
                        << std::endl;
        }

        if (outer > 1)
        {
            // maybe this needs to escalate faster? write a better stepsize finder. Also maybe different gamma for inner and outer loops
            gamma *=  ((real_t)outer) / ((real_t)outer - 1); 
            // gamma = std::max(1.5 * gamma, ((real_t)outer) / ((real_t)outer - 1) * gamma);
            // this needs to go to 0. Opposite reasoning :-) 
            rho *= ((real_t)outer - 1) / ((real_t)outer); 
        }
        if (myid == 0)
        {
            cout << "\nStep = " << outer << endl;
        }
        // cout << "K IS " << k << endl;

        filt->FFilter(&odens, fdens);

        if (total_iterations == 0) {
            odens_gf.ProjectCoefficient(odens);
            hbeta_fdens_e.ProjectCoefficient(post_heaviside_coeff_e);
            hbeta_fdens_d.ProjectCoefficient(post_heaviside_coeff_d);
            paraview_dc.SetCycle(total_iterations);
            paraview_dc.SetTime((real_t)total_iterations);
            paraview_dc.Save();
        }


        // Block for update. Deterministic or sampled without replacement.
        std::vector<std::size_t> block_k;

        if (is_deterministic_step(outer, RUN_DETERMINISTIC_UNTIL, RUN_DETERMINISTIC_AFTER)) {
            block_k = std::vector<size_t>(latent_probabilities_k_0.size()); // Create vector of size N

            std::iota(block_k.begin(), block_k.end(), 0);
        } else {
             std::vector<std::size_t> block_k_non_symmetrized = sample_k_indices_without_replacement(latent_probabilities_k_0, cvar_alpha, BLOCK_SIZE, tau, rng);
            // float delta_k = 0.0;
            // for (auto &[bits, original_probability, latent_probability] : latent_probabilities_k_0){
            //     delta_k += (original_probability / (1 - cvar_alpha)) * sigmoid(latent_probability);
            // }

            if (!RUN_SYMMETRIC) {
                block_k = block_k_non_symmetrized;
            } else {
                block_k = std::vector<size_t>(2 * BLOCK_SIZE);
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    block_k[2*i] = block_k_non_symmetrized[i];
                    // note that this adds replacement! That's a problem, if we're not careful.
                    block_k[2*i + 1] = symmetric_index_vector[block_k_non_symmetrized[i]];   
                }
            }
        }

        if (myid == 0) {
            std::cout << "BLOCK_SIZE = " << BLOCK_SIZE << std::endl;
            std::cout << "latent_probabilities_k_0.size() = " << latent_probabilities_k_0.size() << std::endl;
            std::cout << "block_k: ";
            for (auto idx : block_k) std::cout << idx << " ";
            std::cout << std::endl;
        }

        // add a "membership mask" for the block
        std::vector<bool> block_membership_mask(latent_probabilities_k_0.size(), false);
        for (std::size_t idx : block_k) {
            block_membership_mask[idx] = true;
        }

        if (myid == 0) {
            std::cout << "Block membership mask: ";
            for (size_t i = 0; i < block_membership_mask.size(); ++i) {
                std::cout << (block_membership_mask[i] ? "1" : "0") << " ";
            }
            std::cout << std::endl;
        }

        generate_distribution_from_latents(
            latent_probabilities_k_0,
            cvar_alpha,
            q_distribution
        );      

        /**
        Holds the running value ||G_\eta(x^{k, j}, q^{k, j+1})||_{L^2}.
        **/
        real_t g_eta_cache = std::numeric_limits<real_t>::max(); // reset the gradient holder for the new outer iteration. We will use this to calculate the GBB stepsize, but we won't actually need to calculate the stepsize until we are doing the update, which is after the inner loop.

        /**
         * Clamp psi
         */ 
        odens_latent.median(CLAMP_MIN_VECTOR, CLAMP_MAX_VECTOR);

        real_t material_volume = proj(odens_latent, target_volume, domain_volume, 1e-12, 25,
                                      heaviside_active, heaviside_eta_d, heaviside_beta); // last two are tol and max its
        if (0 == myid)
        {
            cout << "For clamping on outer step " << outer << ", got material " << material_volume << ". Expected Material " << target_volume << "\n";
        }

        for (size_t inner = 1; inner <= inner_loop_iterations; inner++)
        {
            const std::string inner_iter_tag =
                "(outer, inner)=(" + std::to_string(outer) + ", " + std::to_string(inner) +
                "), total_iterations=" + std::to_string(total_iterations);

            if (myid == 0) {
                std::cout << inner_iter_tag << " gamma=" << gamma << "\n";
            }

                        // TODO: Sample ahead of time. Makes it simpler. Fewer computes.
            std::vector<size_t> latent_probability_indices_to_sample;
            std::vector<float> latent_probability_indices_to_sample_weights;
            std::vector<bool> latent_probability_indices_to_sample_mask; // for debugging. Remove later.
            bool xi_in_empty = true;

            /**
             * Determine where we're sampling. If in the stochastic case, we can skip calculating unnecessary compliances.
             * */
            if (is_deterministic_step(outer, RUN_DETERMINISTIC_UNTIL, RUN_DETERMINISTIC_AFTER)) {
                // we need to calculate q_k_jp1 for this. So defer.
                latent_probability_indices_to_sample_mask.assign(latent_probabilities_k_0.size(), true);
            } else {

                // we can sample, and filter later.
                if (myid == 0) std::cout << inner_iter_tag << " Running stochastic sampling for gradient estimation.\n";

                latent_probability_indices_to_sample_mask.assign(latent_probabilities_k_0.size(), false);
                std::vector<real_t> weights_by_index(latent_probabilities_k_0.size(), 0); 

                int TOTAL_GRADIENTS;
                if (!RUN_SYMMETRIC) {
                    TOTAL_GRADIENTS = STOCHASTIC_GRADIENT_MINIBATCH_SIZE;
                } else {
                    TOTAL_GRADIENTS = 2 * STOCHASTIC_GRADIENT_MINIBATCH_SIZE;
                }

                for (int sample = 0; sample < STOCHASTIC_GRADIENT_MINIBATCH_SIZE; ++sample) {
                    size_t sampled_index = q_distribution(rng);

                    if (myid == 0) std::cout << inner_iter_tag << " Sample " << sample << ": sampled_index = " << sampled_index << std::endl;

                    bool is_in_mask = block_membership_mask[sampled_index];
                    if (myid == 0) std::cout << inner_iter_tag << " is_in_mask = " << is_in_mask << std::endl;

                    if (!is_in_mask) {
                        latent_probability_indices_to_sample_mask[sampled_index] = true;

                        if (!RUN_SYMMETRIC) {
                            weights_by_index[sampled_index] += 1.0 / STOCHASTIC_GRADIENT_MINIBATCH_SIZE;        
                        } else {
                            size_t symmetric_index = symmetric_index_vector[sampled_index];

                            latent_probability_indices_to_sample_mask[symmetric_index] = true;

                            weights_by_index[sampled_index] += 0.5 / STOCHASTIC_GRADIENT_MINIBATCH_SIZE;       
                            weights_by_index[symmetric_index] += 0.5 / STOCHASTIC_GRADIENT_MINIBATCH_SIZE;       
                        }


                        if (myid == 0) std::cout << inner_iter_tag << " Added weight for non-block member" << std::endl;
                    } else {
                        // real_t q_k_jp1_over_q_k_j = sigmoid(latent_probability_jp1) / sigmoid(latent_probability_j);
                        // weights_by_index[sampled_index] += q_k_jp1_over_q_k_j / STOCHASTIC_GRADIENT_MINIBATCH_SIZE;
                        latent_probability_indices_to_sample_mask[sampled_index] = true;

                        if (!RUN_SYMMETRIC) {
                            weights_by_index[sampled_index] += 1.0 / STOCHASTIC_GRADIENT_MINIBATCH_SIZE;        
                        } else {
                            size_t symmetric_index = symmetric_index_vector[sampled_index];

                            latent_probability_indices_to_sample_mask[symmetric_index] = true;

                            weights_by_index[sampled_index] += 0.5 / STOCHASTIC_GRADIENT_MINIBATCH_SIZE;       
                            weights_by_index[symmetric_index] += 0.5 / STOCHASTIC_GRADIENT_MINIBATCH_SIZE;       
                        }

                        if (myid == 0) std::cout << inner_iter_tag << " Added weight for block member" << std::endl;
                    }
                }

                latent_probability_indices_to_sample.reserve(TOTAL_GRADIENTS);
                latent_probability_indices_to_sample_weights.reserve(TOTAL_GRADIENTS);

                for (size_t i = 0; i < weights_by_index.size(); ++i) {
                    if (weights_by_index[i] > 0.0) {
                        latent_probability_indices_to_sample.push_back(i);
                        latent_probability_indices_to_sample_weights.push_back(weights_by_index[i]);
                    }
                }

                if (myid == 0) {
                    std::cout << inner_iter_tag << " Sampled indices for gradient estimation: ";
                    for (size_t i = 0; i < latent_probability_indices_to_sample.size(); ++i) {
                        std::cout << latent_probability_indices_to_sample[i] << " (weight: " << latent_probability_indices_to_sample_weights[i] << ") ";
                    }
                    std::cout << std::endl;
                }
            }

            /****** 
                Step 1: The prox update. 
                At this point, we have latent_probabilities_k_0 = latent_probabilities_k_j by default. 
            ******/


            latent_probabilities_k_jp1_unnormalized.clear();
            latent_probabilities_k_jp1.clear(); // fresh start in inner loop

            std::vector<real_t> base_compliances_by_index;
            base_compliances_by_index.reserve(latent_probabilities_k_0.size());

            // iterate through the probability space. Update for the whole block (needs F values)
            for (size_t latent_probability_index = 0; latent_probability_index < latent_probabilities_k_0.size(); ++latent_probability_index)
            {
                // we still need to populate compliances, even if we do not use blocl
                auto &[bits, original_probability, latent_probability] = latent_probabilities_k_0[latent_probability_index];
                
                // neither in the block or a sample. Might as well ignore.
                if (!block_membership_mask[latent_probability_index] && !latent_probability_indices_to_sample_mask[latent_probability_index]) {
                    latent_probabilities_k_jp1_unnormalized.emplace_back(latent_probabilities_k_0[latent_probability_index]);
                    base_compliances_by_index.emplace_back(std::numeric_limits<real_t>::max());
                    continue;
                }

                if (0 == myid)
                {
                    std::cout << inner_iter_tag << " For proximal probability, going through "
                            << original_probability
                            << " probability element: "
                            << bits.to_string()
                            << ". Current latent probability is "
                            << latent_probability
                            << ".\n";
                }

                real_t compliance_i = evaluate_compliance_for_scenario(
                    bits, inner_iter_tag, "prox_update_base_compliance", true);
                if (myid == 0) { total_function_evaluations++; }

                base_compliances_by_index.emplace_back(compliance_i);

                if (myid == 0) {
                    std::cout << inner_iter_tag << " Latent: " << latent_probability << ", gamma: " << gamma << ", compliance: " << compliance_i << ".\n";
                    // std::cout << "Latent: " << latent_probability << ", alpha_stepsize: " << alpha_stepsize << ", compliance: " << compliance_i << ".\n";
                }

                if (block_membership_mask[latent_probability_index]) {
                    latent_probabilities_k_jp1_unnormalized.emplace_back(
                        bits, original_probability, latent_probability + gamma * compliance_i);
                } else {
                    latent_probabilities_k_jp1_unnormalized.emplace_back(latent_probabilities_k_0[latent_probability_index]);
                }
            }

            if (myid == 0) {
                std::cout << inner_iter_tag << " Printing latent_probabilities_k_jp1_unnormalized within inner loop, after calculating.\n";
                for (size_t i = 0; i < latent_probabilities_k_jp1_unnormalized.size(); ++i) {
                    std::cout << i << ":" << std::get<2>(latent_probabilities_k_jp1_unnormalized[i]) << " ";
                }
                std::cout << "\n";
                std::cout << inner_iter_tag << " Done printing latent_probabilities_k_jp1_unnormalized.\n";
            }

            // then we update the new latents to ensure they sum to 1. t is the number such that 
            // q^{k, j+1} = sigmoid(nabla phi_i(q^k_i) + gamma(F(x^{k, j} - t)).
            real_t t_normalizer = proj_latent_onto_probability_simplex(
                latent_probabilities_k_jp1_unnormalized,
                latent_probabilities_k_jp1,
                cvar_alpha,
                1e-10,
                25
            ) / gamma;

            for (auto &[bits, original_probability, latent_probability] : latent_probabilities_k_jp1)
            {
                if (myid == 0) {
                    const real_t q_probability = sigmoid(latent_probability) * original_probability / (1 - cvar_alpha);
                    std::cout << inner_iter_tag << " case: " << bits.to_string()
                              << " p=" << original_probability
                              << " l=" << latent_probability
                              << " q_new=" << q_probability << std::endl;
                }
            }

            if (myid == 0)
            {
                real_t max_latent_symmetry_gap = 0.0;
                real_t max_dual_probability_symmetry_gap = 0.0;
                real_t mean_latent_symmetry_gap = 0.0;
                real_t mean_dual_probability_symmetry_gap = 0.0;
                int counted_pairs = 0;

                for (size_t i = 0; i < latent_probabilities_k_jp1.size(); ++i)
                {
                    const size_t j = symmetric_index_vector[i];
                    if (i > j)
                    {
                        continue;
                    }

                    const auto &[bits_i, original_probability_i, latent_probability_i] = latent_probabilities_k_jp1[i];
                    const auto &[bits_j, original_probability_j, latent_probability_j] = latent_probabilities_k_jp1[j];

                    const real_t latent_gap = std::abs(latent_probability_i - latent_probability_j);
                    const real_t dual_probability_i = dual_probability_from_latent(original_probability_i, latent_probability_i);
                    const real_t dual_probability_j = dual_probability_from_latent(original_probability_j, latent_probability_j);
                    const real_t dual_probability_gap = std::abs(dual_probability_i - dual_probability_j);

                    max_latent_symmetry_gap = std::max(max_latent_symmetry_gap, latent_gap);
                    max_dual_probability_symmetry_gap = std::max(max_dual_probability_symmetry_gap, dual_probability_gap);
                    mean_latent_symmetry_gap += latent_gap;
                    mean_dual_probability_symmetry_gap += dual_probability_gap;
                    counted_pairs++;
                }

                if (counted_pairs > 0)
                {
                    mean_latent_symmetry_gap /= static_cast<real_t>(counted_pairs);
                    mean_dual_probability_symmetry_gap /= static_cast<real_t>(counted_pairs);
                }

                std::cout << inner_iter_tag
                          << " [symmetry-diagnostic] paired_cases=" << counted_pairs
                          << " max_latent_gap=" << max_latent_symmetry_gap
                          << " mean_latent_gap=" << mean_latent_symmetry_gap
                          << " max_dual_prob_gap=" << max_dual_probability_symmetry_gap
                          << " mean_dual_prob_gap=" << mean_dual_probability_symmetry_gap
                          << std::endl;
            }

            if (myid == 0)
            {
                for (size_t scenario_index = 0; scenario_index < latent_probabilities_k_jp1.size(); ++scenario_index)
                {
                    const auto &[bits, original_probability, latent_probability] = latent_probabilities_k_jp1[scenario_index];
                    const real_t dual_probability =
                        sigmoid(latent_probability) * original_probability / (1 - cvar_alpha);

                    dual_probabilities_csv << outer << ","
                                           << inner << ","
                                           << total_iterations << ","
                                           << scenario_index << ","
                                           << bits.to_string() << ","
                                           << original_probability << ","
                                           << latent_probability << ","
                                           << dual_probability << "\n";
                }
                dual_probabilities_csv.flush();
            }

            /**
            Now we are done with the q prox update! 
            **/

            // Now we make it sample-able. Also store this (minor cost, but we should make more opportunistic for best-practice.)   
            
                        /****
            Deterime I_j,  the set of indices we will sample gradients from.
            ****/

            if (is_deterministic_step(outer, RUN_DETERMINISTIC_UNTIL, RUN_DETERMINISTIC_AFTER)) {
                if (myid == 0) std::cout << inner_iter_tag << " Running deterministic sampling for gradient estimation.\n";

                /***
                In the deterministic case, we simply remove the q_i with numerically trivial probabilities
                ***/
                latent_probability_indices_to_sample.reserve(latent_probabilities_k_jp1.size());
                latent_probability_indices_to_sample_weights.reserve(latent_probabilities_k_jp1.size());

                for (size_t i = 0; i < latent_probabilities_k_jp1.size(); ++i) {

                    auto &[bits, original_probability, latent_probability] = latent_probabilities_k_jp1[i];
                    real_t probability = dual_probability_from_latent(original_probability, latent_probability);

                    if (probability > epsilon_q) {
                        latent_probability_indices_to_sample.push_back(i);
                        latent_probability_indices_to_sample_weights.push_back(probability);
                        latent_probability_indices_to_sample_mask[i] = true;
                    }
                }
            } else {
                if (myid == 0) std::cout << inner_iter_tag << " Filtering post-stochastic sampling.\n";

                std::vector<size_t> kept_indices;
                std::vector<float> kept_weights;
                kept_indices.reserve(latent_probability_indices_to_sample.size());
                kept_weights.reserve(latent_probability_indices_to_sample_weights.size());

                for (size_t SAMPLE_INDEX = 0; SAMPLE_INDEX < latent_probability_indices_to_sample.size(); ++SAMPLE_INDEX) {
                    const size_t sample = latent_probability_indices_to_sample[SAMPLE_INDEX];
                    const float sample_weight = latent_probability_indices_to_sample_weights[SAMPLE_INDEX];
                    bool keep_sample = true;

                    if (block_membership_mask[sample]) {
                        const auto [bits_j, original_probability_j, latent_probability_j] = latent_probabilities_k_j[sample];
                        const auto [bits_jp1, original_probability_jp1, latent_probability_jp1] = latent_probabilities_k_jp1[sample];

                        const real_t q_k_jp1 = dual_probability_from_latent(original_probability_jp1, latent_probability_jp1);
                        if (myid == 0) std::cout << inner_iter_tag << " q_k_jp1 = " << q_k_jp1 << ", epsilon_q = " << epsilon_q << std::endl;

                        if (q_k_jp1 < epsilon_q) {
                            if (myid == 0) std::cout << inner_iter_tag << " Dropping sample " << sample << " because q_k_jp1 < epsilon_q" << std::endl;
                            latent_probability_indices_to_sample_mask[sample] = false;
                            keep_sample = false;
                        } else {
                            xi_in_empty = false; // now we know that there is a nontrivial weight.
                        }
                    }

                    if (keep_sample) {
                        kept_indices.push_back(sample);
                        kept_weights.push_back(sample_weight);
                    }
                }

                latent_probability_indices_to_sample.swap(kept_indices);
                latent_probability_indices_to_sample_weights.swap(kept_weights);
            }

            /**
            Iterate through the indices we care about. Assemble our weighted gradient. Keep track of gradient magnitudes.
            **/

            real_t x_j_iteration_max_gradient = -1.0;

            // calculate the weighted gradient 
            odens_weighted_gradient = 0.0;
            for (size_t i = 0; i < latent_probability_indices_to_sample.size(); i++)
            {          
                Index key = latent_probability_indices_to_sample[i];

                const auto [bits, original_probability, latent_probability] = latent_probabilities_k_jp1[key];

                if (is_deterministic_step(outer, RUN_DETERMINISTIC_UNTIL, RUN_DETERMINISTIC_AFTER))
                {
                    const real_t dual_probability = dual_probability_from_latent(original_probability, latent_probability);
                    if (dual_probability < epsilon_q)
                    {
                        if (myid == 0)
                        {
                            std::cout << inner_iter_tag
                                      << " Skipping gradient eval for scenario " << bits.to_string()
                                      << " because dual_probability=" << dual_probability
                                      << " < epsilon_q=" << epsilon_q
                                      << std::endl;
                        }
                        continue;
                    }
                }

                if (0 == myid)
                {
                    std::cout << inner_iter_tag << " For Gradient, going through "
                            << original_probability
                            << " probability element: "
                            << bits.to_string()
                            << ". Current latent probability is "
                            << latent_probability
                            << ".\n";
                }

                // set up the boundary conditions
                // set the boundary conditions [1,2,..,7]
                elsolver->DelDispBC();
                for (int i = 0; i < N; i++)
                {
                    if (!(bits.test(i)))
                    {
                        elsolver->AddDispBC(bit_index_to_mesh_bc_index[i], 4, 0.0); // start with all of them fixed. 0 displacement in all directions.
                    }
                }

                // solve the discrete elastic system
                elsolver->Assemble();
                elsolver->FSolve(); // solve for displacements
                if (myid == 0) { total_gradient_evaluations++; }

                // extract the solution
                ParGridFunction &sol = elsolver->GetDisplacements();
                // if (myid == 0) {
                //     std::cout << "Displacement norm: " << sol.ComputeL2Error(zero) << std::endl;
                // }
                icc.SetGridFunctions(&fdens, &sol);

                real_t grad_norm_after_afilter = 0.0;
                real_t grad_norm_after_smooth = 0.0;
                const bool gradient_ok = compute_smoothed_design_gradient(
                    *filt, icc, Minv, odens_gradient_single, tmp_grad,
                    grad_norm_after_afilter, grad_norm_after_smooth);

                if (myid == 0) {
                    std::cout << inner_iter_tag << " After AFilter, norm odens_gradient_single: " << grad_norm_after_afilter << std::endl;
                }
                if (!std::isfinite(grad_norm_after_afilter)) {
                    if (myid == 0) std::cout << inner_iter_tag << " NaN detected after AFilter for scenario " << bits.to_string() << std::endl;
                    odens_gradient_single = 0.0; // skip this gradient
                    continue;
                }
                if (myid == 0) {
                    std::cout << inner_iter_tag << " After smoothing, norm odens_gradient_single: " << grad_norm_after_smooth << std::endl;
                }
                if (!gradient_ok || !std::isfinite(grad_norm_after_smooth)) {
                    if (myid == 0) std::cout << inner_iter_tag << " NaN detected after smoothing for scenario " << bits.to_string() << std::endl;
                    odens_gradient_single = 0.0; // skip this gradient
                    continue;
                }
                // ogf.GetTrueDofs(ograd); // extracting the information in the ParGridFunction into the ograd. Lives in dual of design space.

                real_t weight = latent_probability_indices_to_sample_weights[i];
                if (!is_deterministic_step(outer, RUN_DETERMINISTIC_UNTIL, RUN_DETERMINISTIC_AFTER) && block_membership_mask[key]) {
                    const auto [bits_0, original_probability_0, latent_probability_0] = latent_probabilities_k_0[key];

                    // initial sampling was from q^k so we have to resample to have the right weight
                    weight = weight * sigmoid(latent_probability) / sigmoid(latent_probability_0); 
                }

                if (is_deterministic_step(outer, RUN_DETERMINISTIC_UNTIL, RUN_DETERMINISTIC_AFTER) || block_membership_mask[key] || xi_in_empty)
                {
                    real_t gradient_magnitude = compute_global_l2_norm(odens_gradient_single);

                    x_j_iteration_max_gradient = std::max(x_j_iteration_max_gradient, gradient_magnitude);
                }

                odens_weighted_gradient.Add(weight, odens_gradient_single); // we will use this for the GBB stepsize calculation after the loop. Note that the weight is already taking into account the ratio q_k_jp1 / q_k_j for the stochastic case, and is just the probability for the deterministic case.
                // odens_latent.Add(-inner_loop_gamma * weight, odens_gradient_single);
                // if (compute_gradient) {
                //     odens_gradient_holder.Add(normalizing_factor, odens_gradient_single);
                // }
            }

            real_t total_gradient = odens_weighted_gradient.ComputeL2Error(zero);
            if (myid == 0) {
                std::cout << inner_iter_tag << " Total gradient norm: " << total_gradient << std::endl;
            }


                    
            // We are running a (stochastic) armijo backtracking line search here, so we need a baseline F
            // We use the same indices and weights as with the gradient approximation
            // We ignore the factor of t, here. No need.
            // Note we are optimizing for fixed q^k
            //


            real_t F_x = 0.0;
            for (size_t INDEX_IN_SAMPLER = 0; INDEX_IN_SAMPLER < latent_probability_indices_to_sample.size(); ++INDEX_IN_SAMPLER) {

                Index key = latent_probability_indices_to_sample[INDEX_IN_SAMPLER];
                real_t weight = latent_probability_indices_to_sample_weights[INDEX_IN_SAMPLER];
                real_t compliance = base_compliances_by_index[key];
                if (!std::isfinite(compliance))
                {
                    const auto &[bits_selected, original_probability_selected, latent_probability_selected] = latent_probabilities_k_jp1[key];
                    compliance = evaluate_compliance_for_scenario(
                        bits_selected,
                        inner_iter_tag,
                        "baseline_function_late_compliance_for_selected_index",
                        true);
                    if (myid == 0) { total_function_evaluations++; }
                    base_compliances_by_index[key] = compliance;
                }
                if (block_membership_mask[key]) {

                    // what we want to calculate is p_i/(1-alpha) h_i(x, t; r)
                    // with h_i(x, t; r) = ln(1 + e^{\nabla_i \psi(latent_old_i) + gamma * (F_i(x) - t)})
                    // We consider this to be a gradient step in this function, fixing all our variables, fixing t.
                    // Now, We sample per q^{k, j+1} = (p_i / (1 - alpha)) * sigma(latent_k_jp1[i])
                    // So a little division has to happen. By q^{k, j+1}. But, since we have access to latent variables,
                    // we can try to use the more-stable multiplication by (e^{-latent_i} + 1)

                    auto &[bits_0, original_probability_0, latent_probability_0] = latent_probabilities_k_0[key];
                    real_t nabla_phi_plus_f = latent_probability_0 + gamma * (compliance - t_normalizer); // can be clever and use the latent k_jp1, but we like standardization
                    
                    // to avoid explosion in the soft-max
                    real_t h_i;
                    if (nabla_phi_plus_f < 0) {
                        h_i = std::log(1.0 + std::exp(nabla_phi_plus_f));
                    } else {
                        h_i = nabla_phi_plus_f + std::log(std::exp(-nabla_phi_plus_f) + 1.0);
                    }
                    
                    real_t summand = h_i * (1 + std::exp(-latent_probability_0)) / gamma; // this is the same as dividing by q^{k, j+1}, but more stable.
                    if (myid == 0) {
                        std::cout << inner_iter_tag << " BASE Block member " << key << ": compliance=" << compliance << ", nabla_phi_plus_f=" << nabla_phi_plus_f
                                  << ", h_i=" << h_i << ", summand=" << summand << ", weight=" << weight << std::endl;
                    }
                    F_x += weight * summand;
                } else {
                    F_x += weight * compliance;
                    if (myid == 0) {
                        std::cout << inner_iter_tag << " BASE Non-Block member " << key << ": compliance=" << compliance << ", weight=" << weight << std::endl;
                    }
                }
            }

            // stepsize for armijo backtracking line search. 
            // We treat each inner loop via armijo backtracking line search.
            // Note that q^{k, j+1} is a function of x. 

            if (inner > 1 || alpha_stepsize > 0) {
                // generalized BB stepsize calculation.

                SumCoefficient odens_difference_coeff(odens, odens_old, 1.0, -1.0);
                // add(odens, -1.0, odens_old, odens_difference);
                odens_difference.ProjectCoefficient(odens_difference_coeff);

                // GridFunctionCoefficient latent_coeff(&odens_latent);
                // GridFunctionCoefficient latent_old_coeff(&odens_latent_old);
                // SumCoefficient odens_latent_difference_coeff(latent_coeff, latent_old_coeff, 1.0, -1.0);
                // odens_latent_difference.ProjectCoefficient(odens_latent_difference_coeff);

                add(odens_latent, -1.0, odens_latent_old, odens_latent_difference);

                GridFunctionCoefficient grad_coeff(&odens_weighted_gradient);
                GridFunctionCoefficient grad_old_coeff(&odens_weighted_gradient_old);
                SumCoefficient odens_gradient_difference_coeff(grad_coeff, grad_old_coeff, 1.0, -1.0);
                odens_gradient_difference.ProjectCoefficient(odens_gradient_difference_coeff);


                // Compute L2 norms of the differences (for ?debugging / sanity checks)
                ConstantCoefficient zero_c(0.0);
                real_t dx_norm = odens_difference.ComputeL2Error(zero_c);
                real_t dpsi_norm = odens_latent_difference.ComputeL2Error(zero_c);
                real_t dg_norm = odens_gradient_difference.ComputeL2Error(zero_c);

                // real_t dx_norm_sq = global_dx_norm * global_dx_norm;
                // real_t dpsi_norm_sq = global_dpsi_norm * global_dpsi_norm;
                // real_t dg_norm_sq = global_dg_norm * global_dg_norm;



                // /* TEMPORARY: Error for norms. */

                // odens_gf.ProjectCoefficient(odens);
                // odens_old_gf.ProjectCoefficient(odens_old);

                // real_t local_odens_norm = odens_gf.ComputeL2Error(zero_c);
                // real_t local_dpsi_norm = odens_latent_difference.ComputeL2Error(zero_c);
                // real_t local_dg_norm = odens_gradient_difference.ComputeL2Error(zero_c);

                // real_t dx_norm_sq = local_dx_norm * local_dx_norm;
                // real_t dpsi_norm_sq = local_dpsi_norm * local_dpsi_norm;
                // real_t dg_norm_sq = local_dg_norm * local_dg_norm;

                // /* END Temporary */


                // real_t global_dx_norm_sq, global_dpsi_norm_sq, global_dg_norm_sq;
                // MPI_Allreduce(&dx_norm_sq, &global_dx_norm_sq, 1, MPI_DOUBLE, MPI_SUM, filt->GetDesignFES()->GetComm());
                // MPI_Allreduce(&dpsi_norm_sq, &global_dpsi_norm_sq, 1, MPI_DOUBLE, MPI_SUM, filt->GetDesignFES()->GetComm());
                // MPI_Allreduce(&dg_norm_sq, &global_dg_norm_sq, 1, MPI_DOUBLE, MPI_SUM, filt->GetDesignFES()->GetComm());

                // real_t dx_norm = std::sqrt(global_dx_norm_sq);
                // real_t dpsi_norm = std::sqrt(global_dpsi_norm_sq);
                // real_t dg_norm = std::sqrt(global_dg_norm_sq);

                // Compute inner products via a linear form (consistent with how we compute gradient norms)
                ParGridFunction one_design(filt->GetDesignFES());
                one_design = 1.0;

                GridFunctionCoefficient dx_coeff(&odens_difference);
                GridFunctionCoefficient dpsi_coeff(&odens_latent_difference);
                GridFunctionCoefficient dg_coeff(&odens_gradient_difference);

                ProductCoefficient dx_dpsi_coeff(dx_coeff, dpsi_coeff);
                ProductCoefficient dx_dg_coeff(dx_coeff, dg_coeff);

                // int delta g * delta rho
                // int delta psi * delta rho
                // -> ell(v): L2 -> R such that ell(v) = int delta rho * v dx
                // -> ell(delta psi) / ell(delta g)

                ParLinearForm ip_dx_dpsi_form(filt->GetDesignFES());
                ip_dx_dpsi_form.AddDomainIntegrator(new DomainLFIntegrator(dx_dpsi_coeff));
                ip_dx_dpsi_form.Assemble();
                real_t ip_xp_global = ip_dx_dpsi_form(one_design);


                ParLinearForm ip_dx_dg_form(filt->GetDesignFES());
                ip_dx_dg_form.AddDomainIntegrator(new DomainLFIntegrator(dx_dg_coeff));
                ip_dx_dg_form.Assemble();
                real_t ip_xg_global = ip_dx_dg_form(one_design);
                // cout << myid << ", " << ip_xp_global << ", " << ip_xg_global << std::endl;

                real_t gbb_ratio = std::abs(ip_xp_global / std::max(std::abs(ip_xg_global), (real_t)1e-10));

                if (myid == 0) {
                    std::cout << inner_iter_tag << " BB inner-product diagnostics:\n";
                    std::cout << "  ||dx||_2 = " << dx_norm << ", ||dpsi||_2 = " << dpsi_norm << ", ||dg||_2 = " << dg_norm << "\n";
                    std::cout << "  <dx, dpsi> = " << ip_xp_global << ", <dx, dg> = " << ip_xg_global << "\n";
                }

                alpha_stepsize = std::min(std::max(std::sqrt(alpha_stepsize * gbb_ratio / ALPHA_SCALING_FACTOR_OVERRIDE), MIN_ALPHA), MAX_ALPHA);
            } else {
                real_t gradient_max = ParNormlp(odens_weighted_gradient, mfem::infinity(), odens_weighted_gradient.ParFESpace()->GetComm());;
                if (myid == 0) cout << inner_iter_tag << " Initial gradient max: " << gradient_max << std::endl;
                // {
                    // gradient_max
                    // real_t locmax = odens_weighted_gradient.Normlinf();
                    // MPI_DOUBLE should be replaced with the MFEM data type
                    // MPI_Allreduce(&locmax, &gradient_max, 1, MPI_DOUBLE, MPI_MAX, odens_weighted_gradient.ParFESpace()->GetComm());
                // }
                alpha_stepsize = std::min(std::max(starting_alpha_stepsize / gradient_max , MIN_ALPHA), MAX_ALPHA); // scale with alpha
            }
            alpha_stepsize = alpha_stepsize * ALPHA_SCALING_FACTOR_OVERRIDE;

            // Initialize "old" state before entering the Armijo backtracking loop.
            // This ensures the first backtracking attempt has a meaningful dx/dpsi/dg
            // when computing inner products and norms for diagnostics.

            
            odens_latent_old = odens_latent;
            // odens_weighted_gradient_old = odens_weighted_gradient;
            // Note: odens_old is a mapped coefficient based on odens_latent_old,
            // so setting odens_latent_old is sufficient to make odens_old match odens.

            int armijo_descents = 0;
            for (size_t prox_iter_attempts = 0; prox_iter_attempts < MAX_BACKTRACKING_ATTEMPTS; ++prox_iter_attempts) {
                if (prox_iter_attempts > 0) {
                    odens_latent = odens_latent_old; // reset to old before trying the next stepsize
                }

                // implement the stepsize
                odens_latent.Add(-alpha_stepsize, odens_weighted_gradient); // this is the "update step" for x. We will check if this satisfies the armijo condition. If not, we will reduce the stepsize and try again.

                // update the design field to match the new latent variables.
                // Note this effects upstairs as well, in the prox step
                // filt->FFilter(&odens, fdens);
                // odens_gf.ProjectCoefficient(odens);

                // project our design onto the one with the proper volume
                real_t material_volume = proj(odens_latent, target_volume, domain_volume, 1e-12, 25,
                                              heaviside_active, heaviside_eta_d, heaviside_beta); // last two are tol and max its
                if (0 == myid)
                {
                    cout << inner_iter_tag << ", prox_iter=" << prox_iter_attempts
                         << " On inner step " << prox_iter_attempts << ", got material " << material_volume
                         << ". Expected Material " << target_volume << "\n";
                }

                // update fdens after projection
                filt->FFilter(&odens, fdens);
                // odens_gf.ProjectCoefficient(odens);

                // now we calculate compliances and find F_x the same way

                real_t F_x_jp1 = 0.0;
                for (size_t INDEX_IN_SAMPLER = 0; INDEX_IN_SAMPLER < latent_probability_indices_to_sample.size(); ++INDEX_IN_SAMPLER) {

                    Index key = latent_probability_indices_to_sample[INDEX_IN_SAMPLER];
                    real_t weight = latent_probability_indices_to_sample_weights[INDEX_IN_SAMPLER];
                    auto &[bits, original_probability, latent_probability] = latent_probabilities_k_0[key];

                    if (is_deterministic_step(outer, RUN_DETERMINISTIC_UNTIL, RUN_DETERMINISTIC_AFTER))
                    {
                        const auto &[bits_jp1, original_probability_jp1, latent_probability_jp1] = latent_probabilities_k_jp1[key];
                        const real_t dual_probability = dual_probability_from_latent(original_probability_jp1, latent_probability_jp1);
                        if (dual_probability < epsilon_q)
                        {
                            if (myid == 0)
                            {
                                std::cout << inner_iter_tag << ", prox_iter=" << prox_iter_attempts
                                          << " Skipping Armijo function eval for scenario " << bits_jp1.to_string()
                                          << " because dual_probability=" << dual_probability
                                          << " < epsilon_q=" << epsilon_q
                                          << std::endl;
                            }
                            continue;
                        }
                    }

                    // calculate compliance.
                    const std::string armijo_eval_tag =
                        inner_iter_tag + ", prox_iter=" + std::to_string(prox_iter_attempts);
                    real_t compliance_i_armijo_test = evaluate_compliance_for_scenario(
                        bits, armijo_eval_tag, "armijo_backtracking_function_test", true);
                    if (myid == 0) { total_function_evaluations++; }

                    if (block_membership_mask[key]) {

                        auto &[bits_0, original_probability_0, latent_probability_0] = latent_probabilities_k_0[key];

                        real_t nabla_phi_plus_f = latent_probability_0 + gamma * (compliance_i_armijo_test - t_normalizer); // can be clever and use the latent k_jp1, but we like standardization
                        
                        real_t h_i;
                        if (nabla_phi_plus_f < 0) {
                            h_i = std::log(1.0 + std::exp(nabla_phi_plus_f));
                        } else {
                            h_i = nabla_phi_plus_f + std::log(std::exp(-nabla_phi_plus_f) + 1.0);
                        }

                        real_t summand = h_i * (1 + std::exp(-latent_probability_0)) / gamma; // this is the same as dividing by q^{k, j+1}, but more stable.
                        if (myid == 0) {
                            std::cout << inner_iter_tag << ", prox_iter=" << prox_iter_attempts
                                      << " Armijo Block member " << key << ": compliance=" << compliance_i_armijo_test << ", nabla_phi_plus_f=" << nabla_phi_plus_f
                                      << ", h_i=" << h_i << ", summand=" << summand << ", weight=" << weight << std::endl;
                        }
                        F_x_jp1 += weight * summand;
                    } else {
                        F_x_jp1 += weight * compliance_i_armijo_test;
                        if (myid == 0) {
                            std::cout << inner_iter_tag << ", prox_iter=" << prox_iter_attempts
                                      << " Armijo Non-Block member " << key << ": compliance=" << compliance_i_armijo_test
                                      << ", weight=" << weight << std::endl;
                        }
                    }
                }
                /**
                    Save the current state into the "old" variables so we can modify odens_latent.
                    odens_latent_old is x_k, odens_latent is x_kp1
                **/

                // Now we run the test. Want F(x_new) <= F(x) - c1 * Gradient * (odens_new - odens_old)
                real_t gradient_inner_product;
                {
                    SumCoefficient rho_diff_coeff(odens, odens_old, 1.0, -1.0);


                    ParLinearForm lf_norm(filt->GetDesignFES());
                    lf_norm.AddDomainIntegrator(new DomainLFIntegrator(rho_diff_coeff));
                    lf_norm.Assemble();

                    gradient_inner_product = lf_norm(odens_weighted_gradient);

                    if (myid == 0) {
                        std::cout << inner_iter_tag << ", prox_iter=" << prox_iter_attempts
                                  << " Gradient inner product: " << gradient_inner_product << std::endl;
                    }
                }

                if (myid == 0) {
                    std::cout << inner_iter_tag << ", prox_iter=" << prox_iter_attempts << " Armijo Test:\n";
                    std::cout << "F(x): " << F_x << "\n";
                    std::cout << "F(x_new): " << F_x_jp1 << "\n";
                    std::cout << "Gradient Inner Product: " << gradient_inner_product << "\n";
                    std::cout << "Alpha Stepsize: " << alpha_stepsize << "\n";
                }

                // Now, run the armijo test
                if (F_x_jp1 <= F_x + ARMIJO_C1 * gradient_inner_product) {
                    if (myid == 0) {
                        std::cout << inner_iter_tag << ", prox_iter=" << prox_iter_attempts << " Armijo condition met.\n";
                    }
                    break; // armijo condition satisfied, break out of backtracking loop
                } else {
                    if (myid == 0) {
                        std::cout << inner_iter_tag << ", prox_iter=" << prox_iter_attempts
                                  << " Armijo condition NOT met. Reducing stepsize.\n";
                    }
                    armijo_descents++;
                    alpha_stepsize *= ARMIJO_BETA; // reduce stepsize and try again
                }
            }

            // we keep the primal iterate when we break either way, so might as well project now.
            {
                total_iterations += 1;
                odens_gf.ProjectCoefficient(odens);
                hbeta_fdens_e.ProjectCoefficient(post_heaviside_coeff_e);
                hbeta_fdens_d.ProjectCoefficient(post_heaviside_coeff_d);
                paraview_dc.SetCycle(total_iterations);
                paraview_dc.SetTime((real_t)total_iterations);
                paraview_dc.Save();
            }

            // Extra CVaR reporting diagnostics.
            // These additional compliance evaluations are intentionally excluded from
            // total_function_evaluations and total_gradient_evaluations.
            std::vector<std::tuple<real_t, real_t, real_t>> scenario_reporting_data;
            scenario_reporting_data.reserve(latent_probabilities_k_jp1.size());
            for (size_t i = 0; i < latent_probabilities_k_jp1.size(); ++i)
            {
                const auto &[bits, original_probability, latent_probability] = latent_probabilities_k_jp1[i];
                const real_t compliance_i = evaluate_compliance_for_scenario(
                    bits,
                    inner_iter_tag,
                    "reporting_all_compliances_for_true_base_cvar_not_counted",
                    false);

                scenario_reporting_data.emplace_back(
                    compliance_i,
                    original_probability,
                    latent_probability);
            }

            real_t current_cvar_estimation = 0.0;
            for (const auto &[compliance_i, original_probability_i, latent_probability_i] : scenario_reporting_data)
            {
                const real_t q_probability_i =
                    sigmoid(latent_probability_i) * original_probability_i / (1 - cvar_alpha);
                current_cvar_estimation += q_probability_i * compliance_i;
            }

            std::vector<std::pair<real_t, real_t>> compliance_probability_pairs;
            compliance_probability_pairs.reserve(scenario_reporting_data.size());
            for (const auto &[compliance_i, original_probability_i, latent_probability_i] : scenario_reporting_data)
            {
                compliance_probability_pairs.emplace_back(compliance_i, original_probability_i);
            }

            std::sort(
                compliance_probability_pairs.begin(),
                compliance_probability_pairs.end(),
                [](const std::pair<real_t, real_t> &a, const std::pair<real_t, real_t> &b)
                {
                    return a.first > b.first;
                });

            const real_t tail_mass_target = 1.0 - cvar_alpha;
            real_t remaining_tail_mass = tail_mass_target;
            real_t tail_weighted_compliance_sum = 0.0;

            for (const auto &[compliance_i, base_probability_i] : compliance_probability_pairs)
            {
                if (remaining_tail_mass <= 0.0)
                {
                    break;
                }

                const real_t tail_mass_taken = std::min(base_probability_i, remaining_tail_mass);
                tail_weighted_compliance_sum += tail_mass_taken * compliance_i;
                remaining_tail_mass -= tail_mass_taken;
            }

            const real_t current_cvar =
                (tail_mass_target > 0.0)
                    ? (tail_weighted_compliance_sum / tail_mass_target)
                    : std::numeric_limits<real_t>::quiet_NaN();

            if (myid == 0)
            {
                std::ostringstream csv_row;
                csv_row << outer << ","
                        << inner << ","
                        << total_iterations << ","
                        << total_gradient_evaluations << ","
                        << total_function_evaluations << ","
                        << armijo_descents << ","
                        << current_cvar << ","
                        << current_cvar_estimation << "\n";

                cvar_csv << csv_row.str();
                cvar_csv.flush();

                std::cout << inner_iter_tag << " [csv] " << csv_row.str();
            }

            // we note, this is still for DETERMINISTIC
            {
                if (myid == 0){
                    std::cout << inner_iter_tag << " Projecting gradient and checking termination conditions for inner loop." << std::endl;
                }
                
                // get the projected gradient
                
                
                projected_gradient = odens_latent_old;

                projected_gradient -= odens_latent;   

                // divide by step size to get true gradient
                projected_gradient *= 1.0 / alpha_stepsize;


                real_t projected_gradient_magnitude;
                {
                    // 1. Compute the local integral of (u^2)
                    // ComputeL2Error returns sqrt(integral). We need the square to sum it.
                    ConstantCoefficient zero(0.0);
                    real_t local_norm = projected_gradient.ComputeL2Error(zero);
                    real_t local_integral_sq = local_norm * local_norm;

                    // 2. Sum the integrals across all processors
                    real_t global_integral_sq;
                    MPI_Allreduce(&local_integral_sq, &global_integral_sq, 1, MPI_DOUBLE, MPI_SUM, projected_gradient.ParFESpace()->GetComm());

                    // 3. Final root to get the L2 norm
                    projected_gradient_magnitude = std::sqrt(global_integral_sq);
                }
                g_eta_cache = projected_gradient_magnitude;

                real_t divergence_from_q_k_0 = cvar_divergence_beween_distributions(latent_probabilities_k_j, latent_probabilities_k_0, cvar_alpha);

                real_t dj = rho * rho * divergence_from_q_k_0;

                real_t divergence_from_q_k_jp1 = cvar_divergence_beween_distributions(latent_probabilities_k_j, latent_probabilities_k_jp1, cvar_alpha);

                const real_t sqrt_dj = (dj >= 0.0) ? std::sqrt(dj) : std::numeric_limits<real_t>::quiet_NaN();
                const real_t gradient_threshold = x_j_iteration_max_gradient * sqrt_dj;
                const bool gradient_condition_met = (projected_gradient_magnitude <= gradient_threshold);
                const bool divergence_condition_met = (divergence_from_q_k_jp1 <= dj);
                const bool inner_termination_met = gradient_condition_met && divergence_condition_met;

                // doing this at the last moment. 
                odens_weighted_gradient_old = odens_weighted_gradient;

                if (myid == 0) {
                    std::cout << inner_iter_tag << " Inner loop termination condition check:\n";
                    std::cout << "Projected Gradient Magnitude: " << projected_gradient_magnitude << "\n";
                    std::cout << "Gradient Threshold (x_j_max_grad * sqrt(dj)): " << gradient_threshold << "\n";
                    std::cout << "Gradient Criterion (g <= thresh): " << gradient_condition_met
                              << " [lhs-rhs=" << (projected_gradient_magnitude - gradient_threshold) << "]\n";
                    std::cout << "Divergence from Base Distribution: " << divergence_from_q_k_0 << "\n";
                    std::cout << "Divergence from Step Distribution: " << divergence_from_q_k_jp1 << "\n";
                    std::cout << "Calculated D_j value: " << dj << "\n";
                    std::cout << "Divergence Criterion (div_step <= dj): " << divergence_condition_met
                              << " [lhs-rhs=" << (divergence_from_q_k_jp1 - dj) << "]\n";
                    std::cout << "Combined Inner Termination Decision: " << inner_termination_met << "\n";
                    if (!std::isfinite(gradient_threshold))
                    {
                        std::cout << "WARNING: gradient_threshold is non-finite (likely dj < 0 or overflow)." << "\n";
                    }
                }

                if (inner_termination_met) {
                    if (myid == 0) {
                        std::cout << inner_iter_tag << " Termination conditions met for inner loop.\n";
                    }
                    break;
                } else if (myid == 0) {
                    std::cout << inner_iter_tag << " Termination conditions NOT met for inner loop.\n";
                }
            }

            // swapping ownership of data is easier than copying.
            std::swap(latent_probabilities_k_j, latent_probabilities_k_jp1);
            latent_probabilities_k_jp1_unnormalized.clear();
            latent_probabilities_k_jp1.clear(); // fresh start in inner loop
        }


        real_t t1_distance_qk0_qkj = 0.0;
        for (size_t i = 0; i < latent_probabilities_k_j.size(); ++i) {
            auto &[bits_j, original_probability_j, latent_probability_j] = latent_probabilities_k_j[i];
            auto &[bits_0, original_probability_0, latent_probability_0] = latent_probabilities_k_0[i];

            t1_distance_qk0_qkj += std::abs(sigmoid(latent_probability_0) - sigmoid(latent_probability_j)) * original_probability_j / (1 - cvar_alpha);
        }

        // for next iteration, we will start with the normalized version.
        latent_probabilities_k_0 = latent_probabilities_k_j;
        if (myid == 0) {
            std::cout << "Outer Loop Iteration " << outer << " Termination Condition Check:\n";
            std::cout << "Gradient Magnitude: " << g_eta_cache << "\n";
            std::cout << "T1 Distance between q_k_0 and q_k_j: " << t1_distance_qk0_qkj << "\n";
        }

        /**
        Test outer termination conditions. We note that, technically, g_eta_cache = ||G_\eta(x^{k, j}, q^{k, j+1})||.
        It add code complication (new gradient calcuation section, extra projections, etc) to compute ||G_eta(x^{k, j+1}, q^{k, j})||, as the original algorithm does.
        **/
        if (g_eta_cache < epsilon_g && 0.5 * t1_distance_qk0_qkj < epsilon_TV) {
            if (myid == 0) {
                std::cout << "Termination conditions met for outer loop at iteration " << outer << ".\n";
            }
            break;
        } else if (myid == 0) {
            std::cout << "Termination conditions NOT met for outer loop at iteration " << outer << ".\n";
        }


    }

    if (myid == 0)
    {
        cvar_csv.close();
        dual_probabilities_csv.close();
    }

    delete elsolver;
    delete filt;

    Mpi::Finalize();
    return 0;
}
