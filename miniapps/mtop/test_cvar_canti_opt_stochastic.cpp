#include "mfem.hpp"
#include "mtop_solvers.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <utility>
#include <bitset>
#include <random>
#include <chrono>

#include <stdexcept>

//@Will I would to copy only the functions you need
#include "ex37_copy.hpp"

using namespace std;
using namespace mfem;

/*
 *
 * Get the Failure probabilities or 0, 1, 2, or 3 contiguous holes failing.
 */

const int N = 6;

std::vector<std::pair<std::bitset<N>, real_t>> getProbabilitySpace(real_t p1, real_t p2, real_t p3, real_t p4)
{
    std::vector<std::pair<std::bitset<N>, real_t>> probability_space;

    if (p1 < 0 || p2 < 0 || p3 < 0 || p4 < 0 || (N * p1 + (N - 1) * p2 + (N - 2) * p3 + (N - 3) * p4 >= 1))
    {
        throw std::runtime_error("Ill defined probabilities.");
    }
    if (N < 3)
    {
        throw std::runtime_error("Ill defined number of holds");
    }

    // real_t p0 = 1.0 - N * p1 - (N - 1) * p2 - (N - 2) * p3 - (N - 3) * p4;
    real_t p0 = 1.0 - (N - 3) * p4;

    bitset<N> noFailures;
    probability_space.emplace_back(noFailures, p0);

    // Example: pair of a vector of ints and a float
    // probability_space.emplace_back(std::bitset<N>(), p0);

    // for (int i = 0; i < N; i++)
    // {
    //     bitset<N> singleFailure;
    //     singleFailure.set(i);
    //     probability_space.emplace_back(singleFailure, p1);
    // }

    // for (int i = 0; i < N - 1; i++)
    // {
    //     bitset<N> doubleFailure;
    //     doubleFailure.set(i);
    //     doubleFailure.set(i + 1);
    //     probability_space.emplace_back(doubleFailure, p2);
    // }

    // for (int i = 0; i < N - 2; i++)
    // {
    //      bitset<N> tripleFailure;
    //      tripleFailure.set(i);
    //      tripleFailure.set(i + 1);
    //      tripleFailure.set(i + 2);
    //      probability_space.emplace_back(tripleFailure, p3);
    // }

    for (int i = 0; i < N - 3; i++)
    {
        bitset<N> quadrupleFailure;
        quadrupleFailure.set(i);
        quadrupleFailure.set(i + 1);
        quadrupleFailure.set(i + 2);
        quadrupleFailure.set(i + 3);
        probability_space.emplace_back(quadrupleFailure, p4);
    }

    return probability_space;
}

#include <bitset>
#include <tuple>
#include <vector>
#include <random>
#include <stdexcept>
#include <cstddef>   // size_t
#include <numeric>   // std::accumulate

// Assumes real_t is already defined, e.g.
// using real_t = double;

template <std::size_t N, class URBG>
std::vector<std::size_t> sample_k_indices_without_replacement(
    const std::vector<std::tuple<std::bitset<N>, real_t, real_t>>& latent_probabilities,
    const real_t cvar_alpha,
    std::size_t k,
    real_t tau,
    URBG& rng)
{
    if (tau < real_t(0)) {
        throw std::invalid_argument("tau must be nonnegative");
    }

    const std::size_t n = latent_probabilities.size();
    if (k > n) {
        throw std::invalid_argument("k cannot exceed latent_probabilities.size()");
    }

    // Extract weights from the 3rd tuple entry, thresholding small ones to zero.
    std::vector<double> weights(n, 0.0);
    std::size_t positive_count = 0;

    for (std::size_t i = 0; i < n; ++i) {
        auto &[bits, original_probability, latent_probability] = latent_probabilities[i];
        real_t w = sigmoid(latent_probability) * original_probability / (1 - cvar_alpha);

        if (w < real_t(0)) {
            throw std::invalid_argument("distribution contains a negative weight");
        }

        if (w >= tau) {
            weights[i] = static_cast<double>(w);
            ++positive_count;
        }
    }

    if (k > positive_count) {
        throw std::invalid_argument(
            "Not enough entries with weight >= tau to sample k distinct indices"
        );
    }

    std::vector<std::size_t> sampled_indices;
    sampled_indices.reserve(k);

    for (std::size_t draw = 0; draw < k; ++draw) {
        const double total_weight =
            std::accumulate(weights.begin(), weights.end(), 0.0);

        if (total_weight <= 0.0) {
            throw std::runtime_error("No positive weight left before completing k draws");
        }

        std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
        const std::size_t idx = dist(rng);

        sampled_indices.push_back(idx);

        // Remove from future draws: sampling without replacement.
        weights[idx] = 0.0;
    }

    return sampled_indices;
}

// [bits, original_probability, latent_probability]

// it is assumed these are in the same order
real_t cvar_divergence_beween_distributions(
    std::vector<std::tuple<std::bitset<N>, real_t, real_t>> latent_probabilities_1,
    std::vector<std::tuple<std::bitset<N>, real_t, real_t>> latent_probabilities_2,
    real_t cvar_alpha
) {
    if (latent_probabilities_1.size() != latent_probabilities_2.size()) {
        std::cout << "Different Distribution Sizes." << std::endl;
        throw std::runtime_error("Cannot calculate divergence between two distributions of different sizes.");
    }

    real_t sum = 0.0;
    real_t double_eps = std::numeric_limits<real_t>::epsilon() * 100.0;
    for (size_t i = 0; i < latent_probabilities_1.size(); i++) {
        auto [bits_1, original_probability_1, latent_probability_1] = latent_probabilities_1[i];
        auto [bits_2, original_probability_2, latent_probability_2] = latent_probabilities_2[i];
       
        if (bits_1 != bits_2) {
            std::cout << "Key error." << std::endl;
            throw std::runtime_error("Cannot calculate divergence between two distributions with different supports.");
        }
        // due to floating-point roundoff we compare with a small tolerance
        {
            const real_t tol = double_eps;
            if (std::abs(original_probability_1 - original_probability_2) > tol)
            {
                throw std::runtime_error(
                    "Cannot calculate divergence between two distributions with "
                    "significantly different original probabilities.");
            }
        }

        sum += (original_probability_1 / (1 - cvar_alpha)) * sigmoid(latent_probability_1) * log((sigmoid(latent_probability_1) + double_eps) / (sigmoid(latent_probability_2) + double_eps));
        sum += (original_probability_1 / (1 - cvar_alpha)) * (1 - sigmoid(latent_probability_1)) * log((1 - sigmoid(latent_probability_1) + double_eps) / (1 - sigmoid(latent_probability_2) + double_eps));
    }
    return sum;
}


real_t proj_latent_onto_probability_simplex(
    std::vector<std::tuple<std::bitset<N>, real_t, real_t>> latent_probabilities_k_jp1_unnormalized,
    std::vector<std::tuple<std::bitset<N>, real_t, real_t>> &latent_probabilities_normalized,
    real_t alpha,
    real_t tol = 1e-12,
    int max_its = 10)
{
    int myid = Mpi::WorldRank();
    max_its = std::max(max_its, 10);

    // todo [asnesw]: Figure out edge cases with these guys.
    real_t negative_latent_max = (real_t) -(DBL_MAX - 1);
    real_t negative_latent_min = (real_t) DBL_MAX;

    for (auto &[bits, original_probability, latent_probability] : latent_probabilities_k_jp1_unnormalized)
    {
        negative_latent_max = std::max(negative_latent_max, -latent_probability);
        negative_latent_min = std::min(negative_latent_min, -latent_probability);
    };

    // these bound 1
    real_t a = inv_sigmoid(1 - alpha) + negative_latent_min - 1.0;
    real_t b = inv_sigmoid(1 - alpha) + negative_latent_max + 1.0;

    // // we avoid summing
    // real_t calculate_summation_for_t(real_t t) {
    //     real_t sum = 0.0;
    //     for (auto &[bits, original_probability, latent_probability] : latent_probabilities_k_jp1_unnormalized) {
    //         sum += (original_probability / (1 - alpha)) * sigmoid(latent_probabilty + t);
    //     }
    //     return sum;
    // };
    auto calculate_summation_for_t = [&](real_t t) -> real_t
    {
        real_t sum = 0.0;
        for (auto &[bits, original_probability, latent_probability] : latent_probabilities_k_jp1_unnormalized)
        {
            sum += (original_probability / (1 - alpha)) * sigmoid(latent_probability + t);
        }
        return sum;
    };

    real_t a_vol_minus = calculate_summation_for_t(a) - 1;
    real_t b_vol_minus = calculate_summation_for_t(b) - 1;

    if (a_vol_minus * b_vol_minus > 0) {
        if (myid == 0) {
            std::cout << "ERROR: a_vol_minus = " << a_vol_minus << ", b_vol_minus = " << b_vol_minus << endl;
        }
        throw std::runtime_error("Invalid bounds for simplex projection.");
    }

    bool done = false;
    real_t x;
    real_t x_vol_minus;

    for (int k = 0; k < max_its; k++) // Illinois iteration
    {
        if (myid == 0) {
            cout << "Iteration count " << k << ".\n";
        }

        // false–position step
        x = b - b_vol_minus * (b - a) / (b_vol_minus - a_vol_minus);
        real_t x_vol_minus = calculate_summation_for_t(x) - 1; // f(x)

        if (b_vol_minus * x_vol_minus < 0)
        {
            a = b;
            a_vol_minus = b_vol_minus;
        }
        else
        {
            a_vol_minus = a_vol_minus / 2;
        }
        b = x;
        b_vol_minus = x_vol_minus;

        if (abs(x_vol_minus) < tol)
        {
            done = true;
            break;
        }
    }

    for (auto &[bits, original_probability, latent_probability] : latent_probabilities_k_jp1_unnormalized)
    {
        latent_probabilities_normalized.emplace_back(bits, original_probability, latent_probability + x);
    }

    if (!done)
    {
        mfem_warning("Simplex Projection reached maximum iteration without converging. "
                     "Result may not be accurate.");
    }

    return x; // this is the error
}

/**
 * @brief Bregman projection of ρ = sigmoid(ψ) onto the subspace
 *        ∫_Ω ρ dx = θ vol(Ω) as follows:
 *
 *        1. Compute the root of the R → R function
 *            f(c) = ∫_Ω sigmoid(ψ + c) dx - θ vol(Ω)
 *        2. Set ψ ← ψ + c.
 *
 * @param psi a GridFunction to be updated
 * @param target_volume θ vol(Ω)
 * @param tol Newton iteration tolerance
 * @param max_its Newton maximum iteration number
 * @return real_t Final volume, ∫_Ω sigmoid(ψ)
 */
real_t proj(ParGridFunction &psi, real_t target_volume, real_t domain_volume, real_t tol = 1e-12,
            int max_its = 10)
{
    ParGridFunction onegf(psi.ParFESpace());
    onegf = 1.0;

    GridFunctionCoefficient psi_coeff(&psi);

    real_t psimax;
    {
        real_t locmax = psi.Normlinf();
        // MPI_DOUBLE should be replaced with the MFEM data type
        MPI_Allreduce(&locmax, &psimax, 1, MPI_DOUBLE, MPI_MAX, psi.ParFESpace()->GetComm());
    }

    const real_t volume_proportion = target_volume / domain_volume;

    real_t a = inv_sigmoid(volume_proportion) - psimax; // lower bound of 0
    real_t b = inv_sigmoid(volume_proportion) + psimax; // upper bound of 0

    ConstantCoefficient aCoefficient(a);
    ConstantCoefficient bCoefficient(b);

    SumCoefficient psiA(psi_coeff, aCoefficient);
    SumCoefficient psiB(psi_coeff, bCoefficient);

    TransformedCoefficient sigmoid_psi_a(&psiA, sigmoid);
    TransformedCoefficient sigmoid_psi_b(&psiB, sigmoid);

    ParLinearForm int_sigmoid_psi_a(psi.ParFESpace());
    int_sigmoid_psi_a.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi_a));
    int_sigmoid_psi_a.Assemble();
    real_t a_vol_minus = int_sigmoid_psi_a(onegf) - target_volume;

    ParLinearForm int_sigmoid_psi_b(psi.ParFESpace());
    int_sigmoid_psi_b.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi_b));
    int_sigmoid_psi_b.Assemble();
    real_t b_vol_minus = int_sigmoid_psi_b(onegf) - target_volume;

    bool done = false;
    real_t x;
    real_t x_vol;

    for (int k = 0; k < max_its; k++) // Illinois iteration
    {
        // cout << "Iteration count " << k << ".\n";
        x = b - b_vol_minus * (b - a) / (b_vol_minus - a_vol_minus);

        ConstantCoefficient xCoefficient(x);
        SumCoefficient psiX(psi_coeff, xCoefficient);
        TransformedCoefficient sigmoid_psi_x(&psiX, sigmoid);

        ParLinearForm int_sigmoid_psi_x(psi.ParFESpace());
        int_sigmoid_psi_x.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi_x));
        int_sigmoid_psi_x.Assemble();
        x_vol = int_sigmoid_psi_x(onegf);

        real_t x_vol_minus = x_vol - target_volume;

        // cout << "target_volume: " << target_volume << ", domain_volume: " << domain_volume << ", a_vol_minus: " << a_vol_minus << ", b_vol_minus: " << b_vol_minus << " x_vol_minus: " << x_vol_minus << ".\n";

        if (b_vol_minus * x_vol_minus < 0)
        {
            a = b;
            a_vol_minus = b_vol_minus;
        }
        else
        {
            a_vol_minus = a_vol_minus / 2;
        }
        b = x;
        b_vol_minus = x_vol_minus;

        if (abs(x_vol_minus) < tol)
        {
            done = true;
            // cout << "x_vol_minus = " << x_vol_minus << " IS WITHINT tolerance of " << tol << ".\n";
            break;
        }
        else
        {
            // cout << "x_vol_minus = " << x_vol_minus << " not within tolerance of " << tol << ".\n";
        }
    }

    psi += x;
    if (!done)
    {
        mfem_warning("Projection reached maximum iteration without converging. "
                     "Result may not be accurate.");
    }
    return x_vol;
}

template <size_t N, class real_t, class Index>
void generate_distribution_from_latents(
    const std::vector<std::tuple<std::bitset<N>, real_t, real_t>>& latent_probabilities,
    real_t cvar_alpha,
    std::discrete_distribution<Index>& dist   // overwritten in-place
) {
    if (latent_probabilities.empty()) throw std::runtime_error("Cannot build distribution from empty items.");

    std::vector<double> weights;
    weights.reserve(latent_probabilities.size());

    for (auto const& [bits, original_p, latent_p] : latent_probabilities) {
        double w = static_cast<double>(sigmoid(latent_p) * original_p / (1 - cvar_alpha));
        if (!std::isfinite(w) || w < 0.0) w = 0.0; // must be >= 0
        weights.push_back(w);
    }

    const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (!(sum > 0.0)) throw std::runtime_error("All weights are zero (or invalid).");

    // overwrite the existing distribution object
    dist = std::discrete_distribution<Index>(weights.begin(), weights.end());
}

// Return true when RUN_DETERMINISTIC_UNTIL < 0 or outer_iteration >= RUN_DETERMINISTIC_UNTIL >= 0
// recall we use 1-indexing for the outer_iteration.
bool is_deterministic_step(
    unsigned int outer_iteration,
    const int RUN_DETERMINISTIC_UNTIL
) {
    return (RUN_DETERMINISTIC_UNTIL < 0) || (RUN_DETERMINISTIC_UNTIL >= 0 && outer_iteration <= RUN_DETERMINISTIC_UNTIL);
}

class DensCoeff : public Coefficient
{
public:
    DensCoeff(real_t ll = 1) : len(ll)
    {
    }

    virtual ~DensCoeff() {};

    virtual real_t Eval(ElementTransformation &T,
                        const IntegrationPoint &ip) override
    {
        T.SetIntPoint(&ip);
        real_t x[3];
        Vector transip(x, 3);
        transip = 0.0;
        T.Transform(ip, transip);

        real_t l = transip.Norml2();
        return ((sin(len * transip(0)) * cos(len * transip(1))) > real_t(-0.1));
    }

private:
    real_t len;
};

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
    const char *mesh_file = "canti_2D_6.msh";
    int order = 2;
    bool static_cond = false;
    bool pa = false;
    bool fa = false;
    const char *device_config = "cpu";
    bool visualization = true;
    bool algebraic_ceed = false;
    real_t filter_radius = real_t(0.06); // original 0.01

    const real_t starting_gamma = 1.0; // gamma is the stepsize for updating the prox for the latent probabilities. Original 0.001.
    real_t gamma = starting_gamma;

    // starting for armijo backtracking line search. Nothing to do with cvar.
    // There are more intelligent ways for pure-primal algorithms to update this, since, in our algorithm
    // we update q using the stepsize before updating x, this is a simpler initialization. 

    const real_t starting_alpha_stepsize = 1.0; // no opinion here. But, hey, let's be aggresive since we are doing backtracking line search. Original 1.0 
    const real_t ARMIJO_C1 = 1e-4; // armijo backtracking line search parameter. Original 1e-4
    const real_t ARMIJO_BETA = 0.1; // armijo backtracking line search parameter. Original 0.5
    const real_t MIN_ALPHA = 1e-2;

    const real_t starting_rho = 1.0;
    real_t rho = starting_rho;



    real_t vol_fraction = 0.45;
    // int max_it = 100;
    real_t itol = 1e-1;
    real_t ntol = 1e-4;
    real_t rho_min = 1e-3;
    real_t E_min = 1e-6; // try 1e-1, maybe 1e-3. Original 1e-6
    real_t E_max = 1.0;
    real_t poisson_ratio = 0.02;
    bool glvis_visualization = true;
    bool paraview_output = true;

    const real_t tau = 1e-10; // threshold for probability truncation.

    const real_t p1 = 0.01; 
    const real_t p2 = 0.01; 
    const real_t p3 = 0.01; 
    const real_t p4 = 0.20; // SET BACK TO 0.005

    const real_t cvar_alpha = 0.05;
    const int outer_loop_iterations = 30;
    const int inner_loop_iterations = 10;
    const int MAX_BACKTRACKING_ATTEMPTS = 10; // maximum number of backtracking attempts in each inner loop iteration.

    const real_t block_size_ratio = 0.8; // change back to 0.6
    const int STOCHASTIC_GRADIENT_MINIBATCH_SIZE = 2;

    // encoded exclusvely. If >=, 0, start running stochastic at the RUN_DETERMINISTIC_UNTILth step
    const int RUN_DETERMINISTIC_UNTIL = 0; 


    // "const" or "gradient"
    const float epsilon_TV = 1e-3;
    const float epsilon_g = 1e-3;
    const float epsilon_q = 1e-3;

    std::vector<std::pair<std::bitset<N>, real_t>> probability_space = getProbabilitySpace(p1, p2, p3, p4);

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
    filt->AddBC(1, 1.0); // free hold
    filt->AddBC(2, 1.0);
    filt->AddBC(3, 1.0);
    filt->AddBC(4, 1.0);
    filt->AddBC(5, 1.0);
    filt->AddBC(6, 1.0);
    filt->AddBC(7, 1.0);
    filt->AddBC(8, 0.0); // outer boundary
    // allocate the slover after setting the BC and before applying the filter
    filt->Assemble();

    // define the control latent variable. Get an "old" version for saving later
    // However, we note we don't need to be as careful as with q because we only need the latest version of the density
    ParGridFunction odens_latent(filt->GetDesignFES());
    ParGridFunction odens_latent_old(filt->GetDesignFES());
    // ParGridFunction odens_latent_old_old(filt->GetDesignFES()); // for GBB test

    // define the control field
    MappedGridFunctionCoefficient odens(&odens_latent, sigmoid);
    MappedGridFunctionCoefficient odens_old(&odens_latent_old, sigmoid);
    // MappedGridFunctionCoefficient odens_old_old(&odens_latent_old_old, sigmoid); // for GBB test
    // define the filtered field
    ParGridFunction fdens(filt->GetFilterFES());

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
    odens_latent = inv_sigmoid(vol_fraction); //@Will is that stable? What is inv_sigmoid
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

    ParGridFunction& sol=elsolver->GetDisplacements();

    // set up the paraview
    mfem::ParaViewDataCollection paraview_dc("cvar_optimization_stochastic_13Mar2026_true_stochastic_highgama_TRUE", &pmesh);
    // rho_gf.ProjectCoefficient(rho);
    paraview_dc.SetPrefixPath("ParaView");
    paraview_dc.SetLevelsOfDetail(order);
    paraview_dc.SetDataFormat(VTKFormat::BINARY);
    paraview_dc.SetHighOrderOutput(true);
    paraview_dc.SetCycle(0);
    paraview_dc.SetTime(0.0);
    paraview_dc.RegisterField("density", &odens_gf);
    paraview_dc.RegisterField("filtered_density", &fdens);
    paraview_dc.RegisterField("control_gradient", &odens_gradient_single);
    paraview_dc.RegisterField("latent_density", &odens_latent);
    paraview_dc.RegisterField("disp",&sol);
    // paraview_dc.RegisterField("displacements", &displacements);
    paraview_dc.Save();

    // define material and interpolation parameters coefficient factory
    IsoComplCoef icc;
    icc.SetGridFunctions(&fdens, &(elsolver->GetDisplacements())); // (1) density (2) displacements. Deferred calculation.
    icc.SetMaterial(E_min, E_max, poisson_ratio);
    icc.SetSIMP(3.0);

    // set the material to the elasticity solver
    elsolver->SetMaterial(*(icc.GetE()), *(icc.GetNu())); // take parameters from ICC. It's a coefficient factory. GetE() is a function of density, GetNu() is a constant function (might be a density)

    // set surface load
    elsolver->AddSurfLoad(1, 0.0, 1.0); // set the load on the free hole. 0.0 == x direction, 1.0 == y direction.

    ParBilinearForm mass(filt->GetDesignFES());
    mass.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator(one)));
    mass.Assemble();
    HypreParMatrix M;
    Array<int> empty;
    mass.FormSystemMatrix(empty, M);
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
    real_t alpha_stepsize;// / std::max(gradient_max, 1e-8); // initial stepsize

    // loop
    for (unsigned int k = 1; k <= outer_loop_iterations; k++)
    {
        if (k > 1)
        {
            // maybe this needs to escalate faster? write a better stepsize finder. Also maybe different gamma for inner and outer loops
            gamma *= ((real_t)k) / ((real_t)k - 1);
            // this needs to go to 0. Opposite reasoning :-) 
            rho *= ((real_t)k - 1) / ((real_t)k); 
        }
        if (myid == 0)
        {
            cout << "\nStep = " << k << endl;
        }
        // cout << "K IS " << k << endl;

        filt->FFilter(&odens, fdens);

        {
            odens_gf.ProjectCoefficient(odens);
            paraview_dc.SetCycle(total_iterations);
            paraview_dc.SetTime((real_t)total_iterations);
            paraview_dc.Save();
        }

        // first we calculate and find the new latent probabilities.
        // real_t inner_loop_gamma;
        // if (inner_loop_gamma_choice == "restart") {
        //     inner_loop_gamma = starting_gamma;
        // } else if (inner_loop_gamma_choice == "lockstep") {
        //     inner_loop_gamma = gamma;
        // } else {
        //     throw std::runtime_error("Invalid gamma");
        // }

        // cout << "STARTING RUNNING LOOP" << endl;

        // int inner = 1;

        // now we also set 

        std::vector<std::size_t> block_k = sample_k_indices_without_replacement(latent_probabilities_k_0, cvar_alpha, BLOCK_SIZE, tau, rng);
        // float delta_k = 0.0;
        // for (auto &[bits, original_probability, latent_probability] : latent_probabilities_k_0){
        //     delta_k += (original_probability / (1 - cvar_alpha)) * sigmoid(latent_probability);
        // }

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

        for (size_t inner = 1; inner <= inner_loop_iterations; inner++)
        {
                        // TODO: Sample ahead of time. Makes it simpler. Fewer computes.
            std::vector<size_t> latent_probability_indices_to_sample;
            std::vector<float> latent_probability_indices_to_sample_weights;
            std::vector<bool> latent_probability_indices_to_sample_mask; // for debugging. Remove later.
            bool xi_in_empty = true;

            // determine where we're sampling. If in the stochastic case, we can skip calculating unnecessary compliances.
            if (is_deterministic_step(k, RUN_DETERMINISTIC_UNTIL)) {
                // we need to calculate q_k_jp1 for this. So defer.
                latent_probability_indices_to_sample_mask.assign(latent_probabilities_k_0.size(), true);
            } else {

                // we can sample, and filter later.
                std::cout << "Running stochastic sampling for gradient estimation at outer iteration " << k << ".\n";   

                latent_probability_indices_to_sample_mask.assign(latent_probabilities_k_0.size(), false);
                std::vector<real_t> weights_by_index(latent_probabilities_k_0.size(), 0); 

                for (int sample = 0; sample < STOCHASTIC_GRADIENT_MINIBATCH_SIZE; ++sample) {
                    size_t sampled_index = q_distribution(rng);

                    if (myid == 0) std::cout << "Sample " << sample << ": sampled_index = " << sampled_index << std::endl;

                    bool is_in_mask = block_membership_mask[sampled_index];
                    if (myid == 0) std::cout << "  is_in_mask = " << is_in_mask << std::endl;

                    if (!is_in_mask) {
                        weights_by_index[sampled_index] += 1.0 / STOCHASTIC_GRADIENT_MINIBATCH_SIZE;
                        latent_probability_indices_to_sample_mask[sampled_index] = true;

                        if (myid == 0) std::cout << "  Added weight for non-block member" << std::endl;
                    } else {
                        // real_t q_k_jp1_over_q_k_j = sigmoid(latent_probability_jp1) / sigmoid(latent_probability_j);
                        // weights_by_index[sampled_index] += q_k_jp1_over_q_k_j / STOCHASTIC_GRADIENT_MINIBATCH_SIZE;
                        latent_probability_indices_to_sample_mask[sampled_index] = true;

                        weights_by_index[sampled_index] += 1.0 / STOCHASTIC_GRADIENT_MINIBATCH_SIZE;
                        if (myid == 0) std::cout << "  Added weight for block member" << std::endl;
                    }
                }

                latent_probability_indices_to_sample.reserve(STOCHASTIC_GRADIENT_MINIBATCH_SIZE);
                latent_probability_indices_to_sample_weights.reserve(STOCHASTIC_GRADIENT_MINIBATCH_SIZE);

                for (size_t i = 0; i < weights_by_index.size(); ++i) {
                    if (weights_by_index[i] > 0.0) {
                        latent_probability_indices_to_sample.push_back(i);
                        latent_probability_indices_to_sample_weights.push_back(weights_by_index[i]);
                    }
                }

                std::cout << "Sampled indices for gradient estimation: ";
                for (size_t i = 0; i < latent_probability_indices_to_sample.size(); ++i) {
                    std::cout << latent_probability_indices_to_sample[i] << " (weight: " << latent_probability_indices_to_sample_weights[i] << ") ";
                }
                std::cout << std::endl;
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
                    std::cout << "For proximal probability, going through "
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
                for (int i = 2; i < 8; i++)
                {
                    if (!(bits.test(i - 2)))
                    {
                        elsolver->AddDispBC(i, 4, 0.0); // start with all of them fixed. 0 displacement in all directions.
                    } else {
                        if (myid == 0) {
                            std::cout << "Skipping bit " << (i - 2) << " in scenario " << bits << ".\n";
                        }
                    }
                }

                // solve the discrete elastic system
                elsolver->Assemble();
                elsolver->FSolve(); // solve for displacements

                ParGridFunction &sol = elsolver->GetDisplacements();
                icc.SetGridFunctions(&fdens, &sol);

                real_t compliance_i = 0.0;
                {
                    ParLinearForm compl_form(filt->GetFilterFES());
                    compl_form.AddDomainIntegrator(new DomainLFIntegrator(icc));
                    compl_form.Assemble();
                    compliance_i = compl_form(onegf); // inner product of compliance and constant 1 function. IE the integral.
                }

                base_compliances_by_index.emplace_back(compliance_i);

                if (myid == 0) {
                    std::cout << "Latent: " << latent_probability << ", gamma: " << gamma << ", compliance: " << compliance_i << ".\n";
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
                std::cout << "Printing latent_probabilities_k_jp1_unnormalized within inner loop, after calculating.\n";
                for (size_t i = 0; i < latent_probabilities_k_jp1_unnormalized.size(); ++i) {
                    std::cout << i << ":" << std::get<2>(latent_probabilities_k_jp1_unnormalized[i]) << " ";
                }
                std::cout << "\n";
                std::cout << "Done printing latent_probabilities_k_jp1_unnormalized.\n";
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
                    std::cout <<"case: "<<bits.to_string()<<" "<<" p="<<original_probability<<" l="<<latent_probability<<std::endl;
                }
            }

            /**
            Now we are done with the q prox update! 
            **/

            // Now we make it sample-able. Also store this (minor cost, but we should make more opportunistic for best-practice.)   
            
                        /****
            Deterime I_j,  the set of indices we will sample gradients from.
            ****/

            if (is_deterministic_step(k, RUN_DETERMINISTIC_UNTIL)) {
                std::cout << "Running deterministic sampling for gradient estimation at outer iteration " << k << ".\n";

                /***
                In the deterministic case, we simply remove the q_i with numerically trivial probabilities
                ***/
                latent_probability_indices_to_sample.reserve(latent_probabilities_k_jp1.size());
                latent_probability_indices_to_sample_weights.reserve(latent_probabilities_k_jp1.size());

                for (size_t i = 0; i < latent_probabilities_k_jp1.size(); ++i) {

                    auto &[bits, original_probability, latent_probability] = latent_probabilities_k_jp1[i];
                    float probability = sigmoid(latent_probability) * original_probability / (1 - cvar_alpha);

                    if (probability > epsilon_q) {
                        latent_probability_indices_to_sample.push_back(i);
                        latent_probability_indices_to_sample_weights.push_back(probability);
                        latent_probability_indices_to_sample_mask[i] = true;
                    }
                }
            } else {
                std::cout << "Filtering post-stochastic sampling. " << k << ".\n";   

                for (size_t SAMPLE_INDEX = 0; SAMPLE_INDEX < latent_probability_indices_to_sample.size(); ++SAMPLE_INDEX) {
                    real_t sample = latent_probability_indices_to_sample[SAMPLE_INDEX];

                    if (block_membership_mask[sample]) {
                        real_t sample_weight = latent_probability_indices_to_sample_weights[SAMPLE_INDEX];

                        auto [bits_j, original_probability_j, latent_probability_j] = latent_probabilities_k_j[sample];
                        auto [bits_jp1, original_probability_jp1, latent_probability_jp1] = latent_probabilities_k_jp1[sample];

                        real_t q_k_jp1 = sigmoid(latent_probability_jp1) * original_probability_jp1 / (1 - cvar_alpha);
                        if (myid == 0) std::cout << "  q_k_jp1 = " << q_k_jp1 << ", epsilon_q = " << epsilon_q << std::endl;

                        if (q_k_jp1 < epsilon_q) {
                            if (myid == 0) std::cout << "Erasing sample " << sample << " because q_k_jp1 < epsilon_q" << std::endl;

                            latent_probability_indices_to_sample.erase(latent_probability_indices_to_sample.begin() + SAMPLE_INDEX);
                            latent_probability_indices_to_sample_weights.erase(latent_probability_indices_to_sample_weights.begin() + SAMPLE_INDEX);
                            latent_probability_indices_to_sample_mask[sample] = false;

                            continue; // skip if the probability is too small, for non-triviality and numerical issues in ratio
                        } else {
                            xi_in_empty = false; // now we know that there is a nontrivial weight.
                        }
                    } 
                }
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

                if (0 == myid)
                {
                    std::cout << "For Gradient, going through "
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
                for (int i = 2; i < 8; i++)
                {
                    if (!(bits.test(i - 2)))
                    {
                        elsolver->AddDispBC(i, 4, 0.0); // start with all of them fixed. 0 displacement in all directions.
                    }
                }

                // solve the discrete elastic system
                elsolver->Assemble();
                elsolver->FSolve(); // solve for displacements

                // extract the solution
                ParGridFunction &sol = elsolver->GetDisplacements();
                // if (myid == 0) {
                //     std::cout << "Displacement norm: " << sol.ComputeL2Error(zero) << std::endl;
                // }
                icc.SetGridFunctions(&fdens, &sol);

                // Apply the adjoint filter operation
                filt->AFilter(icc.GetGradIsoComp(), odens_gradient_single); // adjoint operation to projection to find differential in design space. Chain rule.
                // if (myid == 0) {
                //     std::cout << "After AFilter, norm odens_gradient_single: " << odens_gradient_single.ComputeL2Error(zero) << std::endl;
                // }
                // get the true vector of the gradient
                odens_gradient_single.SetTrueVector();
                M.Mult(odens_gradient_single.GetTrueVector(), tmp_grad);
                odens_gradient_single.SetFromTrueDofs(tmp_grad);
                // if (myid == 0) {
                //     std::cout << "After smoothing, norm odens_gradient_single: " << odens_gradient_single.ComputeL2Error(zero) << std::endl;
                // }
                // ogf.GetTrueDofs(ograd); // extracting the information in the ParGridFunction into the ograd. Lives in dual of design space.

                real_t weight = latent_probability_indices_to_sample_weights[i];
                if (!is_deterministic_step(k, RUN_DETERMINISTIC_UNTIL) && block_membership_mask[key]) {
                    const auto [bits_0, original_probability_0, latent_probability_0] = latent_probabilities_k_0[key];

                    // initial sampling was from q^k so we have to resample to have the right weight
                    weight = weight * sigmoid(latent_probability) / sigmoid(latent_probability_0); 
                }

                if (is_deterministic_step(k, RUN_DETERMINISTIC_UNTIL) || block_membership_mask[key] || xi_in_empty)
                {
                    real_t gradient_magnitude;
                    {
                        real_t locl2 = odens_gradient_single.Norml2();
                        // MPI_DOUBLE should be replaced with the MFEM data type
                        MPI_Allreduce(&locl2, &gradient_magnitude, 1, MPI_DOUBLE, MPI_SUM, odens_gradient_single.ParFESpace()->GetComm());
                    }
                    
                    x_j_iteration_max_gradient = std::max(x_j_iteration_max_gradient, gradient_magnitude);
                }

                odens_weighted_gradient.Add(weight, odens_gradient_single); // we will use this for the GBB stepsize calculation after the loop. Note that the weight is already taking into account the ratio q_k_jp1 / q_k_j for the stochastic case, and is just the probability for the deterministic case.
                // odens_latent.Add(-inner_loop_gamma * weight, odens_gradient_single);
                // if (compute_gradient) {
                //     odens_gradient_holder.Add(normalizing_factor, odens_gradient_single);
                // }
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
                        std::cout << "BASE Block member " << key << ": compliance=" << compliance << ", nabla_phi_plus_f=" << nabla_phi_plus_f 
                                  << ", h_i=" << h_i << ", summand=" << summand << ", weight=" << weight << std::endl;
                    }
                    F_x += weight * summand;
                } else {
                    F_x += weight * compliance;
                    if (myid == 0) {
                        std::cout << "BASE Non-Block member " << key << ": compliance=" << compliance << ", weight=" << weight << std::endl;
                    }
                }
            }

            // stepsize for armijo backtracking line search. 
            // We treat each inner loop via armijo backtracking line search.
            // Note that q^{k, j+1} is a function of x. 

            if (alpha_stepsize > 0) {
                // generalized BB stepsize calculation.

                SumCoefficient odens_difference_coeff(odens, odens_old, 1.0, -1.0);
                odens_difference.ProjectCoefficient(odens_difference_coeff);

                GridFunctionCoefficient latent_coeff(&odens_latent);
                GridFunctionCoefficient latent_old_coeff(&odens_latent_old);
                SumCoefficient odens_latent_difference_coeff(latent_coeff, latent_old_coeff, 1.0, -1.0);
                odens_latent_difference.ProjectCoefficient(odens_latent_difference_coeff);

                GridFunctionCoefficient grad_coeff(&odens_weighted_gradient);
                GridFunctionCoefficient grad_old_coeff(&odens_weighted_gradient_old);
                SumCoefficient odens_gradient_difference_coeff(grad_coeff, grad_old_coeff, 1.0, -1.0);
                odens_gradient_difference.ProjectCoefficient(odens_gradient_difference_coeff);

                // Compute L2 norms of the differences (for debugging / sanity checks)
                ConstantCoefficient zero_c(0.0);
                real_t local_dx_norm = odens_difference.ComputeL2Error(zero_c);
                real_t local_dpsi_norm = odens_latent_difference.ComputeL2Error(zero_c);
                real_t local_dg_norm = odens_gradient_difference.ComputeL2Error(zero_c);

                real_t dx_norm_sq = local_dx_norm * local_dx_norm;
                real_t dpsi_norm_sq = local_dpsi_norm * local_dpsi_norm;
                real_t dg_norm_sq = local_dg_norm * local_dg_norm;

                real_t global_dx_norm_sq, global_dpsi_norm_sq, global_dg_norm_sq;
                MPI_Allreduce(&dx_norm_sq, &global_dx_norm_sq, 1, MPI_DOUBLE, MPI_SUM, filt->GetDesignFES()->GetComm());
                MPI_Allreduce(&dpsi_norm_sq, &global_dpsi_norm_sq, 1, MPI_DOUBLE, MPI_SUM, filt->GetDesignFES()->GetComm());
                MPI_Allreduce(&dg_norm_sq, &global_dg_norm_sq, 1, MPI_DOUBLE, MPI_SUM, filt->GetDesignFES()->GetComm());

                real_t dx_norm = std::sqrt(global_dx_norm_sq);
                real_t dpsi_norm = std::sqrt(global_dpsi_norm_sq);
                real_t dg_norm = std::sqrt(global_dg_norm_sq);

                // Compute inner products via a linear form (consistent with how we compute gradient norms)
                ParGridFunction one_design(filt->GetDesignFES());
                one_design = 1.0;

                GridFunctionCoefficient dx_coeff(&odens_difference);
                GridFunctionCoefficient dpsi_coeff(&odens_latent_difference);
                GridFunctionCoefficient dg_coeff(&odens_gradient_difference);

                ProductCoefficient dx_dpsi_coeff(dx_coeff, dpsi_coeff);
                ProductCoefficient dx_dg_coeff(dx_coeff, dg_coeff);

                ParLinearForm ip_dx_dpsi_form(filt->GetDesignFES());
                ip_dx_dpsi_form.AddDomainIntegrator(new DomainLFIntegrator(dx_dpsi_coeff));
                ip_dx_dpsi_form.Assemble();
                real_t ip_xp_global = ip_dx_dpsi_form(one_design);

                ParLinearForm ip_dx_dg_form(filt->GetDesignFES());
                ip_dx_dg_form.AddDomainIntegrator(new DomainLFIntegrator(dx_dg_coeff));
                ip_dx_dg_form.Assemble();
                real_t ip_xg_global = ip_dx_dg_form(one_design);

                real_t gbb_ratio = std::abs(ip_xp_global / std::max(std::abs(ip_xg_global), (real_t)1e-10));

                if (myid == 0) {
                    std::cout << "BB inner-product diagnostics:\n";
                    std::cout << "  ||dx||_2 = " << dx_norm << ", ||dpsi||_2 = " << dpsi_norm << ", ||dg||_2 = " << dg_norm << "\n";
                    std::cout << "  <dx, dpsi> = " << ip_xp_global << ", <dx, dg> = " << ip_xg_global << "\n";
                }

                alpha_stepsize = std::max(std::sqrt(alpha_stepsize * gbb_ratio), MIN_ALPHA);
            } else {
                real_t gradient_max;
                {
                    real_t locmax = odens_weighted_gradient.Normlinf();
                    // MPI_DOUBLE should be replaced with the MFEM data type
                    MPI_Allreduce(&locmax, &gradient_max, 1, MPI_DOUBLE, MPI_MAX, odens_weighted_gradient.ParFESpace()->GetComm());
                }
                alpha_stepsize = std::max(starting_alpha_stepsize / gradient_max, MIN_ALPHA); // scale with alpha
            }

            // Initialize "old" state before entering the Armijo backtracking loop.
            // This ensures the first backtracking attempt has a meaningful dx/dpsi/dg
            // when computing inner products and norms for diagnostics.

            
            odens_latent_old = odens_latent;
            // odens_weighted_gradient_old = odens_weighted_gradient;
            // Note: odens_old is a mapped coefficient based on odens_latent_old,
            // so setting odens_latent_old is sufficient to make odens_old match odens.

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
                real_t material_volume = proj(odens_latent, target_volume, domain_volume, 1e-12, 25); // last two are tol and max its
                if (0 == myid)
                {
                    cout << "On inner step " << prox_iter_attempts << ", got material " << material_volume << ". Expected Material " << target_volume << "\n";
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

                    // calculate compliance. 
                    real_t compliance_i_armijo_test;
                    {
                        elsolver->DelDispBC();
                        for (int bit_index = 2; bit_index < 8; bit_index++)
                        {
                            if (!(bits.test(bit_index - 2)))
                            {
                                elsolver->AddDispBC(bit_index, 4, 0.0); // start with all of them fixed. 0 displacement in all directions.
                            } else {
                                if (myid == 0) {
                                    std::cout << "Running Armijo Test. Skipping bit " << (bit_index - 2) << " in scenario " << bits << ".\n";
                                }
                            }
                        }

                        // solve the discrete elastic system
                        elsolver->Assemble();
                        elsolver->FSolve(); // solve for displacements

                        ParGridFunction &sol = elsolver->GetDisplacements();
                        icc.SetGridFunctions(&fdens, &sol);

                        compliance_i_armijo_test = 0.0;
                        {
                            ParLinearForm compl_form(filt->GetFilterFES());
                            compl_form.AddDomainIntegrator(new DomainLFIntegrator(icc));
                            compl_form.Assemble();
                            compliance_i_armijo_test = compl_form(onegf); // inner product of compliance and constant 1 function. IE the integral.
                        }
                    }

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
                            std::cout << "Armijo Block member " << key << ": compliance=" << compliance_i_armijo_test << ", nabla_phi_plus_f=" << nabla_phi_plus_f 
                                      << ", h_i=" << h_i << ", summand=" << summand << ", weight=" << weight << std::endl;
                        }
                        F_x_jp1 += weight * summand;
                    } else {
                        F_x_jp1 += weight * compliance_i_armijo_test;
                        if (myid == 0) {
                            std::cout << "Armijo Non-Block member " << key << ": compliance=" << compliance_i_armijo_test << ", weight=" << weight << std::endl;
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
                        std::cout << "Gradient inner product: " << gradient_inner_product << std::endl;
                    }
                }

                if (myid == 0) {
                    std::cout << "Armijo Test for prox iteration " << prox_iter_attempts << ":\n";
                    std::cout << "F(x): " << F_x << "\n";
                    std::cout << "F(x_new): " << F_x_jp1 << "\n";
                    std::cout << "Gradient Inner Product: " << gradient_inner_product << "\n";
                    std::cout << "Alpha Stepsize: " << alpha_stepsize << "\n";
                }

                // Now, run the armijo test
                if (F_x_jp1 <= F_x + ARMIJO_C1 * gradient_inner_product) {
                    if (myid == 0) {
                        std::cout << "Armijo condition met for prox iteration " << prox_iter_attempts << ".\n";
                    }
                    break; // armijo condition satisfied, break out of backtracking loop
                } else {
                    if (myid == 0) {
                        std::cout << "Armijo condition NOT met for prox iteration " << prox_iter_attempts << ". Reducing stepsize.\n";
                    }
                    alpha_stepsize *= ARMIJO_BETA; // reduce stepsize and try again
                }
            }

            // we keep the primal iterate when we break either way, so might as well project now.
            {
                total_iterations += 1;
                odens_gf.ProjectCoefficient(odens);
                paraview_dc.SetCycle(k);
                paraview_dc.SetTime((real_t)k);
                paraview_dc.Save();
            }

            // we note, this is still for DETERMINISTIC
            {
                if (myid == 0){
                    std::cout << "Projecting gradient and checking termination conditions for inner loop." << std::endl;
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

                // doing this at the last moment. 
                odens_weighted_gradient_old = odens_weighted_gradient;

                if (myid == 0) {
                    std::cout << "Inner Loop Iteration " << inner << " Termination Condition Check:\n";
                    std::cout << "Projected Gradient Magnitude: " << projected_gradient_magnitude << "\n";
                    std::cout << "Divergence from Base Distribution: " << divergence_from_q_k_0 << "\n";
                    std::cout << "Divergence from Step Distribution: " << divergence_from_q_k_jp1 << "\n";
                    std::cout << "Calculated D_j value: " << dj << "\n";
                }

                if (projected_gradient_magnitude <= x_j_iteration_max_gradient * sqrt(dj) && divergence_from_q_k_jp1 <=  dj) {
                    if (myid == 0) {
                        std::cout << "Termination conditions met for inner loop at iteration " << inner << ".\n";
                    }
                    break;
                } else if (myid == 0) {
                    std::cout << "Termination conditions NOT met for inner loop at iteration " << inner << ".\n";
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
            std::cout << "Outer Loop Iteration " << k << " Termination Condition Check:\n";
            std::cout << "Gradient Magnitude: " << g_eta_cache << "\n";
            std::cout << "T1 Distance between q_k_0 and q_k_j: " << t1_distance_qk0_qkj << "\n";
        }

        /**
        Test outer termination conditions. We note that, technically, g_eta_cache = ||G_\eta(x^{k, j}, q^{k, j+1})||.
        It add code complication (new gradient calcuation section, extra projections, etc) to compute ||G_eta(x^{k, j+1}, q^{k, j})||, as the original algorithm does.
        **/
        if (g_eta_cache < epsilon_g && 0.5 * t1_distance_qk0_qkj < epsilon_TV) {
            if (myid == 0) {
                std::cout << "Termination conditions met for outer loop at iteration " << k << ".\n";
            }
            break;
        } else if (myid == 0) {
            std::cout << "Termination conditions NOT met for outer loop at iteration " << k << ".\n";
        }


    }

    delete elsolver;
    delete filt;

    Mpi::Finalize();
    return 0;
}
