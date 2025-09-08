#include "mfem.hpp"
#include "mtop_solvers.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <utility>

//@Will I would to copy only the functions you need
#include "ex37_copy.hpp"

using namespace std;
using namespace mfem;

/*
 *
 * Get the Failure probabilities or 0, 1, 2, or 3 contiguous holes failing.
 */
// change to bitset

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

    real_t p0 = 1.0 - N * p1 - (N - 1) * p2 - (N - 2) * p3 - (N - 3) * p4;

    bitset<N> noFailures;
    probability_space.emplace_back(noFailures, p0);

    // Example: pair of a vector of ints and a float
    // probability_space.emplace_back(std::bitset<N>(), p0);

    for (int i = 0; i < N; i++)
    {
        bitset<N> singleFailure;
        singleFailure.set(i);
        probability_space.emplace_back(singleFailure, p1);
    }

    for (int i = 0; i < N - 1; i++)
    {
        bitset<N> doubleFailure;
        doubleFailure.set(i);
        doubleFailure.set(i + 1);
        probability_space.emplace_back(doubleFailure, p2);
    }

    for (int i = 0; i < N - 2; i++)
    {
         bitset<N> tripleFailure;
         tripleFailure.set(i);
         tripleFailure.set(i + 1);
         tripleFailure.set(i + 2);
         probability_space.emplace_back(tripleFailure, p3);
    }

    // for (int i = 0; i < N - 3; i++)
    // {
    //     bitset<N> quadrupleFailure;
    //     quadrupleFailure.set(i);
    //     quadrupleFailure.set(i + 1);
    //     quadrupleFailure.set(i + 2);
    //     quadrupleFailure.set(i + 3);
    //     probability_space.emplace_back(quadrupleFailure, p4);
    // }

    return probability_space;
}

//
real_t proj_latent_onto_probability_simplex(
    std::vector<std::tuple<std::bitset<N>, real_t, real_t>> latent_probabilities_unnormalized,
    std::vector<std::tuple<std::bitset<N>, real_t, real_t>> &latent_probabilities_normalized,
    real_t alpha,
    real_t tol = 1e-12,
    int max_its = 10)
{
    int myid = Mpi::WorldRank();
    max_its = std::max(max_its, 10);

    if (myid == 0) {
        std::cout << "Printing latent_probabilities_unnormalized within Illinois.\n";
        for (size_t i = 0; i < latent_probabilities_unnormalized.size(); ++i) {
            std::cout << i << ":" << std::get<1>(latent_probabilities_unnormalized[i]) << "_" << std::get<2>(latent_probabilities_unnormalized[i]) << " ";
        }
        std::cout << "\n";
        std::cout << "Done printing latent_probabilities_unnormalized.\n";
    }

    // todo [asnesw]: Figure out edge cases with these guys.
    real_t negative_latent_max = (real_t) -(DBL_MAX - 1);
    real_t negative_latent_min = (real_t) DBL_MAX;
    // if (myid == 0) {
    //     cout << "negative_latent_max: " << negative_latent_max << ", negative_latent_min: " << negative_latent_min << "\n";
    // }

    for (auto &[bits, original_probability, latent_probability] : latent_probabilities_unnormalized)
    {
        negative_latent_max = std::max(negative_latent_max, -latent_probability);
        negative_latent_min = std::min(negative_latent_min, -latent_probability);
        if (myid == 0) {
            cout << "negative_latent_max: " << negative_latent_max << ", negative_latent_min: " << negative_latent_min << "\n";
        }
    };

    // these bound 1
    real_t a = inv_sigmoid(1 - alpha) + negative_latent_min - 1.0;
    real_t b = inv_sigmoid(1 - alpha) + negative_latent_max + 1.0;

    if (myid == 0) {
        cout << "a: " << a << ", b: " << b << "\n";
    }

    // // we avoid summing
    // real_t calculate_summation_for_t(real_t t) {
    //     real_t sum = 0.0;
    //     for (auto &[bits, original_probability, latent_probability] : latent_probabilities_unnormalized) {
    //         sum += (original_probability / (1 - alpha)) * sigmoid(latent_probabilty + t);
    //     }
    //     return sum;
    // };
    auto calculate_summation_for_t = [&](real_t t) -> real_t
    {
        real_t sum = 0.0;
        for (auto &[bits, original_probability, latent_probability] : latent_probabilities_unnormalized)
        {
            sum += (original_probability / (1 - alpha)) * sigmoid(latent_probability + t);
        }
        return sum;
    };

    real_t a_vol_minus = calculate_summation_for_t(a) - 1;
    real_t b_vol_minus = calculate_summation_for_t(b) - 1;

    bool done = false;
    real_t x;
    real_t x_vol_minus;

    for (int k = 0; k < max_its; k++) // Illinois iteration
    {
        if (myid == 0) {
            cout << "Iteration count " << k << ".\n";
        }

        x = b - b_vol_minus * (b - a) / (b_vol_minus - a_vol_minus);

        real_t x_vol_minus = calculate_summation_for_t(x) - 1;

        if (myid == 0) {
            cout << "a: " << a << ", b: " << b << ", x: " << x << ".\n";
            cout << "a_vol_minus: " << a_vol_minus << ", b_vol_minus: " << b_vol_minus << " x_vol_minus: " << x_vol_minus << ".\n";
        }

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
            if (myid == 0) {
            cout << "x_vol_minus = " << x_vol_minus << " IS WITHIN tolerance of " << tol << ".\n";
            }
            break;
        }
        else
        {
            if (myid == 0) {
                cout << "x_vol_minus = " << x_vol_minus << " not within tolerance of " << tol << ".\n";
            }
        }
    }

    for (auto &[bits, original_probability, latent_probability] : latent_probabilities_unnormalized)
    {
        latent_probabilities_normalized.emplace_back(bits, original_probability, latent_probability + x);
    }

    if (!done)
    {
        mfem_warning("Simplex Projection reached maximum iteration without converging. "
                     "Result may not be accurate.");
    }
    return x_vol_minus + 1;
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
    real_t filter_radius = real_t(0.03);

    real_t gamma = 0.001;
    real_t vol_fraction = 0.3;
    int max_it = 100;
    real_t itol = 1e-1;
    real_t ntol = 1e-4;
    real_t rho_min = 1e-3;
    real_t E_min = 1e-6;
    real_t E_max = 1.0;
    real_t poisson_ratio = 0.2;
    bool glvis_visualization = true;
    bool paraview_output = true;

    const real_t p1 = 0.05;
    const real_t p2 = 0.05;
    const real_t p3 = 0.05;
    const real_t p4 = 0.0;

    const real_t cvar_alpha = 0.05;
    const int outer_loop_iterations = 150;
    const int inner_loop_iterations = 2;

    std::vector<std::pair<std::bitset<N>, real_t>> probability_space = getProbabilitySpace(p1, p2, p3, p4);

    if (myid == 0) {
    std::cout << "Size of probability space: " << probability_space.size() << "\n";
    }

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

    // define the control latent variable
    ParGridFunction odens_latent(filt->GetDesignFES());
    // define the control field
    MappedGridFunctionCoefficient odens(&odens_latent, sigmoid);
    // define the filtered field
    ParGridFunction fdens(filt->GetFilterFES());

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

    // create initial density distribution
    // DensCoeff odc(2 * M_PI); // define the coefficient, arbitrary
    // Vector odv(filt->GetDesignFES()->TrueVSize());
    // odens.GetTrueDofs(odv);
    // odens = vol_fraction; // override density to 1. Better to make it satisfy constraint.
    // odens.ProjectCoefficient(odc);
    // real_t initial_volume = vol_form(odens);
    // odens *= target_volume / initial_volume;
    // odens.SetTrueVector();
    // odens.GetTrueDofs(odv);
    Vector odens_latent_vector(filt->GetDesignFES()->TrueVSize());
    odens_latent = inv_sigmoid(vol_fraction); //@Will is that stable? What is inv_sigmoid
    odens_latent.SetTrueVector();
    odens_latent.GetTrueDofs(odens_latent_vector);

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
    ParGridFunction odens_latent_cumulative(filt->GetDesignFES());
    odens_latent_cumulative = 0.0;

    Vector fdv(filt->GetFilterFES()->TrueVSize());
    fdv = 0.0; // define the true vector of the filtered field

    // filter the initial density
    filt->FFilter(&odens,fdens);

    ParGridFunction& sol=elsolver->GetDisplacements();

    // set up the paraview
    mfem::ParaViewDataCollection paraview_dc("cvar_optimization", &pmesh);
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
    std::vector<std::tuple<std::bitset<N>, real_t, real_t>> latent_probabilities_old;
    {
        real_t latent_probability_i = log((1 - cvar_alpha) / cvar_alpha);
        for (auto &[bits, value] : probability_space)
        {
            latent_probabilities_old.emplace_back(bits, value, latent_probability_i);
        }
    }

    std::vector<std::tuple<std::bitset<N>, real_t, real_t>> latent_probabilities_unnormalized;
    std::vector<std::tuple<std::bitset<N>, real_t, real_t>> latent_probabilities;

    // loop
    for (int k = 1; k <= outer_loop_iterations; k++)
    {
        if (k > 1)
        {
            gamma *= ((real_t)k) / ((real_t)k - 1);
        }
        if (myid == 0)
        {
            cout << "\nStep = " << k << endl;
        }

        filt->FFilter(&odens, fdens);

        {
            odens_gf.ProjectCoefficient(odens);
            paraview_dc.SetCycle(k);
            paraview_dc.SetTime((real_t)k);
            paraview_dc.Save();
        }

        // first we calculate and find the new latent probabilities.
        for (int inner = 1; inner <= inner_loop_iterations; inner++)
        {
            latent_probabilities_unnormalized.clear();
            latent_probabilities.clear(); // fresh start in inner loop

            // iterate through the probability space
            for (auto &[bits, original_probability, latent_probability] : latent_probabilities_old)
            {
                if (0 == myid)
                {
                    std::cout << "Going through "
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

                if (myid == 0) {
                    std::cout << "Latent: " << latent_probability << ", gamma: " << gamma << ", compliance: " << compliance_i << ".\n";
                }

                latent_probabilities_unnormalized.emplace_back(
                    bits, original_probability, latent_probability + gamma * compliance_i);
            }

            if (myid == 0) {
                std::cout << "Printing latent_probabilities_unnormalized within inner loop, after calculating.\n";
                for (size_t i = 0; i < latent_probabilities_unnormalized.size(); ++i) {
                    std::cout << i << ":" << std::get<2>(latent_probabilities_unnormalized[i]) << " ";
                }
                std::cout << "\n";
                std::cout << "Done printing latent_probabilities_unnormalized.\n";
            }

            // then we update the new latents to ensure they sum to 1.
            proj_latent_onto_probability_simplex(
                latent_probabilities_unnormalized,
                latent_probabilities,
                cvar_alpha,
                1e-10,
                25
            );

            for (auto &[bits, original_probability, latent_probability] : latent_probabilities)
            {
                if (myid == 0) {
                    std::cout <<"case: "<<bits.to_string()<<" "<<" p="<<original_probability<<" l="<<latent_probability<<std::endl;
                }
            }


            // iterate through the probability space
            for (auto &[bits, original_probability, latent_probability] : latent_probabilities)
            {
                if (0 == myid)
                {
                    std::cout << "Going through "
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
                icc.SetGridFunctions(&fdens, &sol);

                // Apply the adjoint filter operation
                filt->AFilter(icc.GetGradIsoComp(), odens_gradient_single); // adjoint operation to projection to find differential in design space. Chain rule.
                // get the true vector of the gradient
                odens_gradient_single.SetTrueVector();
                M.Mult(odens_gradient_single.GetTrueVector(), tmp_grad);
                odens_gradient_single.SetFromTrueDofs(tmp_grad);
                // ogf.GetTrueDofs(ograd); // extracting the information in the ParGridFunction into the ograd. Lives in dual of design space.

                // #Will the ogf in the update below is not scaled with M^{-1}
                real_t new_weight = sigmoid(latent_probability) * original_probability / (1 - cvar_alpha);
                odens_latent.Add(-gamma * new_weight, odens_gradient_single);
            }

            real_t material_volume = proj(odens_latent, target_volume, domain_volume, 1e-12, 25); // last two are tol and max its
            // odens_latent.SetTrueVector();
            // odens_latent.GetTrueDofs(odens_latent_vector);

            if (0 == myid)
            {
                cout << "Got material " << material_volume << ". Expected Material " << target_volume << "\n";
            }

            // real_t total_volume_current = vol_form(odens);
            // odens *= target_volume / total_volume_current;
            // odens.GetTrueDofs(odv);

            {
                odens_gf.ProjectCoefficient(odens);
                paraview_dc.SetCycle(k);
                paraview_dc.SetTime((real_t)k);
                paraview_dc.Save();
            }
        }
        latent_probabilities_old = latent_probabilities;
    }

    delete elsolver;
    delete filt;

    Mpi::Finalize();
    return 0;
}
