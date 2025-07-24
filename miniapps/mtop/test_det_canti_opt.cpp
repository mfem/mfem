#include "mfem.hpp"
#include "mtop_solvers.hpp"

#include <fstream>
#include <iostream>

#include "ex37_copy.hpp"

using namespace std;
using namespace mfem;

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

// For Boyan: Please go through this and make sure it's parallel ready.
real_t proj(ParGridFunction &psi, real_t target_volume, real_t domain_volume, real_t tol=1e-12,
            int max_its=10)
{
    GridFunctionCoefficient psi_coeff(&psi);

    const real_t psimax = psi.Normlinf();
    const real_t volume_proportion = target_volume / domain_volume;

    real_t a = inv_sigmoid(volume_proportion) - psimax; // lower bound of 0
    real_t b = inv_sigmoid(volume_proportion) + psimax; // upper bound of 0

    ConstantCoefficient aCoefficient(a);
    ConstantCoefficient bCoefficient(b);

    SumCoefficient psiA(psi_coeff, aCoefficient);
    SumCoefficient psiB(psi_coeff, bCoefficient);

    TransformedCoefficient sigmoid_psi_a(&psiA, sigmoid);
    TransformedCoefficient sigmoid_psi_b(&psiB, sigmoid);

    ParLinearForm int_sigmoid_psi_a(psi.FESpace());
    int_sigmoid_psi_a.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi_a));
    int_sigmoid_psi_a.Assemble();
    real_t a_vol_minus = int_sigmoid_psi_a.Sum() - target_volume;

    ParLinearForm int_sigmoid_psi_b(psi.FESpace());
    int_sigmoid_psi_b.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi_b));
    int_sigmoid_psi_b.Assemble();
    real_t b_vol_minus = int_sigmoid_psi_b.Sum() - target_volume;

   bool done = false;
   real_t x;
   real_t x_vol;

   for (int k=0; k<max_its; k++) // Illinois iteration
   {
        // cout << "Iteration count " << k << ".\n"; 
        x = b - b_vol_minus * (b - a) / (b_vol_minus - a_vol_minus);

        LinearForm int_sigmoid_psi_x(psi.FESpace());
        ConstantCoefficient xCoefficient(x);
        SumCoefficient psiX(psi_coeff, xCoefficient);
        TransformedCoefficient sigmoid_psi_x(&psiX, sigmoid);
        int_sigmoid_psi_x.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi_x));
        int_sigmoid_psi_x.Assemble();
        x_vol = int_sigmoid_psi_x.Sum();

        real_t x_vol_minus = x_vol - target_volume;

        // cout << "target_volume: " << target_volume << ", domain_volume: " << domain_volume << ", a_vol_minus: " << a_vol_minus << ", b_vol_minus: " << b_vol_minus << " x_vol_minus: " << x_vol_minus << ".\n";

        if (b_vol_minus * x_vol_minus < 0) {
            a = b;
            a_vol_minus = b_vol_minus;   
        } else {
            a_vol_minus = a_vol_minus / 2;
        }
        b = x;
        b_vol_minus = x_vol_minus;

        if (abs(x_vol_minus) < tol) {
            done = true;
            // cout << "x_vol_minus = " << x_vol_minus << " IS WITHINT tolerance of " << tol << ".\n"; 
            break;
        } else {
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
    real_t filter_radius = real_t(0.1);

    real_t alpha = 1.0;
    real_t vol_fraction = 0.5;
    int max_it = 100;
    real_t itol = 1e-1;
    real_t ntol = 1e-4;
    real_t rho_min = 1e-6;
    real_t E_min = 1e-3;
    real_t E_max = 1.0;
    real_t poisson_ratio = 0.2;
    bool glvis_visualization = true;
    bool paraview_output = true;

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
            0;//(int)floor(log(1000. / mesh.GetNE()) / log(2.) / dim);
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
    // set the boundary conditions [1,2,..,7]
    for (int i = 2; i < 8; i++)
    {
        elsolver->AddDispBC(i, 4, 0.0); // start with all of them fixed. 0 displacement in all directions.
    }

    // set surface load
    elsolver->AddSurfLoad(1, 0.0, 1.0); // set the load on the free hole. 0.0 == x direction, 1.0 == y direction.

    // set up variables

    // 9. Define some tools for later.
    ConstantCoefficient zero(0.0);
    ConstantCoefficient one(1.0);
    ParGridFunction onegf(filt->GetDesignFES());
    onegf = 1.0;
    ParGridFunction zerogf(filt->GetDesignFES());
    zerogf = 0.0;
    ParLinearForm vol_form(filt->GetDesignFES());
    vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
    vol_form.Assemble();
    real_t domain_volume = vol_form(onegf);
    const real_t target_volume = domain_volume * vol_fraction;

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
    odens_latent = inv_sigmoid(vol_fraction);
    odens_latent.SetTrueVector();
    odens_latent.GetTrueDofs(odens_latent_vector);

    // Vector odens_vector(filt->GetDesignFES()->TrueVSize());
    ParGridFunction odens_gf = ParGridFunction(filt->GetDesignFES());
    odens_gf.ProjectCoefficient(odens);
    // odens_gf.SetTrueVector();
    // odens_gf.GetTrueDofs(odens_vector);

    // gradient vector in design space
    // true gradient vector on 
    // Vector ograd(odens_gf.GetTrueVector().Size());
    // gradient grid function
    ParGridFunction ogf(filt->GetDesignFES());
    ogf = 0.0;
    // ogf.SetTrueVector();
    // ogf.GetTrueDofs(ograd);

    // ParGridFunction& displacements=elsolver->GetDisplacements();

    Vector fdv(filt->GetFilterFES()->TrueVSize());
    fdv = 0.0; // define the true vector of the filtered field
    
    // set up the paraview
    mfem::ParaViewDataCollection paraview_dc("deterministic_optimizaztion", &pmesh);
    // rho_gf.ProjectCoefficient(rho);
    paraview_dc.SetPrefixPath("ParaView");
    paraview_dc.SetLevelsOfDetail(order);
    paraview_dc.SetDataFormat(VTKFormat::BINARY);
    paraview_dc.SetHighOrderOutput(true);
    paraview_dc.SetCycle(0);
    paraview_dc.SetTime(0.0);
    paraview_dc.RegisterField("density",&odens_gf);
    paraview_dc.RegisterField("filtered_density",&fdens);
    paraview_dc.RegisterField("control_gradient",&ogf);
    paraview_dc.RegisterField("latent_density",&odens_latent);
    // paraview_dc.RegisterField("displacements", &displacements);
    paraview_dc.Save();

    // loop
    for (int k = 1; k <= max_it; k++)
    {
        if (k > 1)
        {
            alpha *= ((real_t)k) / ((real_t)k - 1);
        }
        cout << "\nStep = " << k << endl;

        // project the density
        // odens_gf.ProjectCoefficient(odens);
        // odens_gf.SetTrueVector();
        // odens_gf.GetTrueDofs(odens_vector);

        // filter the density field

        // For Boyan: Replace this with Coefficient [WILL NOT COMPILE]

        filt->Mult(odens_gf, fdv); // apply filt to odv, write into fdv. Can do this with ParGrid functions or VectorCoefficients. Generally better to work in true vectors.


        // set the filtered pargrid function using the filtered true vector
        fdens.SetFromTrueDofs(fdv); // fdens is the ParGridFunction. Basically "algebratizes / makes accessible" dofs
        cout << "\nSet filtration from true dofs." << endl;

        // define material and interpolation parameters coefficient factory
        IsoComplCoef icc;
        // set density, solution, and interpolation parameters for the compliance coefficient factory
        icc.SetGridFunctions(&fdens, &(elsolver->GetDisplacements())); // (1) density (2) displacements. Deferred calculation.
        icc.SetMaterial(E_min, E_max, poisson_ratio);                  // E_min = 1e-3, E_max = 1.0, Poisson Ratio = 0.2
        icc.SetSIMP(3.0);

        // set the material to the elasticity solver
        elsolver->SetMaterial(*(icc.GetE()), *(icc.GetNu())); // take parameters from ICC. It's a coefficient factory. GetE() is a function of density, GetNu() is a constant function (might be a density)
        // solve the discrete elastic system
        elsolver->Assemble();
        elsolver->FSolve(); // solve for displacements

        // extract the solution
        ParGridFunction &sol = elsolver->GetDisplacements();

        // // project the coefficients
        // ParGridFunction egf(filt->GetFilterFES());
        // egf.ProjectCoefficient(*(icc.GetE()));
        // ParGridFunction ngf(filt->GetFilterFES());
        // ngf.ProjectCoefficient(*(icc.GetNu()));

        // // compute the gradients
        // ParGridFunction ggf(filt->GetFilterFES());
        // ggf.ProjectCoefficient(*(icc.GetGradIsoComp())); // Computes gradient of compliance onto filtered density space
        // ParGridFunction cgf(filt->GetFilterFES());
        // cgf.ProjectCoefficient(icc); // Local compliance at every point. Our optimized quantity is the integral of this.

        // // compute compliance
        // double vcompl = 0.0;
        // {
        //     ParLinearForm compl_form(filt->GetFilterFES());
        //     compl_form.AddDomainIntegrator(new DomainLFIntegrator(icc));
        //     compl_form.Assemble();
        //     vcompl = compl_form(onegf); // inner product of compliance and constant 1 function. IE the integral.
        // }
        // if (myid == 0)
        // {
        //     std::cout << "Reference Compliance = " << vcompl << std::endl;
        // }

        // Apply the adjoint filter operation
        filt->AFilter(icc.GetGradIsoComp(), ogf); // adjoint operation to projection to find differential in design space. Chain rule.
        // get the true vector of the gradient
        // ogf.SetTrueVector();
        // ogf.GetTrueDofs(ograd); // extracting the information in the ParGridFunction into the ograd. Lives in dual of design space.
 
        odens_latent.Add(-alpha, ogf);
        real_t material_volume = proj(odens_latent, target_volume, domain_volume, 1e-12, 25); // last two are tol and max its
        // odens_latent.SetTrueVector();
        // odens_latent.GetTrueDofs(odens_latent_vector);

        cout << "Got material " << material_volume << ". Expected Material " << target_volume << "\n";
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

    delete elsolver;
    delete filt;

    Mpi::Finalize();
    return 0;
}
