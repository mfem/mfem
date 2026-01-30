#include "mtop_solvers.hpp"
#include "frac_noise.hpp"

using namespace std;
using namespace mfem;

constexpr auto MESH_TRI = MFEM_SOURCE_DIR "/miniapps/mtop/sq_2D_9_tri.mesh";
constexpr auto MESH_QUAD = MFEM_SOURCE_DIR "/miniapps/mtop/sq_2D_9_quad.mesh";

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init();
   Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = MESH_QUAD;
   const char *device_config = "cpu";
   int order = 1;
   bool pa = false;
   bool dfem = false;
   bool mesh_tri = false;
   bool mesh_quad = false;
   int par_ref_levels = 4;
   int ser_ref_levels = 0;
   bool paraview = false;
   bool visualization = true;
   real_t s=0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&dfem, "-dfem", "--dFEM", "-no-dfem", "--no-dFEM",
                  "Enable or not dFEM.");
   args.AddOption(&mesh_tri, "-tri", "--triangular", "-no-tri",
                  "--no-triangular", "Enable or not triangular mesh.");
   args.AddOption(&mesh_quad, "-quad", "--quadrilateral", "-no-quad",
                  "--no-quadrilateral", "Enable or not quadrilateral mesh.");
   args.AddOption(&par_ref_levels, "-prl", "--par-ref-levels",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&ser_ref_levels, "-srl", "--ser-ref-levels",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or not Paraview visualization");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&s, "-s", "--s", "fractional exponent s in [0,0.5).");
   args.ParseCheck();
   MFEM_VERIFY(!(pa && dfem), "pa and dfem cannot be both set");

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

    // Read the (serial) mesh from the given mesh file on all processors.  We
    // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
    // and volume meshes with the same code.
    Mesh mesh(mesh_tri ? MESH_TRI : mesh_quad ? MESH_QUAD : mesh_file, 1, 1);
    const int dim = mesh.Dimension();
    for (int l = 0; l < ser_ref_levels; l++)
    {
        mesh.UniformRefinement();
    }

    ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();


    // Create the FracRandomFieldGenerator object
    FracRandomFieldGenerator* frac_rng = new FracRandomFieldGenerator(*pmesh, par_ref_levels, order, 1.0, s);

    ParFiniteElementSpace& fes = frac_rng->GetFinestFESpace();
    HYPRE_BigInt size = fes.GlobalTrueVSize();
    if (Mpi::Root())
    {
       cout << "Number of finite element unknowns: " << size << endl;
    }


    int myrank;
    MPI_Comm_rank(pmesh->GetComm(), &myrank); 
    //Set up the parallel random linear form
    unique_ptr<ParLinearForm> b(new ParLinearForm(&fes));
    LinearFormIntegrator *rint = 
      new WhiteGaussianNoiseDomainLFIntegrator(pmesh->GetComm(), 7497+17*myrank);
    b->AddDomainIntegrator(rint);
    b->Assemble();
    unique_ptr<HypreParVector> r(b->ParallelAssemble());
    r->UseDevice(true);

    Vector rfv; 
    rfv.SetSize(fes.GetTrueVSize());


    frac_rng->Mult(*r, rfv);


    ParGridFunction x(&fes);
    x.UseDevice(true);
    x.SetTrueVector(); x.GetTrueVector().UseDevice(true);
    x.SetFromTrueDofs(rfv);
    //Save the solution
    {
       ParaViewDataCollection paraview_dc("rnd",fes.GetParMesh());
       paraview_dc.SetPrefixPath("ParaView");
       paraview_dc.SetLevelsOfDetail(order);
       paraview_dc.SetDataFormat(VTKFormat::BINARY);
       paraview_dc.SetHighOrderOutput(true);
       paraview_dc.SetCycle(0);
       paraview_dc.SetTime(0.0);
       paraview_dc.RegisterField("rnd", &x);
       paraview_dc.Save();
    }

    delete frac_rng;
    delete pmesh;
    return EXIT_SUCCESS;
   
}
   
