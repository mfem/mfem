
#include "solvers.hpp"

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
   int order = 2;
   bool mesh_tri = false;
   bool mesh_quad = false;
   int par_ref_levels = 1;
   bool paraview = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&mesh_tri, "-tri", "--triangular", "-no-tri",
                  "--no-triangular", "Enable or not triangular mesh.");
   args.AddOption(&mesh_quad, "-quad", "--quadrilateral", "-no-quad",
                  "--no-quadrilateral", "Enable or not quadrilateral mesh.");
   args.AddOption(&par_ref_levels, "-prl", "--par-ref-levels",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or not Paraview visualization");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   Mesh mesh(mesh_tri ? MESH_TRI : mesh_quad ? MESH_QUAD : mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. We choose
   // 'ref_levels' to be the largest number that gives a final mesh with no
   // more than 1000 elements.
   {
      const int ref_levels =
         (int)floor(log(1000. / mesh.GetNE()) / log(2.) / dim);
      for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }
   }
   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   for (int l = 0; l < par_ref_levels; l++) { pmesh.UniformRefinement(); }

   const int design_dim = 3;
   const int num_angles = 9;
   DenseMatrix V(design_dim, num_angles+1);
   V = 0.0;
   V(0, 0) = 1.0;
   const real_t angle_step = M_PI*2 / (num_angles + 1);
   for (int i=0; i<num_angles; i++)
   {
      const real_t angle = (i+1)*angle_step;
      V(1, i+1) = std::cos(angle);
      V(2, i+1) = std::sin(angle);
   }

   QuadratureSpace design_space(&pmesh, 0);
   QuadratureFunction design_qf(&design_space, design_dim);
   design_qf = 0.0;
   VectorQuadratureFunctionCoefficient design_qf_cf(design_qf);
   PolytopeMirrorCF eta_cf(V, design_qf_cf);

   real_t E_void(1e-06), nu_void(0.3);
   real_t E(1.0), nu(0.3);
   real_t aniso_E(0.5), aniso_Ex(1.0), aniso_nu(0.3);


   // Create the solver
   AnisoLinElasticSolver elsolver(&pmesh, order);
   if (Mpi::Root())
   {
      std::cout << "Number of unknowns: "
                << elsolver.GetSolutionVector().Size() << std::endl;
   }
   elsolver.SetDesignField(eta_cf);
   elsolver.SetIsoMaterials(E_void, nu_void, E, nu);
   elsolver.SetAnisotropicTensor2D(aniso_E, aniso_nu, aniso_Ex);
   elsolver.Assemble();


   return EXIT_SUCCESS;
}
