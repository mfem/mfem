#include "linear_elasticity.hpp"

using namespace mfem;
using namespace std;

constexpr auto MESH_TRI = MFEM_SOURCE_DIR "/miniapps/mtop/examples/dyn_hex2d_tri.msh";
constexpr auto MESH_QUAD = MFEM_SOURCE_DIR "/miniapps/mtop/examples/dyn_hex2d_quad.msh";


///////////////////////////////////////////////////////////////////////////////
/// \brief The IsoElasticyLambdaCoeff class converts E modulus of elasticity
/// and Poisson's ratio to Lame's lambda coefficient
class IsoElasticyLambdaCoeff : public mfem::Coefficient
{
   mfem::Coefficient *E, *nu;

public:
   /// Constructor - takes as inputs E modulus and Poisson's ratio
   IsoElasticyLambdaCoeff(mfem::Coefficient *E,
                          mfem::Coefficient *nu):
      E(E), nu(nu) { }

   /// Evaluates the Lame's lambda coefficient
   real_t Eval(mfem::ElementTransformation &T,
               const mfem::IntegrationPoint &ip) override
   {
      const real_t EE = E->Eval(T, ip);
      const real_t nn = nu->Eval(T, ip);
      constexpr auto Lambda = [](const real_t E, const real_t nu)
      {
         return E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
      };
      return Lambda(EE, nn);
   }
};



///////////////////////////////////////////////////////////////////////////////
/// \brief The IsoElasticySchearCoeff class converts E modulus of elasticity
/// and Poisson's ratio to Shear coefficient
///
class IsoElasticySchearCoeff : public mfem::Coefficient
{
   mfem::Coefficient *E, *nu;

public:
   /// Constructor - takes as inputs E modulus and Poisson's ratio
   IsoElasticySchearCoeff(mfem::Coefficient *E_, mfem::Coefficient *nu_):
      E(E_), nu(nu_) { }

   /// Evaluates the shear coefficient coefficient
   real_t Eval(mfem::ElementTransformation &T,
               const mfem::IntegrationPoint &ip) override
   {
      const real_t EE = E->Eval(T, ip);
      const real_t nn = nu->Eval(T, ip);
      constexpr auto Schear = [](const real_t E, const real_t nu)
      {
         return E / (2.0 * (1.0 + nu));
      };
      return Schear(EE, nn);
   }
};



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
   bool paraview = true;
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
                  "Number of parallel mesh refinement levels.");
   args.AddOption(&paraview, "-pa", "--paraview", "-no-pa",
                  "--no-paraview", "Enable or not Paraview output.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or not visualization.");

   args.Parse();

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

   // Allocate the time dependent linear elasticity operator
   LinearElasticityTimeDependentOperator lin_elasticity_op(pmesh, order);

   // Set the material coefficients
   ConstantCoefficient rho_coef(0.5); // density coefficient for topology optimization
   
   // Set elasticity coefficients for material 1 and 2
   ConstantCoefficient E1(0.1);
   ConstantCoefficient E2(1.0);
   ConstantCoefficient nu1(0.3);
   ConstantCoefficient nu2(0.3);

   //Lame coefficients
   IsoElasticyLambdaCoeff lambda1(&E1, &nu1);
   IsoElasticySchearCoeff mu1(&E1, &nu1);
   IsoElasticyLambdaCoeff lambda2(&E2, &nu2);
   IsoElasticySchearCoeff mu2(&E2, &nu2);

   // Set density coefficients for material 1 and 2
   ConstantCoefficient dens1_coef(0.5);
   ConstantCoefficient dens2_coef(1.0);

   // Set damping coefficients
   ProductCoefficient cm1_coef(0.02, dens1_coef);
   ProductCoefficient cm2_coef(0.02, dens2_coef);

   ProductCoefficient cl1_coef(0.01, lambda1);
   ProductCoefficient cmu1_coef(0.01, mu1);
   ProductCoefficient cl2_coef(0.01, lambda2);
   ProductCoefficient cmu2_coef(0.01, mu2);

   lin_elasticity_op.SetElasticityCoefficients(lambda1, mu1, lambda2, mu2);
   
   lin_elasticity_op.SetDensityMaterialCoefficients(dens1_coef, dens2_coef);
   
   lin_elasticity_op.SetDampingMaterialCoefficients(cm1_coef, cm2_coef);
   lin_elasticity_op.SetDampingMaterialCoefficients(cl1_coef, cmu1_coef,
                                                    cl2_coef, cmu2_coef);

   lin_elasticity_op.SetDensity(rho_coef);
   
   //set bottom bdr to zero (both the velocities and the displacements)
   lin_elasticity_op.SetZeroBdr(1);

   lin_elasticity_op.AssembleExplicit();


   
   if (paraview)
   {
      ParaViewDataCollection paraview_dc("isoel", &pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      //paraview_dc.RegisterField("disp", &sol);
      paraview_dc.Save();
   }

   return EXIT_SUCCESS;
}