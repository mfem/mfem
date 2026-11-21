// =====================================================================================
//          2D FSI Example: Fluid + Elastic Solid
// -------------------------------------------------------------------------------------
// Sample runs:
//
// 1. FSI (FSI1, RN coupling, with preloading)
// mpirun -np 4 ./navier-fsi-turek -rs 0 --relaxation 1.0 --time-adaptivity -dt 1e-4 --dt-max 1e-3 --aitken -p 1 -fo 4 -eo 2 -sf 10 --preload -pt 2.0 --coupling 1 --enable-stiffening -of ./Output/FSI1-Turek
//
// 2. FSI (FSI2, RN coupling, no preloading)
// mpirun -np 4 ./navier-fsi-turek -rs 0 --relaxation 1.0 --time-adaptivity -dt 1e-4 --dt-max 1e-3 --aitken -p 2 -fo 4 -eo 2 -sf 10 --no-preload --coupling 1 --enable-stiffening -of ./Output/FSI2-Turek
//
// 3. FSI (FSI3, RN coupling, with preloading)
// mpirun -np 4 ./navier-fsi-turek -rs 0 --relaxation 1.0 --time-adaptivity -dt 1e-4 --dt-max 5e-4 --aitken -p 3 -fo 4 -eo 2 -sf 10  --preload -pt 2.0 --coupling 1 --enable-stiffening -of ./Output/FSI3-Turek
//
// 4. CFD only (Rigid beam)
// mpirun -np 4 ./navier-fsi-turek -tf 20 --cfd-only -rs 0 --time-adaptivity -dt 1e-3 -p 1 -sfp 5 -fo 4 -of ./Output/CFD1-Turek
//
// 5. CFD only (Almost rigid beam)
// mpirun -np 4 ./navier-fsi-turek -tf 20 --cfd-almost-rigid -rs 0 -fo 4 -eo 2 --time-adaptivity --dt-max 5e-3 -dt 1e-5 -p 1 -sf 5 -of ./Output/CFD1-Turek-AlmostRigid
//
// 6. CSM only (Beam subject to gravity)
// mpirun -np 4 ./navier-fsi-turek -pt 2.0 --csm-only -rs 0 -dt 1e-3 -sfp 10 -p 1 -of ./Output/CSM-Turek
//
// 2D FSI Benchmark: Flow around a cylinder with an attached elastic beam.
// Setup is described in:
// https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_definitions.html
//
//
//                                     No slip
//             |\     + ----------------------------------------+
//             | \    |       ___                               |
// Parabolic -->  \   |      /   \----------|                   |   Traction free (outflow)
//  inflow   -->  /   |      \___/----------|                   |
//             | /    |                                         |
//             |/     + ----------------------------------------+
//                                     No slip
//

// Navier module
#include "navier_solver.hpp"
#include "kernels/boundary_normal_stress_integrator.hpp"
#include "kernels/boundary_normal_pressure_integrator.hpp"
#include "kernels/boundary_normal_stress_evaluator.hpp"

// Solid mechanics module
#include "structure/elasticity_solver.hpp"

// Mesh deformation module
#include "ale/mesh_laplacian.hpp"
#include "ale/mesh_linear_elasticity.hpp"

// Interface transfer/FSI
#include "interface_transfer.hpp"
#include "aitken.hpp"

// Mesh utilities
#include "fsi_utils.hpp"

// Output
#include <fstream>
#include "FilesystemHelper.hpp"

using namespace mfem;
using namespace navier;
using namespace elasticity;
using namespace fsi;
using namespace common_fsi_utils;

using TransferBackend = InterfaceTransfer::Backend;

// Forward declarations of functions
template <typename T>
void saveArray(const Array<T> &data, const std::string &outfolder, const std::string &name);

void inflow_fun(const Vector &x, real_t time, Vector &v);
void noSlip_fun(const Vector &x, real_t time, Vector &u);
real_t zero_pressure(const Vector &x, real_t t);
void gravitational_force(const Vector &x, real_t time, Vector &u);

void ComputeDragLift(real_t &drag, real_t &lift, ParLinearForm &lift_drag_form);

// Fluid traction sigma_f n_f = tau_n - p n_f at a point of a boundary element, with n_f taken from the boundary
// element itself. A nodal (H1) sigma.n field cannot represent the traction at a corner of the interface (e.g. the
// beam tip corners): the corner node holds the average of the two face tractions, -p (n_1 + n_2)/2 for the
// pressure part, which is wrong on both faces. The pressure is continuous, so p and the (small) viscous traction
// tau_n are transferred as nodal fields and the normal is applied per face.
// - tau_n: viscous traction field; p: pressure in the first component of a vector field (p, 0)
// - normal_sign: +1 if the boundary element normal is n_f (fluid mesh), -1 if it is n_s = -n_f (solid mesh)
// - scale: overall factor (e.g. -1 for the NavierSolver Robin datum, which expects sigma_s n_s = -sigma_f n_f)
class FaceNormalTractionCoefficient : public VectorCoefficient
{
   const GridFunction &tau_n, &p;
   const real_t normal_sign, scale;
   Vector pv, nor;

public:
   FaceNormalTractionCoefficient(const GridFunction &tau_n_, const GridFunction &p_, real_t normal_sign_,
                                 real_t scale_ = 1.0)
      : VectorCoefficient(tau_n_.VectorDim()), tau_n(tau_n_), p(p_), normal_sign(normal_sign_), scale(scale_) {}

   void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip) override
   {
      MFEM_ASSERT(T.ElementType == ElementTransformation::BDR_ELEMENT,
                  "FaceNormalTractionCoefficient: needs a boundary element transformation");
      T.SetIntPoint(&ip);
      nor.SetSize(vdim);
      CalcOrtho(T.Jacobian(), nor);
      nor /= nor.Norml2();
      tau_n.GetVectorValue(T, ip, V);
      p.GetVectorValue(T, ip, pv);
      V.Add(-normal_sign * pv(0), nor);
      V *= scale;
   }
};

enum class CouplingType : int
{
   Dirichlet_Neumann = 0,
   Robin_Neumann = 1,
   RobinAMP_Neumann = 2
};

struct s_NavierContext // Navier Stokes params
{
   int u_order = 2;                   // polynomial order of the velocity space
   int max_bdf_order = 3;             // maximum BDF order for time integration
   real_t density = 1e3;              // Density of the fluid
   real_t dynamic_viscosity = 1e-3;   // Dynamic viscosity of the fluid
   real_t kinematic_viscosity = 1e-6; // Kinematic viscosity of the fluid
   real_t Umax = 0.2;                 // Maximum velocity at the inlet
   real_t preloadT = 2.0;             // Preload time
   real_t scaling = 35.69303985;      // Scaling factor for the inflow velocity 1.5 * 4.0/Mesh_ctx.D
   int filter_type = 0;               // Enable filtering. 0 = None, 1 = HPFRT_LOS, 2 = HPFRT_LPOLY
} NS_ctx;

struct s_ElasticityContext // elasticity params
{
   int order = 2;        // Polynomial order of the elasticity space
   real_t density = 1e3; // Density
   real_t nu = 0.4;      // Poisson's ratio
   real_t E = 2.0e6;     // Young's modulus
   real_t mu;            // G, Shear modulus, Lame's second parameter
   real_t lambda;        // Lame's first parameter
   // const real_t g = 9.80665; // m/s² (standard gravity)
   const real_t g = 2.0;
} Elasticity_ctx;

struct s_MeshContext // mesh
{
   real_t D = 0.41;  // Channel diameter [recomputed from the mesh bounding box]
   real_t x0 = 0.0;  // x-coordinate of the inlet [from the mesh bounding box]
   real_t y0 = 0.0;  // y-coordinate of the bottom wall [from the mesh bounding box]
   real_t tip_x = 0.6; // Beam tip, point A of the benchmark [from the solid bounding box]
   real_t tip_y = 0.2;
   real_t L = 2.2;   // Channel length
   real_t BL = 0.35; // Elastic beam length
   real_t BT = 0.02; // Elastic beam thickness
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   int order = -1;  // polynomial order of the mesh nodes
   real_t E = 1e6;  // Young's modulus for the mesh deformation
   real_t nu = 0.35; // Poisson's ratio for the mesh deformation
   int stiffening_type = 1; // 0: Jacobian-based, 1: Distance-based
   bool enable_stiffening = false;
   MeshDeformationType mesh_deformation_type = MeshDeformationType::LinearElasticity; // Default is LinearElasticity
} Mesh_ctx;

struct s_DomainDecompositionContext // domain decomposition params
{
   CouplingType coupling = CouplingType::Dirichlet_Neumann; // Default coupling type
   bool aitken = false;         // Enable Aitken relaxation
   int robin_traction = 0;      // Traction in the fluid Robin datum: 0 - fluid traction (consistent), 1 - solid sigma_s(u_h).n
   bool face_traction = false;  // Fluid traction as tau.n - p n_face (face normal), instead of a nodal sigma.n field
   real_t safety_factor = 1.15; // Safety factor for Aitken relaxation
   real_t omega = 0.5;
} DD_ctx;


const char *outfolder = "./Output/FSI-Turek";
int num_procs;

real_t dt;

int main(int argc, char *argv[])
{

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 1. Initialize MPI and Hypre
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Mpi::Init(argc, argv);
   Hypre::Init();

   num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 2. Parse command-line options.
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Mesh file
   const char *mesh_file = "../../data/turek-fsi-new.e";

   // FSI test
   int test_type = 0;             // Determines the set of parameters used for the simulation. 1-3 = FSI/CFD
   bool cfd_rigid = false;        // If true, the beam is fully rigid (we just solve the preload phase)
   bool cfd_almost_rigid = false; // If true, the beam is almost rigid. We solve the coupled FSI but the beam is very stiff.
   bool csm_only = false;         // If true, we just solve the solid mechanics problem

   // Timestepping parameters
   real_t current_time = 0.0;
   real_t t_final = 10;
   dt = 1e-5;
   real_t dt_preload = 1e-3; // Time step for the preload phase
   real_t dt_max = 1e-2;
   real_t cfl_target = 0.8;
   real_t cfl_tol = 1e-1;
   bool time_adaptivity = false; // Enable time adaptivity
   bool enable_preload = true;   // Enable preload phase --> otherwise we start with the FSI problem directly (but still preloading the inflow)

   // Coupling parameters
   real_t relaxation = 1.0;


   // Output parameters
   int save_freq = 1;
   int save_freq_preload = 1;

   int current_step = 0;
   bool last_step = false;

   bool verbose = false; // verbose output during time integration

   OptionsParser args(argc, argv);
   // FSI test
   args.AddOption(&test_type, "-p", "--problem-type",
                  "FSI parameter set: 1, 2, 3");
   args.AddOption(&cfd_rigid, "-cfd-only", "--cfd-only", "-no-cfd-only", "--no-cfd-only",
                  "Run only the CFD part of the simulation (no solid mechanics).");
   args.AddOption(&cfd_almost_rigid, "-cfd-almost-rigid", "--cfd-almost-rigid", "-no-cfd-almost-rigid", "--no-cfd-almost-rigid",
                  "Run the CFD part of the simulation with an almost rigid beam.");
   args.AddOption(&csm_only, "-csm-only", "--csm-only", "-no-csm-only", "--no-csm-only",
                  "Run only the solid mechanics part of the simulation (no CFD).");
   // FE
   args.AddOption(&NS_ctx.u_order, "-fo", "--fluid-order", "polynomial_order for fluid velocity space");
   args.AddOption(&Elasticity_ctx.order, "-eo", "--elasticity-order",
                  "polynomial_order for solid elasticity space");
   // Navier Stokes time integration
   args.AddOption(&NS_ctx.max_bdf_order, "-mbdf", "--max_bdf_order", "max_bdf_order");
   args.AddOption(&time_adaptivity, "-ta", "--time-adaptivity", "-no-ta", "--no-time-adaptivity",
                  "Enable or disable time adaptivity.");
   args.AddOption(&NS_ctx.Umax, "-inflow_velocity", "--inflow_velocity",
                  "inflow_velocity");
   args.AddOption(&NS_ctx.filter_type, "-filter", "--filter",
                  "Enable filtering. 0 = None, 1 = HPFRT_LOS, 2 = HPFRT_LPOLY");
   // Elasticity parameters
   args.AddOption(&Elasticity_ctx.E, "-E", "--young-modulus",
                  "Young's modulus (default: 2.0e6).");
   args.AddOption(&Elasticity_ctx.nu, "-nu", "--poisson-ratio",
                  "Poisson's ratio (default: 0.4).");
   args.AddOption(&Elasticity_ctx.density, "-rho-s", "--rho-solid",
                  "Density of the solid (default: 1e3).");
   // Navier Stokes parameters
   args.AddOption(&NS_ctx.density, "-rho-f", "--rho-fluid",
                  "Density of the fluid (default: 1e3).");
   args.AddOption(&NS_ctx.dynamic_viscosity, "-visc", "--viscosity",
                  "Dynamic viscosity of the fluid (default: 1e-3).");
   // Time integration parameters
   args.AddOption(&t_final, "-tf", "--t_final", "t_final");
   args.AddOption(&dt, "-dt", "--dt", "dt.");
   args.AddOption(&dt_max, "-dtmax", "--dt-max", "Maximum time step size.");
   args.AddOption(&dt_preload, "-dtp", "--dt-preload",
                  "Time step for the preload phase (default: 1e-3).");
   args.AddOption(&NS_ctx.preloadT, "-pt", "--preload-time",
                  "Preload time.");
   args.AddOption(&enable_preload, "-preload", "--preload", "-no-preload", "--no-preload",
                  "Enable or disable the preload phase (default: true).");
   // FSI coupling parameters
   args.AddOption((int *)&DD_ctx.coupling, "-c", "--coupling",
                  "Coupling type: 0 - Dirichlet_Neumann, 1 - Robin_Neumann, 2 - RobinAMP_Neumann.");
   args.AddOption(&relaxation, "-rel", "--relaxation",
                  "Relaxation factor for the fluid and elastic problem (default: 1.0).");
   args.AddOption(&DD_ctx.aitken, "-aa", "--aitken", "-no-aa", "--no-aitken",
                  "Enable or disable Aitken relaxation.");
   args.AddOption(&DD_ctx.robin_traction, "-rt", "--robin-traction",
                  "Traction used in the fluid Robin datum (Robin couplings): 0 - fluid traction sigma_f.n of the "
                  "previous sub-iterate, i.e. the load applied to the solid (default; at convergence the Robin "
                  "condition reduces to u_f = u_s), 1 - solid traction sigma_s(u_h).n from the displacement gradient.");
   args.AddOption(&DD_ctx.face_traction, "-ft", "--face-traction", "-no-ft", "--no-face-traction",
                  "Fluid traction on the interface (solid load and Robin datum with -rt 0) as tau.n - p n with the "
                  "face normal (correct at the corners of the interface), or as a nodal sigma.n field.");
   args.AddOption(&DD_ctx.safety_factor, "-sfa", "--safety-factor",
                  "Safety factor for the Aitken relaxation method (default: 1.15). ");
   // Mesh parameters
   args.AddOption(&Mesh_ctx.serial_ref_levels, "-rs", "--rs", "serial_ref_levels");
   args.AddOption(&Mesh_ctx.parallel_ref_levels, "-pr", "--pr", "parallel_ref_levels");
   args.AddOption((int *)&Mesh_ctx.mesh_deformation_type, "-mdt", "--mesh-deformation-type",
                  "Type of mesh deformation to use (0 - Laplacian, 1 - IncrementalLaplacian, 2 - Elasticity). "
                  "Default is 0 (Laplacian).");
   args.AddOption(&Mesh_ctx.order, "-mo", "--mesh-order",
                  "Polynomial order of the mesh nodes (default: 2).");
   args.AddOption(&Mesh_ctx.enable_stiffening, "-es", "--enable-stiffening", "-no-es", "--no-enable-stiffening",
                  "Enable mesh local stiffening (default: false).");
   args.AddOption(&Mesh_ctx.stiffening_type, "-st", "--stiffening-type",
                  "Type of mesh stiffening to use (0 - Jacobian-based, 1 - Distance-based). "
                  "Default is 1 (Jacobian-based).");
   // Output parameters
   args.AddOption(&save_freq, "-sf", "--save-freq",
                  "Save frequency.");
   args.AddOption(&save_freq_preload, "-sfp", "--save-freq-preload",
                  "Save frequency during preload phase (default: 1).");
   args.AddOption(&outfolder, "-of", "--output-folder",
                  "Output folder.");
   args.AddOption(&verbose, "-v", "--verbose", "-no-v", "--no-verbose",
                  "Enable verbose output during time integration (default: false).");
   args.ParseCheck();

   // If FSI test is provided we override the parameters given from CLI
   switch (test_type)
   {
   case 1: // FSI1
      // Fluid
      NS_ctx.density = 1e3;                                                   // Density of the fluid [kg/m^3]
      NS_ctx.kinematic_viscosity = 1e-3;                                      // Kinematic viscosity of the fluid [m^2/s]
      NS_ctx.dynamic_viscosity = NS_ctx.density * NS_ctx.kinematic_viscosity; // Dynamic viscosity of the fluid [Pa.s]
      NS_ctx.Umax = 0.2;                                                      // Maximum velocity at the inlet [m/s]
      // Solid
      Elasticity_ctx.density = 1e3; // Density of the solid [kg/m^3]
      Elasticity_ctx.E = 1.4e6;     // Young's modulus of the solid [Pa] = 1.4 MPa --> mu = 0.5 MPa
      Elasticity_ctx.nu = 0.4;      // Poisson's ratio of the solid
      break;
   case 2: // FSI2
      // Fluid
      NS_ctx.density = 1e3;                                                   // Density of the fluid [kg/m^3]
      NS_ctx.kinematic_viscosity = 1e-3;                                      // Kinematic viscosity of the fluid [m^2/s]
      NS_ctx.dynamic_viscosity = NS_ctx.density * NS_ctx.kinematic_viscosity; // Dynamic viscosity of the fluid [Pa.s]
      NS_ctx.Umax = 1.0;                                                      // Maximum velocity at the inlet [m/s]
      // Solid
      Elasticity_ctx.density = 1e4; // Density of the solid [kg/m^3]
      Elasticity_ctx.E = 1.4e6;     // Young's modulus of the solid [Pa] = 1.4 MPa --> mu = 0.5 MPa
      Elasticity_ctx.nu = 0.4;      // Poisson's ratio of the solid
      break;
   case 3: // FSI3
      // Fluid
      NS_ctx.density = 1e3;                                                   // Density of the fluid [kg/m^3]
      NS_ctx.kinematic_viscosity = 1e-3;                                      // Kinematic viscosity of the fluid [m^2/s]
      NS_ctx.dynamic_viscosity = NS_ctx.density * NS_ctx.kinematic_viscosity; // Dynamic viscosity of the fluid [Pa.s]
      NS_ctx.Umax = 2.0;                                                      // Maximum velocity at the inlet [m/s]
      // Solid
      Elasticity_ctx.density = 1e3; // Density of the solid [kg/m^3]
      Elasticity_ctx.E = 5.6e6;     // Young's modulus of the solid [Pa] = 5.6 MPa --> mu = 2 MPa
      Elasticity_ctx.nu = 0.4;      // Poisson's ratio of the solid
      break;
   default:
      break;
   }

   if (cfd_almost_rigid)
   {
      Elasticity_ctx.E = 1e12; // Set a very high Young's modulus to make the beam almost rigid
      Elasticity_ctx.density = 1e6;
      enable_preload = false; // Disable preload phase for almost rigid beam
   }

   // Compute Lame's parameters
   Elasticity_ctx.mu = Elasticity_ctx.E / (2.0 * (1.0 + Elasticity_ctx.nu));                                                     // G, Shear modulus, Lame's second parameter
   Elasticity_ctx.lambda = Elasticity_ctx.E * Elasticity_ctx.nu / ((1.0 + Elasticity_ctx.nu) * (1.0 - 2.0 * Elasticity_ctx.nu)); // Lame's first parameter

   // Set the relaxation factors for the fluid and solid problems
   // If Aitken relaxation is enabled, these factors will be used as the initial omega values at each timestep
   DD_ctx.omega = relaxation; // Relaxation factor for the fluid problem
   DD_ctx.omega = relaxation; // Relaxation factor for the solid problem

   // Check that only one restricted test has been selected
   MFEM_VERIFY(!(cfd_rigid && cfd_almost_rigid) &&
                   !(cfd_rigid && csm_only) &&
                   !(cfd_almost_rigid && csm_only),
               "You can only select one of the following options: --cfd-only, --cfd-almost-rigid, --csm-only");

   // Check test type
   if (cfd_rigid)
   {
      dt_preload = dt;
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create serial Mesh, parallel and fluid/solid sub-meshes
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\nLoading mesh and refining in serial...";

   Mesh *parent_mesh = new Mesh(mesh_file);
   const int dim = parent_mesh->Dimension();

   if (Mpi::Root())
   {
      mfem::out << "Refining serial mesh...";
   }

   // 1. Refine serial mesh
   for (int i = 0; i < Mesh_ctx.serial_ref_levels; i++)
   {
      parent_mesh->UniformRefinement();
   }

   // Create the sub-domains for the beam and fluid domains
   AttributeSets &attr_sets = parent_mesh->attribute_sets;
   AttributeSets &bdr_attr_sets = parent_mesh->bdr_attribute_sets;

   Mesh_ctx.order = Mesh_ctx.order < 0 ? Elasticity_ctx.order : Mesh_ctx.order;
   parent_mesh->SetCurvature(Mesh_ctx.order, false);

   if (Mpi::Root())
   {
      mfem::out << " done." << std::endl;
      mfem::out << "Partitioning and creating parallel mesh..." << std::endl;
   }

   // Generate mesh partitioning and create a parallel mesh.
   int partition_type = 1;
   int np = num_procs;
   int *partitioning = parent_mesh->GeneratePartitioning(np, partition_type);

   // Create parallel mesh
   ParMesh pmesh(MPI_COMM_WORLD, *parent_mesh, partitioning, partition_type);
   ExportMeshwithPartitioning(outfolder, *parent_mesh, partitioning, "Partitioning"); // filename not defined
   delete[] partitioning;

   for (int l = 0; l < Mesh_ctx.parallel_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   pmesh.SetAttributes();
   pmesh.EnsureNodes();

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
      mfem::out << "Creating sub-meshes...";
   }

   // Extract parallel submeshes
   Array<int> solid_domain_attribute;
   Array<int> fluid_domain_attribute;
   solid_domain_attribute = attr_sets.GetAttributeSet("beam");
   fluid_domain_attribute = attr_sets.GetAttributeSet("fluid");

   auto solid_pmesh = ParSubMesh::CreateFromDomain(pmesh, solid_domain_attribute);
   auto fluid_pmesh = ParSubMesh::CreateFromDomain(pmesh, fluid_domain_attribute);

   // The benchmark geometry can be placed anywhere (e.g. turek-fsi.e has the
   // inlet/bottom wall corner at the origin, turek-fsi-new.e is centered), so
   // take the origin and the channel height used by the inflow profile, and
   // the probe points, from the mesh bounding boxes.
   {
      Vector bb_min, bb_max;
      pmesh.GetBoundingBox(bb_min, bb_max);
      Mesh_ctx.x0 = bb_min(0);
      Mesh_ctx.y0 = bb_min(1);
      Mesh_ctx.D = bb_max(1) - bb_min(1);
      NS_ctx.scaling = 1.5 * 4.0 / (Mesh_ctx.D * Mesh_ctx.D);

      // Beam tip: midpoint of the free end, moved slightly inside the
      // (reference) solid mesh so that the point search is robust.
      solid_pmesh.GetBoundingBox(bb_min, bb_max);
      Mesh_ctx.tip_x = bb_max(0) - 1e-8 * (bb_max(0) - bb_min(0));
      Mesh_ctx.tip_y = 0.5 * (bb_min(1) + bb_max(1));
      if (Mpi::Root())
      {
         mfem::out << "Channel: inlet at x = " << Mesh_ctx.x0 << ", bottom wall at y = "
                   << Mesh_ctx.y0 << ", height D = " << Mesh_ctx.D
                   << "; beam tip probe at (" << Mesh_ctx.tip_x << ", " << Mesh_ctx.tip_y
                   << ")" << std::endl;
      }
   }

   //////////////////
   // SET MARKERS
   //////////////////

   // Setup boundary markers
   // If we use the parent mesh size we're sure that these markers will be compatible with the different submeshes
   Array<int> fluid_inlet = bdr_attr_sets.GetAttributeSet("inlet");
   Array<int> fluid_outlet = bdr_attr_sets.GetAttributeSet("outlet");
   Array<int> fluid_top_wall = bdr_attr_sets.GetAttributeSet("top_wall");
   Array<int> fluid_bottom_wall = bdr_attr_sets.GetAttributeSet("bottom_wall");
   Array<int> fluid_cylinder = bdr_attr_sets.GetAttributeSet("cylinder");

   // Interface attributes for the solid-fluid interface in parent mesh, used in  BidirectionalInterfaceTransfer
   // @NOTE: if we don't use marker from the parent mesh there might be compatibility issues with the markers depending on the direction of the transfer
   Array<int> beam_cylinder_curve = bdr_attr_sets.GetAttributeSet("beam_cylinder_curve");
   Array<int> beam_sides = bdr_attr_sets.GetAttributeSet("beam_sides");   
   Array<int> beam_tip = bdr_attr_sets.GetAttributeSet("beam_tip");

   // Combine sides and tip to have a single marker for the whole beam boundary 
   Array<int> fluid_solid_interface(2); 
   fluid_solid_interface[0] = beam_sides[0];
   fluid_solid_interface[1] = beam_tip[0];
   Array<int> fluid_solid_interface_marker = bdr_attr_sets.AttrToMarker(parent_mesh->bdr_attributes.Max(), fluid_solid_interface); 
   Array<int> fluid_solid_interface_tip_marker = bdr_attr_sets.AttrToMarker(parent_mesh->bdr_attributes.Max(), beam_tip);

   delete parent_mesh; // we don't need the parent mesh anymore

   if (Mpi::Root())
      mfem::out << "done." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Create Flow, Solid, and MeshDeformation Solvers
   ///////////////////////////////////////////////////////////////////////////////////////////////

   ///////////////
   // FLOW
   ///////////////

   if (Mpi::Root())
      mfem::out << "\nCreating Navier-Stokes solver..." << std::endl;

   // Print fluid properties
   if (Mpi::Root())
   {
      mfem::out << "Fluid properties:" << std::endl;
      mfem::out << "  - Density: " << std::scientific << NS_ctx.density << " kg/m³" << std::endl;
      mfem::out << "  - Kinematic viscosity: " << std::scientific << NS_ctx.kinematic_viscosity << " m²/s" << std::endl;
   }

   NavierSolver navier(&fluid_pmesh, NS_ctx.u_order, NS_ctx.kinematic_viscosity, NS_ctx.density, false);
   navier.SetMaxBDFOrder(NS_ctx.max_bdf_order);

   // Enable filtering if requested
   switch (NS_ctx.filter_type)
   {
   case 0: // No filtering
      break;
   case 1: // HPFRT_LOS
      navier.EnableFilter(NavierSolver::FilterMethod::HPFRT_LOS);
      navier.SetCutoffModes(1);
      navier.SetFilterAlpha(0.5);
      break;
   case 2: // HPFRT_LPOLY
      navier.EnableFilter(NavierSolver::FilterMethod::HPFRT_LPOLY);
      navier.SetCutoffModes(1);
      navier.SetFilterAlpha(10.0);
      break;
   default:
      MFEM_ABORT("Invalid filter type. Use 0 for None, 1 for HPFRT_LOS, or 2 for HPFRT_LPOLY.");
   }

   ParGridFunction *u_gf = navier.GetCurrentVelocity();
   ParGridFunction *un_next_gf = navier.GetProvisionalVelocity();
   ParGridFunction *p_gf = navier.GetCurrentPressure();
   ParGridFunction *nu_gf = navier.GetVariableViscosity();

   Vector u_fluid, p_fluid;
   Vector u_fluid_tn, p_fluid_tn;
   u_gf->GetTrueDofs(u_fluid);
   p_gf->GetTrueDofs(p_fluid);
   u_gf->GetTrueDofs(u_fluid_tn);
   p_gf->GetTrueDofs(p_fluid_tn);

   ParGridFunction solid_vel_onFluid_gf(u_gf->ParFESpace());
   solid_vel_onFluid_gf = 0.0;
   ParGridFunction solid_sigmaN_onFluid_gf(u_gf->ParFESpace());
   solid_sigmaN_onFluid_gf = 0.0;
   // Face-normal traction (--face-traction): viscous traction and pressure of the fluid on the fluid interface (as
   // computed by the forward transfer), and the Robin datum built from them (relaxed as solid_sigmaN_onFluid_gf)
   ParGridFunction fluid_tauN_onFluid_gf(u_gf->ParFESpace()), fluid_p_onFluid_gf(u_gf->ParFESpace());
   ParGridFunction datum_tauN_onFluid_gf(u_gf->ParFESpace()), datum_p_onFluid_gf(u_gf->ParFESpace());
   fluid_tauN_onFluid_gf = 0.0;
   fluid_p_onFluid_gf = 0.0;
   datum_tauN_onFluid_gf = 0.0;
   datum_p_onFluid_gf = 0.0;

   int order_quad_u_fluid = std::max(2, 2 * NS_ctx.u_order + 2);
   const IntegrationRule *irs_vel[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs_vel[i] = &(IntRules.Get(i, order_quad_u_fluid));
   }

   /////////////
   // SOLID
   /////////////

   if (Mpi::Root())
      mfem::out << "\nCreating Elasticity solver..." << std::endl;

   // Print mechanical properties
   if (Mpi::Root())
   {
      mfem::out << "Elasticity properties:" << std::endl;
      mfem::out << "  - Young's modulus: " << std::scientific << Elasticity_ctx.E << " Pa" << std::endl;
      mfem::out << "  - Poisson's ratio: " << Elasticity_ctx.nu << " (dimensionless)" << std::endl;
      mfem::out << "  - Density: " << std::scientific << Elasticity_ctx.density << " kg/m³" << std::endl;
      mfem::out << "  - Shear modulus (mu): " << std::scientific << Elasticity_ctx.mu << " Pa" << std::endl;
      mfem::out << "  - Lame's first parameter (lambda): " << std::scientific << Elasticity_ctx.lambda << " Pa" << std::endl;
   }

   // Create the elasticity solver
   int solver_type = 15; // HHTAlphaSolver
   ElasticitySolver elasticity(solver_type, &solid_pmesh, Elasticity_ctx.order, Elasticity_ctx.density, Elasticity_ctx.lambda, Elasticity_ctx.mu);

   // Grid functions and vectors for the solid
   Vector solid_displacement, solid_displacement_tn;
   Vector solid_vel, solid_vel_tn;

   auto elasticity_fes = elasticity.GetFESpace();

   ParGridFunction solid_displacement_gf(elasticity_fes);
   solid_displacement_gf = 0.0;
   solid_displacement_gf.GetTrueDofs(solid_displacement);
   solid_displacement_gf.GetTrueDofs(solid_displacement_tn);

   ParGridFunction solid_vel_gf(elasticity_fes);
   solid_vel_gf = 0.0;
   solid_vel_gf.GetTrueDofs(solid_vel);
   solid_vel_gf.GetTrueDofs(solid_vel_tn);

   ParGridFunction fluid_sigmaN_onSolid_gf(elasticity_fes);
   fluid_sigmaN_onSolid_gf = 0.0;
   ParGridFunction fluid_tauN_onSolid_gf(elasticity_fes), fluid_p_onSolid_gf(elasticity_fes); // --face-traction
   fluid_tauN_onSolid_gf = 0.0;
   fluid_p_onSolid_gf = 0.0;

   int order_quad_disp = std::max(2, 2 * Elasticity_ctx.order + 2);
   const IntegrationRule *irs_disp[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs_disp[i] = &(IntRules.Get(i, order_quad_disp));
   }

   /////////////////////////
   // MESH DEFORMATION
   /////////////////////////

   if (Mpi::Root())
      mfem::out << "Creating mesh deformation solver...";

   auto vfec = H1_FECollection(2, fluid_pmesh.Dimension());
   auto vfes = ParFiniteElementSpace(&fluid_pmesh, &vfec, fluid_pmesh.Dimension());

   MeshDeformation *fluid_mesh_deformation = nullptr;

   switch (Mesh_ctx.mesh_deformation_type)
   {
   case MeshDeformationType::Laplacian: // Laplacian
      fluid_mesh_deformation = new MeshLaplacian(&fluid_pmesh);
      break;
   case MeshDeformationType::IncrementalLaplacian: // Incremental Laplacian
      fluid_mesh_deformation = new MeshLaplacian(&fluid_pmesh, true);
      break;
   case MeshDeformationType::LinearElasticity: // Elasticity
      fluid_mesh_deformation = new MeshLinearElasticity(&fluid_pmesh);
      break;
   default:
      MFEM_ABORT("Invalid mesh deformation type.");
   }

   std::vector<std::string> mesh_deformation_names = {"Laplacian", "IncrementalLaplacian", "LinearElasticity"};

   if (Mpi::Root())
      mfem::out << "\nUsing mesh deformation type: " << mesh_deformation_names[static_cast<int>(Mesh_ctx.mesh_deformation_type)] << std::endl;

   ParFiniteElementSpace *mesh_nodes_fes = fluid_mesh_deformation->GetFESpace();

   ParGridFunction *mesh_velocity_gf = new ParGridFunction(mesh_nodes_fes); // Mesh velocity field
   *mesh_velocity_gf = 0.0;
   // ALE convection (u - w)·grad u: the BDF time derivative is taken at the (moving) mesh nodes, so the fluid
   // solver needs the mesh velocity w. Must be set before navier.Setup(), which builds the convection integrator.
   navier.SetMeshVelocity(mesh_velocity_gf, true); // navier owns mesh_velocity_gf

   ParGridFunction *mesh_displacement_gf = new ParGridFunction(mesh_nodes_fes);
   VectorGridFunctionCoefficient *mesh_displacement_coeff = new VectorGridFunctionCoefficient(&solid_displacement_gf);
   *mesh_displacement_gf = 0.0;

   Vector mesh_displacement_tn;
   mesh_displacement_gf->GetTrueDofs(mesh_displacement_tn);

   ParGridFunction *solid_relative_displacement_gf = nullptr; // Relative solid displacement field (on solid fes)
   if (!fluid_mesh_deformation->IsNonIncremental())
   {
      solid_relative_displacement_gf = new ParGridFunction(elasticity_fes); // Relative solid displacement field
      *solid_relative_displacement_gf = 0.0;
   }

   if (Mpi::Root())
      mfem::out << "done." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Add boundary conditions and initial conditions
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\nSetting up boundary conditions and initial conditions..." << std::endl;

   ////////////
   // FLUID
   ////////////
   // - Dirichlet on inlet (parabolic profile)
   // - Pressure on outlet (p = 0)
   // - No slip on Γ = top/bottom wall and cylinder (u = 0)
   // - Dirichlet/Robin on interface Γfs

   Array<int> fluid_inlet_attr = AttributeSets::AttrToMarker(fluid_pmesh.bdr_attributes.Max(), fluid_inlet);
   Array<int> fluid_outlet_attr = AttributeSets::AttrToMarker(fluid_pmesh.bdr_attributes.Max(), fluid_outlet);
   Array<int> fluid_top_wall_attr = AttributeSets::AttrToMarker(fluid_pmesh.bdr_attributes.Max(), fluid_top_wall);
   Array<int> fluid_bottom_wall_attr = AttributeSets::AttrToMarker(fluid_pmesh.bdr_attributes.Max(), fluid_bottom_wall);
   Array<int> fluid_cylinder_attr = AttributeSets::AttrToMarker(fluid_pmesh.bdr_attributes.Max(), fluid_cylinder);
   Array<int> fluid_interface_attr = AttributeSets::AttrToMarker(fluid_pmesh.bdr_attributes.Max(), fluid_solid_interface);

   navier.AddVelDirichletBC(inflow_fun, fluid_inlet_attr);
   navier.AddPresDirichletBC(zero_pressure, fluid_outlet_attr);
   navier.AddVelDirichletBC(noSlip_fun, fluid_top_wall_attr);
   navier.AddVelDirichletBC(noSlip_fun, fluid_bottom_wall_attr);
   navier.AddVelDirichletBC(noSlip_fun, fluid_cylinder_attr);

   // Add dirichlet at the interface
   VectorGridFunctionCoefficient solid_vel_onFluid_coeff(&solid_vel_onFluid_gf); // Holds the solid velocity projected onto the fluid mesh
   VectorGridFunctionCoefficient solid_sigmaN_onFluid_nodal_coeff(&solid_sigmaN_onFluid_gf);
   // Robin datum sigma_s n_s = -(tau_n - p n_f), with the fluid face normal n_f
   FaceNormalTractionCoefficient solid_sigmaN_onFluid_face_coeff(datum_tauN_onFluid_gf, datum_p_onFluid_gf, 1.0, -1.0);
   const bool face_robin_datum = DD_ctx.face_traction && DD_ctx.robin_traction == 0;
   VectorCoefficient &solid_sigmaN_onFluid_coeff = face_robin_datum
                                                    ? static_cast<VectorCoefficient &>(solid_sigmaN_onFluid_face_coeff)
                                                    : solid_sigmaN_onFluid_nodal_coeff;
   real_t cp =std::sqrt((Elasticity_ctx.lambda + 2 * Elasticity_ctx.mu) / Elasticity_ctx.density); // compression wave speed
   real_t cs = std::sqrt(Elasticity_ctx.mu / Elasticity_ctx.density);                               // shear wave speed
   real_t zp = Elasticity_ctx.density * cp;
   real_t zs = Elasticity_ctx.density * cs;

   switch (DD_ctx.coupling)
   {
   case CouplingType::Dirichlet_Neumann:
      navier.AddVelDirichletBC(&solid_vel_onFluid_coeff, fluid_interface_attr, false);
      break;
   case CouplingType::Robin_Neumann:
      navier.AddRobinBC(zp, &solid_vel_onFluid_coeff, &solid_sigmaN_onFluid_coeff, fluid_interface_attr, false);
      break;
   case CouplingType::RobinAMP_Neumann:
      navier.AddAMPRobinBC(zs, zp, &solid_vel_onFluid_coeff, &solid_sigmaN_onFluid_coeff, fluid_interface_attr, false);
      break;
   default:
      break;
   }

   // Setup solver
   navier.Setup(dt);

   ////////////
   // SOLID
   ////////////
   // - Fully Constrained at the cylinder base 
   // - Neumann (traction) on interface Γfs (σ·n = -σ_f · n) --> negative sign because n_f = -n_s
   VectorGridFunctionCoefficient fluid_sigmaN_onSolid_nodal_coeff(&fluid_sigmaN_onSolid_gf);
   // sigma_f n_f = tau_n - p n_f, with n_f = -n_s from the solid face normal
   FaceNormalTractionCoefficient fluid_sigmaN_onSolid_face_coeff(fluid_tauN_onSolid_gf, fluid_p_onSolid_gf, -1.0);
   VectorCoefficient &fluid_sigmaN_onSolid_coeff = DD_ctx.face_traction
                                                    ? static_cast<VectorCoefficient &>(fluid_sigmaN_onSolid_face_coeff)
                                                    : fluid_sigmaN_onSolid_nodal_coeff;
   elasticity.AddFixedConstraint(beam_cylinder_curve[0]);
   elasticity.AddBoundaryLoad(&fluid_sigmaN_onSolid_coeff, beam_sides[0], -1.0, false);
   elasticity.AddBoundaryLoad(&fluid_sigmaN_onSolid_coeff, beam_tip[0], -1.0, false);

   if (csm_only)
   {
      elasticity.AddBodyForce(gravitational_force, solid_domain_attribute[0]);
   }

   elasticity.Setup();

   ////////////////////
   // MESH DEFORMATION
   ////////////////////

   Array<int> fluid_mesh_deformation_ess_bdr(fluid_pmesh.bdr_attributes.Max());
   fluid_mesh_deformation_ess_bdr = 1;

   fluid_mesh_deformation->SetEssentialBoundary(fluid_mesh_deformation_ess_bdr);

   bool pa_mesh = false;
   int prec_type_mesh = 1;

   if (Mesh_ctx.enable_stiffening)
   { // Enable distance-based stiffening only at the tip of the beam
      fluid_mesh_deformation->EnableLocalStiffening(fluid_solid_interface_tip_marker, Mesh_ctx.stiffening_type);
   }

   switch (Mesh_ctx.mesh_deformation_type)
   {
   case MeshDeformationType::Laplacian: // Laplacian
      static_cast<MeshLaplacian *>(fluid_mesh_deformation)->Setup(1.0, pa_mesh, prec_type_mesh);
      break;
   case MeshDeformationType::IncrementalLaplacian: // Incremental Laplacian
      static_cast<MeshLaplacian *>(fluid_mesh_deformation)->Setup(1.0, pa_mesh, prec_type_mesh);
      break;
   case MeshDeformationType::LinearElasticity: // Elasticity
   {
      static_cast<MeshLinearElasticity *>(fluid_mesh_deformation)->Setup(Mesh_ctx.E, Mesh_ctx.nu, pa_mesh, prec_type_mesh);
      break;
   }
   default:
      MFEM_ABORT("Invalid mesh deformation type.");
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Setup interface transfer
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "Setting up interface transfer for FSI problem... " << std::endl;

   // For the SOLID-->FLUID use the undeformed coordinates (as SOLID is in the undeformed configuration)
   // For the FLUID-->SOLID use the deformed coordinates (as FLUID is in the deformed configuration). We'll provide the offset as a coefficient in the Update()
   BidirectionalInterfaceTransfer finder_fluid_to_solid(u_gf->ParFESpace(), elasticity_fes, fluid_solid_interface_marker, TransferBackend::GSLIB, pmesh.GetComm(), false, true);
   InterfaceTransfer finder_solid_to_mesh(solid_vel_gf.ParFESpace(), mesh_nodes_fes, fluid_solid_interface_marker, TransferBackend::GSLIB, pmesh.GetComm(), true);

   // Tolerance for the boundary distance: points for which GSLIB finds distance > tol will be considered as not found.
   // Useful when the offset function used to update the finders is not exact.
   real_t boundary_distance_tolerance = 1e-4;
   finder_fluid_to_solid.SetBoundaryDistanceTolerance(boundary_distance_tolerance, boundary_distance_tolerance);
   finder_solid_to_mesh.SetBoundaryDistanceTolerance(boundary_distance_tolerance);

   // Define lamdbas to compute QoI for the interface transfer
   int sdim = fluid_pmesh.Dimension();

   // Compute σ·n = ( - p I + 2 ν ε(u) )·n = ( - p I + ν (∇u + ∇u^T) )·n
   auto FluidBoundaryNormalStress = [&](ElementTransformation &Tr, const IntegrationPoint &ip, Vector &qoi_loc)
   {
      real_t nu, p;
      Vector sigmaN, nor(sdim);
      DenseMatrix dudx, A(sdim, sdim);

      // Compute the normal vector
      CalcOrtho(Tr.Jacobian(), nor);
      const real_t scale = nor.Norml2();
      nor /= scale;

      // Get velocity anc pressure
      un_next_gf->GetVectorGradient(Tr, dudx);
      p = p_gf->GetValue(Tr, ip);
      nu = nu_gf->GetValue(Tr, ip);

      // Compute  σ·n = ( - p I + rho 2 ν ε(u) )·n = ( - p I + rho ν (∇u + ∇u^T) )·n
      for (int i = 0; i < sdim; i++)
      {
         for (int j = 0; j < sdim; j++)
         {
            A(i, j) = -p * (i == j) + NS_ctx.density * nu * (dudx(i, j) + dudx(j, i));
         }
      }

      for (int i = 0; i < sdim; i++)
      {
         real_t s = 0.0;
         for (int j = 0; j < sdim; j++)
         {
            s += A(i, j) * nor(j);
         }
         qoi_loc(i) = s;
      }
   };

   // --face-traction: viscous traction tau·n = rho ν (∇u + ∇u^T)·n_f, and the pressure as the vector (p, 0)
   auto FluidBoundaryViscousTraction = [&](ElementTransformation &Tr, const IntegrationPoint &ip, Vector &qoi_loc)
   {
      Vector nor(sdim);
      DenseMatrix dudx;
      CalcOrtho(Tr.Jacobian(), nor);
      nor /= nor.Norml2();
      un_next_gf->GetVectorGradient(Tr, dudx);
      const real_t nu = nu_gf->GetValue(Tr, ip);
      for (int i = 0; i < sdim; i++)
      {
         real_t s = 0.0;
         for (int j = 0; j < sdim; j++) { s += NS_ctx.density * nu * (dudx(i, j) + dudx(j, i)) * nor(j); }
         qoi_loc(i) = s;
      }
   };
   auto FluidBoundaryPressure = [&](ElementTransformation &Tr, const IntegrationPoint &ip, Vector &qoi_loc)
   {
      qoi_loc = 0.0;
      qoi_loc(0) = p_gf->GetValue(Tr, ip);
   };

   // Compute σ·n = [ \lambda (∇·u) + mu (∇u + ∇u^T) ]·n_s
   // where n_s is the outward normal vector of the solid
   // a -1 correction factor is applied to the coefficient when provided to the fluid solver
   auto SolidBoundaryNormalStress = [&](ElementTransformation &Tr, const IntegrationPoint &ip, Vector &qoi_loc)
   {
      Vector sigmaN, nor(sdim);
      DenseMatrix dudx, A(sdim, sdim);

      // Compute the normal vector
      CalcOrtho(Tr.Jacobian(), nor);
      const real_t scale = nor.Norml2();
      nor /= scale;

      // Get the gradient of the solid displacement
      solid_displacement_gf.GetVectorGradient(Tr, dudx);

      // Compute divergence: div_u = ∇·u
      real_t div_u = dudx.Trace();
      div_u *= Elasticity_ctx.lambda;

      // Compute stress tensor: σ = λ(∇·u)I + μ(∇u + ∇u^T)
      for (int i = 0; i < sdim; i++)
      {
         for (int j = 0; j < sdim; j++)
         {
            A(i, j) = div_u * (i == j) +
                      Elasticity_ctx.mu * (dudx(i, j) + dudx(j, i));
         }
      }

      // Compute σ·n
      for (int i = 0; i < sdim; i++)
      {
         real_t s = 0.0;
         for (int j = 0; j < sdim; j++)
         {
            s += A(i, j) * nor(j);
         }
         qoi_loc(i) = s;
      }
   };

   // GridFunctionCoefficients, vectors for error evaluation and relaxation steps
   ParGridFunction solid_vel_onFluid_prev_gf(u_gf->ParFESpace());
   solid_vel_onFluid_prev_gf = 0.0;
   ParGridFunction fluid_sigmaN_onSolid_prev_gf(elasticity_fes);
   fluid_sigmaN_onSolid_prev_gf = 0.0;
   ParGridFunction solid_sigmaN_onFluid_prev_gf(u_gf->ParFESpace());
   solid_sigmaN_onFluid_prev_gf = 0.0;

   VectorGridFunctionCoefficient solid_vel_onFluid_prev_coeff(&solid_vel_onFluid_prev_gf);
   VectorGridFunctionCoefficient solid_sigmaN_onFluid_prev_coeff(&solid_sigmaN_onFluid_prev_gf);
   VectorGridFunctionCoefficient fluid_sigmaN_onSolid_prev_coeff(&fluid_sigmaN_onSolid_prev_gf);

   Vector solid_vel_onFluid, fluid_sigmaN_onSolid, solid_sigmaN_onFluid, solid_vel_onFluid_prev, fluid_sigmaN_onSolid_prev, solid_sigmaN_onFluid_prev;
   solid_vel_onFluid_gf.GetTrueDofs(solid_vel_onFluid);
   solid_sigmaN_onFluid_gf.GetTrueDofs(solid_sigmaN_onFluid);
   fluid_sigmaN_onSolid_gf.GetTrueDofs(fluid_sigmaN_onSolid);
   solid_vel_onFluid_prev_gf.GetTrueDofs(solid_vel_onFluid_prev);
   fluid_sigmaN_onSolid_prev_gf.GetTrueDofs(fluid_sigmaN_onSolid_prev);
   solid_sigmaN_onFluid_prev_gf.GetTrueDofs(solid_sigmaN_onFluid_prev);

   // Extract the indices of elements at the interface and convert them to markers
   // Useful to restrict the computation of the L2 error to the elements on the interface
   // NOTE: i don't think we should update this after the transfer is updated, as the elements should be the same even if the mesh is deformed
   Array<int> solid_interfaces_element_idx;
   Array<int> fluid_interfaces_element_idx;
   finder_fluid_to_solid.GetElementIdxSrc(fluid_interfaces_element_idx);
   finder_fluid_to_solid.GetElementIdxDst(solid_interfaces_element_idx);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Set up coefficients and output
   ///////////////////////////////////////////////////////////////////////////////////////////////

   ParLinearForm lift_drag_form(u_gf->ParFESpace());
   auto stress_integ = new BoundaryNormalStressIntegrator(*u_gf, *p_gf, *nu_gf, NS_ctx.density);
   auto stress_integ_copy = new BoundaryNormalStressIntegrator(*u_gf, *p_gf, *nu_gf, NS_ctx.density);
   stress_integ->SetIntRule(&navier.gll_ir_face);
   lift_drag_form.AddBoundaryIntegrator(stress_integ, fluid_interface_attr);
   lift_drag_form.AddBoundaryIntegrator(stress_integ_copy, fluid_cylinder_attr);

   ParaViewDataCollection pvdc_fluid("Fluid", &fluid_pmesh);
   pvdc_fluid.SetPrefixPath(outfolder);
   pvdc_fluid.SetDataFormat(VTKFormat::BINARY);
   pvdc_fluid.SetHighOrderOutput(true);
   pvdc_fluid.SetCompression(true);
   pvdc_fluid.SetCompressionLevel(9);
   pvdc_fluid.SetLevelsOfDetail(NS_ctx.u_order);
   pvdc_fluid.SetCycle(current_step);
   pvdc_fluid.SetTime(0.0);
   pvdc_fluid.RegisterField("velocity", u_gf);
   pvdc_fluid.RegisterField("provisional_velocity", navier.GetProvisionalVelocity());
   pvdc_fluid.RegisterField("pressure", p_gf);
   pvdc_fluid.RegisterField("interface_velocity", &solid_vel_onFluid_gf);
   pvdc_fluid.RegisterField("mesh_velocity", mesh_velocity_gf); // CHECK --> Can we store the mesh velocity even tho it has a different fes?
   ParGridFunction *distance_gf = nullptr;
   if (Mesh_ctx.enable_stiffening)
   {
      if (Mesh_ctx.stiffening_type == 1)
      {
         auto stiffening_coeff = fluid_mesh_deformation->GetLocalStiffeningCoefficient();
         distance_gf = static_cast<DistanceBasedStiffeningCoefficient *>(stiffening_coeff)->GetDistanceGf();
         pvdc_fluid.RegisterField("distance", distance_gf);
      }
   }   
   
   pvdc_fluid.Save();

   ParaViewDataCollection pvdc_solid("Solid", &solid_pmesh);
   pvdc_solid.SetPrefixPath(outfolder);
   pvdc_solid.SetDataFormat(VTKFormat::BINARY);
   pvdc_solid.SetHighOrderOutput(true);
   pvdc_solid.SetCompression(true);
   pvdc_solid.SetCompressionLevel(9);
   pvdc_solid.SetLevelsOfDetail(Elasticity_ctx.order);
   pvdc_solid.SetCycle(current_step);
   pvdc_solid.SetTime(0.0);
   pvdc_solid.RegisterField("displacement", &solid_displacement_gf);
   pvdc_solid.RegisterField("velocity", &solid_vel_gf);
   pvdc_solid.RegisterField("interface_sigmaN", &fluid_sigmaN_onSolid_gf);
   pvdc_solid.Save();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Solve FSI problem
   ///////////////////////////////////////////////////////////////////////////////////////////////

   ////////////////////////////////////////////////////////////////////////////////////////////////
   /// 8.1 ----- Preload phase
   ////////////////////////////////////////////////////////////////////////////////////////////////

   // Nothing for this test case, but might be useful for other cases
   int paraview_step = 0; // ParaView step counter

   // Fluid traction on the fluid interface (as computed by the forward transfer): Robin datum with --robin-traction 0
   ParGridFunction fluid_sigmaN_onFluid_gf(u_gf->ParFESpace());
   fluid_sigmaN_onFluid_gf = 0.0;
   // FLUID --> SOLID traction transfer. The nodal sigma.n is always computed (Aitken relaxation and convergence of
   // the Robin datum); with --face-traction also tau.n and p, from which the solid load and the Robin datum are built.
   auto TransferFluidTraction = [&]()
   {
      finder_fluid_to_solid.InterpolateQoIForward(FluidBoundaryNormalStress, fluid_sigmaN_onSolid_gf, true);
      fluid_sigmaN_onFluid_gf = finder_fluid_to_solid.GetForwardSourceQoI();
      if (DD_ctx.face_traction)
      {
         finder_fluid_to_solid.InterpolateQoIForward(FluidBoundaryViscousTraction, fluid_tauN_onSolid_gf, true);
         fluid_tauN_onFluid_gf = finder_fluid_to_solid.GetForwardSourceQoI();
         finder_fluid_to_solid.InterpolateQoIForward(FluidBoundaryPressure, fluid_p_onSolid_gf, true);
         fluid_p_onFluid_gf = finder_fluid_to_solid.GetForwardSourceQoI();
      }
   };

   // Robin-type couplings during the (rigid) preload: the interface condition is the Robin one, alpha u + sigma n =
   // alpha u_s + g, with u_s = 0. A zero traction datum g would make the "rigid" beam porous, u.n = p / alpha (and the
   // datum would jump to the fluid traction at the start of the coupled phase). Instead g is the fluid traction of the
   // previous step, so that u = 0 up to the change of the traction in one step.
   const bool robin_preload_datum = DD_ctx.coupling != CouplingType::Dirichlet_Neumann;
   auto UpdatePreloadRobinDatum = [&]()
   {
      TransferFluidTraction();
      solid_sigmaN_onFluid_gf = fluid_sigmaN_onFluid_gf;
      solid_sigmaN_onFluid_gf.Neg();
      datum_tauN_onFluid_gf = fluid_tauN_onFluid_gf;
      datum_p_onFluid_gf = fluid_p_onFluid_gf;
   };
   // Interface volume flux int_Gamma v.n_f of a vector field v (u_f: zero for an impermeable rigid beam)
   auto InterfaceFlux = [&](ParGridFunction &v)
   {
      VectorGridFunctionCoefficient v_coeff(&v);
      ParLinearForm flux_form(p_gf->ParFESpace());
      flux_form.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(v_coeff), fluid_interface_attr);
      flux_form.Assemble();
      real_t flux = flux_form.Sum(); // shape functions sum to one; each rank holds its own boundary elements
      MPI_Allreduce(MPI_IN_PLACE, &flux, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, pmesh.GetComm());
      return flux;
   };

   ////////////////////////////////////////////////////////////////////////////////////////////////
   /// 8.1 ----- Preload phase
   ////////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\nStarting preload phase..." << std::endl;

   if (Mpi::Root())
   {
      out << "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
      out << std::left
          << std::setw(8) << "Step"
          << std::setw(16) << "Time"
          << std::setw(16) << "dt"
          << std::setw(16) << "Lift"
          << std::setw(16) << "Drag"
          << std::setw(16) << "CFL"
          << std::setw(16) << "VelNorm"
          << std::setw(16) << "p_probe"
          << std::setw(16) << "disp_x"
          << std::setw(16) << "disp_y"
          << std::setw(16) << "subiter"
          << std::endl;
      out << "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
   }

   real_t error_est = 0.0; // Error estimate for time adaptivity
   Array<real_t> lift_history, drag_history, delta_p_history;
   real_t t_preload_final = NS_ctx.preloadT; // Preload time for rigid CFD problem is the same as t_final --> we change it here, otherwise changing preloadT will alter the inflow profiles
   real_t transitionT = 0.99*t_preload_final; // Time after which timestep decreases to match the fsi simulation one
   real_t transition_duration = t_preload_final - transitionT;

   if (cfd_rigid)
   {
      t_preload_final = t_final; // For rigid CFD, we preload until t_final
      transitionT = t_preload_final;
   }
   else if (!enable_preload)
   {
      // t_preload_final = (NS_ctx.max_bdf_order)*dt;
      t_preload_final = 0.0;
      dt_preload = dt; // For FSI problems, we preload only for one time step
   }

   if (t_preload_final > 0.0)
   {
      for (int step = 0; !last_step; ++step) // TIME-INTEGRATION LOOP
      {
         if (current_time + dt_preload >= t_preload_final - dt_preload / 2)
         {
            last_step = true;
         }

         real_t previous_dt = dt_preload;
         real_t lift = 0.0, drag = 0.0;
         real_t cfl = 0.0;

         if (csm_only)
         { // Solve CSM only problem
            solid_displacement_gf.GetTrueDofs(solid_displacement);
            solid_vel_gf.GetTrueDofs(solid_vel);

            elasticity.Step(solid_displacement, solid_vel, current_time, dt_preload);

            solid_displacement_gf.SetFromTrueDofs(solid_displacement);
            solid_vel_gf.SetFromTrueDofs(solid_vel);
         }
         else // Solve Preload for FSI problem or rigid CFD problem
         {
            // Solver Navier-Stokes
            navier.Step(current_time, dt_preload, step, true);

            // Compute the CFL based on the provisional velocity
            if (time_adaptivity)
            {
               cfl = navier.ComputeCFL(*un_next_gf, dt_preload);
               error_est = cfl / (cfl_target + cfl_tol);
               if (error_est >= 1.0) // Retry fluid step
               {
                  // We will retry the step with a smaller time step
                  dt_preload *= 0.125;

                  if (Mpi::Root())
                     mfem::out << "\n Step " << step << " reached maximum CFL, retrying with smaller step size."
                               << std::setw(8) << " | old dt = " << std::scientific << std::setprecision(3) << previous_dt
                               << " | new dt = " << std::scientific << std::setprecision(3) << dt_preload << "\n"
                               << std::endl;

                  // Resets initial conditions and re-try the step and convergence fsi loop
                  // Otherwise when we re-enter the convergence loop, we  will assign *_prev to
                  // the current values of the grid functions.
                  step--; // Decrement step to retry
                  continue;
               }
            }

            // Accept the time step
            current_time += dt_preload;
            current_step++;

            // Queue new time step in the history array
            navier.UpdateTimestepHistory(dt_preload);

            if (robin_preload_datum) { UpdatePreloadRobinDatum(); }

            if (time_adaptivity && current_time < transitionT)
            { // Predict new step size
               real_t fac_safety = 2.0;
               real_t eta = pow(1.0 / (fac_safety * error_est), 1.0 / (1.0 + 3.0));
               real_t fac_min = 0.1;
               real_t fac_max = 1.4;
               dt_preload = dt_preload * std::min(fac_max, std::max(fac_min, eta));
            }

            if (current_time >= transitionT)
            {
               real_t transition_progress = (current_time - transitionT) / transition_duration;
               transition_progress = std::min(1.0, transition_progress);

               // Simple linear interpolation from current dt_preload to target dt
               real_t target_dt = dt_preload * (1.0 - transition_progress) + dt * transition_progress;
               
               dt_preload = target_dt;
            }
         }
         // --------------------------------------------------------------------------------- //

         // Output
         paraview_step++;
         if ((step + 1) % save_freq_preload == 0 || last_step)
         {
            if (Mpi::Root() && verbose)
               mfem::out << "\nWriting output to ParaView...";
            pvdc_fluid.SetCycle(paraview_step);
            pvdc_fluid.SetTime(current_time);
            pvdc_fluid.Save();
            pvdc_solid.SetCycle(paraview_step);
            pvdc_solid.SetTime(current_time);
            pvdc_solid.Save();
            if (Mpi::Root() && verbose)
               mfem::out << " done." << std::endl;
         }

         // Output some information
         // Output some information
         bool compute_output = true;
         real_t disp_x = 0.0, disp_y = 0.0;
         real_t p_probe1 = 0.0, p_probe2 = 0.0, delta_p = 0.0;
         real_t vel_norm1 = 0.0, vel_norm2 = 0.0;
         if (compute_output) // TODO: change with fixes from test_csm_only
         {
            Vector point1(2), point2(2);
            point1(0) = Mesh_ctx.x0 + 0.15; // Cylinder left vertex (fixed)
            point1(1) = Mesh_ctx.tip_y;
            point2(0) = Mesh_ctx.tip_x; // Beam right vertex (free)
            point2(1) = Mesh_ctx.tip_y;
            DenseMatrix points(dim, 2);
            points.SetCol(0, point1);
            points.SetCol(1, point2);

            Array<int> elem_ids;
            Array<IntegrationPoint> ips;

            fluid_pmesh.FindPoints(points, elem_ids, ips, false);

            // Initialize values to zero
            vel_norm1 = 0.0;
            vel_norm2 = 0.0;
            p_probe1 = 0.0;
            p_probe2 = 0.0;

            Vector val;

            // Only evaluate grid functions if the points were found on this rank
            if (elem_ids[0] >= 0)
            {
               u_gf->GetVectorValue(elem_ids[0], ips[0], val);
               vel_norm1 = val.Norml2();
               p_probe1 = p_gf->GetValue(elem_ids[0], ips[0]);
            }

            if (elem_ids[1] >= 0)
            {
               u_gf->GetVectorValue(elem_ids[1], ips[1], val);
               vel_norm2 = val.Norml2();
               p_probe2 = p_gf->GetValue(elem_ids[1], ips[1]);
            }

            // Reduce values across all ranks to get the result from whichever rank found the point
            real_t global_vals[4] = {vel_norm1, vel_norm2, p_probe1, p_probe2};
            real_t reduced_vals[4];
            MPI_Allreduce(global_vals, reduced_vals, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            vel_norm1 = reduced_vals[0];
            vel_norm2 = reduced_vals[1];
            p_probe1 = reduced_vals[2];
            p_probe2 = reduced_vals[3];

            delta_p = p_probe1 - p_probe2;

            Vector tip(2);
            tip(0) = Mesh_ctx.tip_x;
            tip(1) = Mesh_ctx.tip_y;
            DenseMatrix points_solid(dim, 1);
            points_solid.SetCol(0, tip);
            solid_pmesh.FindPoints(points_solid, elem_ids, ips);

            // Initialize displacement values
            disp_x = 0.0;
            disp_y = 0.0;

            // Only evaluate grid functions if the point was found on this rank
            if (elem_ids[0] >= 0)
            {
               solid_displacement_gf.GetVectorValue(elem_ids[0], ips[0], val);
               disp_x = val(0);
               disp_y = val(1);
            }

            // Reduce displacement values across all ranks
            real_t global_disp_vals[2] = {disp_x, disp_y};
            real_t reduced_disp_vals[2];
            MPI_Allreduce(global_disp_vals, reduced_disp_vals, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            disp_x = reduced_disp_vals[0];
            disp_y = reduced_disp_vals[1];

            ComputeDragLift(lift, drag, lift_drag_form);

            // Add to history
            lift_history.Append(lift);
            drag_history.Append(drag);
            delta_p_history.Append(delta_p);
         }

         if (Mpi::Root())
         {
            out << std::left
                << std::setw(8) << step
                << std::setw(16) << std::scientific << std::setprecision(8) << current_time
                << std::setw(16) << std::scientific << std::setprecision(8) << previous_dt
                << std::setw(16) << std::scientific << std::setprecision(8) << lift
                << std::setw(16) << std::scientific << std::setprecision(8) << drag
                << std::setw(16) << std::scientific << std::setprecision(8) << cfl
                << std::setw(16) << std::scientific << std::setprecision(8) << vel_norm1
                << std::setw(16) << std::scientific << std::setprecision(8) << delta_p
                << std::setw(16) << std::scientific << std::setprecision(8) << disp_x
                << std::setw(16) << std::scientific << std::setprecision(8) << disp_y
                << std::setw(16) << std::scientific << std::setprecision(8) << 1
                << std::endl;
         }

      } // END OF TIME-INTEGRATION LOOP for PRELOAD PHASE

      if (!csm_only)
      {
         // Leak through the (rigid) beam during the preload, and the one a zero Robin datum would give: u.n = -(sigma_f n_f).n_f / alpha
         const real_t flux = InterfaceFlux(*un_next_gf);
         ParGridFunction slip_zero_datum(u_gf->ParFESpace());
         slip_zero_datum = fluid_sigmaN_onFluid_gf;
         slip_zero_datum *= -1.0 / (Elasticity_ctx.density * std::sqrt((Elasticity_ctx.lambda + 2 * Elasticity_ctx.mu) / Elasticity_ctx.density));
         const real_t flux_zero = InterfaceFlux(slip_zero_datum);
         if (Mpi::Root())
         {
            mfem::out << std::scientific << std::setprecision(4) << "End of preload: interface volume flux int u.n = " << flux
                      << " m^2/s (with a zero Robin datum it would be ~ " << flux_zero << "; inflow flux "
                      << NS_ctx.Umax * Mesh_ctx.D * 2.0 / 3.0 * 1.5 << ")" << std::endl;
         }
      }

      if (cfd_rigid || csm_only)
      {
         if (Mpi::Root())
            mfem::out << "\nCFD/CSM only test done. Exiting..." << std::endl;

         if (cfd_rigid)
         {
            saveArray(lift_history, outfolder, "lift.txt");
            saveArray(drag_history, outfolder, "drag.txt");
            saveArray(delta_p_history, outfolder, "delta_p.txt");
         }

         return 0; // Exit if we are only running the CFD or CSM test
      }
   }

   ////////////////////////////////////////////////////////////////////////////////////////////////
   /// 8.2 ----- Coupled FSI
   ////////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\nStarting coupled FSI problem..." << std::endl;

   // Setup Aitken acceleration
   real_t omega_max = 1.0;                       // Maximum relaxation factor
   real_t omega_min = 1e-6;                      // Minimum relaxation factor
   real_t omega_init = DD_ctx.omega; // Relaxation factor for the solid problem
   int init_steps = 3;                           // Number of initial steps with constant relaxation factor

   AitkenRelaxation aitken_fluid_velocity(fluid_pmesh.GetComm(), omega_init, omega_min, omega_max, init_steps);
   AitkenRelaxation aitken_solid_sigma(solid_pmesh.GetComm(), omega_init, omega_min, omega_max, init_steps);
   aitken_fluid_velocity.SetSafetyFactor(DD_ctx.safety_factor);
   aitken_solid_sigma.SetSafetyFactor(DD_ctx.safety_factor);

   aitken_fluid_velocity.SetSize(u_fluid);
   aitken_solid_sigma.SetSize(solid_sigmaN_onFluid);

   // Array to store number of sub-iterations in inner convergence loop
   Array<int> subiter;
   Array<real_t> omega_history_fluid;
   Array<real_t> omega_history_solid;

   if (Mpi::Root())
   {
      out << "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
      out << std::left
          << std::setw(8) << "Step"
          << std::setw(16) << "Time"
          << std::setw(16) << "dt"
          << std::setw(16) << "Lift"
          << std::setw(16) << "Drag"
          << std::setw(16) << "CFL"
          << std::setw(16) << "VelNorm"
          << std::setw(16) << "p_probe"
          << std::setw(16) << "disp_x"
          << std::setw(16) << "disp_y"
          << std::setw(16) << "subiter"
          << std::endl;
      out << "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
   }

   last_step = false; // Reset last_step for the coupled FSI problem

   // Consistent t_0 interface data (as in navier-fsi-two-domains):
   // - the solid acceleration a_0 is computed from the t_0 fluid traction;
   // - Robin-type couplings start from the t_0 interface traction balance, i.e. the Robin traction datum of the
   //   first sub-iteration is the fluid traction itself: the solid released from the (rigid) preload is unstressed,
   //   so sigma_s(u_0) n = 0 would impose a spurious interface velocity alpha u = -sigma_f n.
   bool use_initial_robin_data = false;
   TransferFluidTraction();
   fluid_sigmaN_onSolid_gf.GetTrueDofs(fluid_sigmaN_onSolid);
   if (DD_ctx.face_traction)
   {
      // Check: net interface force of the nodal and of the face-normal traction, on the solid and on the fluid side
      // (equal up to the corner contributions; the fluid-side datum coefficient has the -1 of the Robin datum).
      auto NetForce = [&](ParMesh &m, VectorCoefficient &c, const Array<int> &elems)
      {
         Vector f(sdim), v(sdim);
         f = 0.0;
         for (int be : elems)
         {
            ElementTransformation *T = m.GetBdrElementTransformation(be);
            const IntegrationRule &ir = IntRules.Get(m.GetBdrElementGeometry(be), 2 * NS_ctx.u_order + 2);
            for (int q = 0; q < ir.GetNPoints(); q++)
            {
               T->SetIntPoint(&ir.IntPoint(q));
               c.Eval(v, *T, ir.IntPoint(q));
               f.Add(ir.IntPoint(q).weight * T->Weight(), v);
            }
         }
         MPI_Allreduce(MPI_IN_PLACE, f.GetData(), sdim, MPITypeMap<real_t>::mpi_type, MPI_SUM, m.GetComm());
         return f;
      };
      Array<int> s_elems, f_elems;
      for (int be = 0; be < solid_pmesh.GetNBE(); be++)
      {
         if (fluid_solid_interface_marker[solid_pmesh.GetBdrAttribute(be) - 1]) { s_elems.Append(be); }
      }
      for (int be = 0; be < fluid_pmesh.GetNBE(); be++)
      {
         if (fluid_solid_interface_marker[fluid_pmesh.GetBdrAttribute(be) - 1]) { f_elems.Append(be); }
      }
      datum_tauN_onFluid_gf = fluid_tauN_onFluid_gf;
      datum_p_onFluid_gf = fluid_p_onFluid_gf;
      solid_sigmaN_onFluid_gf = fluid_sigmaN_onFluid_gf;
      solid_sigmaN_onFluid_gf.Neg();
      Vector fs_nodal = NetForce(solid_pmesh, fluid_sigmaN_onSolid_nodal_coeff, s_elems);
      Vector fs_face = NetForce(solid_pmesh, fluid_sigmaN_onSolid_face_coeff, s_elems);
      Vector ff_nodal = NetForce(fluid_pmesh, solid_sigmaN_onFluid_nodal_coeff, f_elems);
      Vector ff_face = NetForce(fluid_pmesh, solid_sigmaN_onFluid_face_coeff, f_elems);
      if (Mpi::Root())
      {
         mfem::out << std::scientific << std::setprecision(4)
                   << "Face-traction check, net sigma_f.n_f on the solid interface: nodal (" << fs_nodal(0) << ", "
                   << fs_nodal(1) << "), face (" << fs_face(0) << ", " << fs_face(1) << ")\n"
                   << "                     net Robin datum on the fluid interface:  nodal (" << ff_nodal(0) << ", "
                   << ff_nodal(1) << "), face (" << ff_face(0) << ", " << ff_face(1) << ")" << std::endl;
      }
      solid_sigmaN_onFluid_gf = 0.0;
   }
   if (elasticity.GetLatestState() == nullptr)
   {
      Vector u0, v0;
      solid_displacement_gf.GetTrueDofs(u0);
      solid_vel_gf.GetTrueDofs(v0);
      elasticity.InitializeState(u0, v0, current_time, dt);
   }
   if (DD_ctx.coupling != CouplingType::Dirichlet_Neumann)
   {
      solid_sigmaN_onFluid_gf = 0.0;
      finder_fluid_to_solid.InterpolateBackward(fluid_sigmaN_onSolid_gf, solid_sigmaN_onFluid_gf);
      solid_sigmaN_onFluid_gf.Neg(); // Robin datum sigma_s.n_s = -sigma_f.n_f at the interface
      solid_sigmaN_onFluid_gf.GetTrueDofs(solid_sigmaN_onFluid);
      // With the fluid-traction datum the first sub-iteration evaluates it directly (t_0 fluid state)
      use_initial_robin_data = (DD_ctx.robin_traction == 1);
   }
   // current_step=0;  // Reset current step for the coupled FSI problem: this implies that we start timestepping with BDF1 with an already populated un
   real_t fluid_mesh_displacement_norm = 0.0;

   for (int step = current_step; !last_step; ++step) // TIME-INTEGRATION LOOP
   {
      if (current_time + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      real_t previous_dt = dt;

      real_t lift = 0.0, drag = 0.0;

      // ----------------------------------------------------------------------------------- //
      //// ----- BEGIN CONVERGENCE LOOP -----
      // ----------------------------------------------------------------------------------- //

      real_t cfl = 0.0, error_est = 0.0;

      bool converged = false;
      bool converged_absolute = false;
      bool converged_relative = false;

      int iter = 0;
      const int max_iter = 500;
      const int min_iter = 3; // TODO lower it. 10 is just for debugging
      real_t rel_tol = 1e-4;
      real_t abs_tol = 1e-7;

      //real_t subiter_time = current_time;

      // Store initial conditions
      solid_displacement_gf.GetTrueDofs(solid_displacement_tn);
      solid_vel_gf.GetTrueDofs(solid_vel_tn);
      u_gf->GetTrueDofs(u_fluid_tn);
      p_gf->GetTrueDofs(p_fluid_tn);
      fluid_sigmaN_onSolid_gf.GetTrueDofs(fluid_sigmaN_onSolid);
      u_fluid = u_fluid_tn;                                  // Initialize the fluid velocity with the initial condition
      p_fluid = p_fluid_tn;                                  // Initialize the fluid pressure with the initial condition
      solid_displacement = solid_displacement_tn;            // Initialize the solid displacement with the initial condition
      solid_vel = solid_vel_tn;                              // Initialize the solid velocity with the initial condition
      // a^n for resetting the solid at every sub-iteration. Deep copy: GetLatestState() aliases the ODE solver's
      // live state, which Step() overwrites in place (ResetState would otherwise be a self-assignment).
      Vector acceleration_tn_copy;
      Vector *acceleration_tn = nullptr;
      if (Vector *acc = elasticity.GetLatestState())
      {
         acceleration_tn_copy = *acc;
         acceleration_tn = &acceleration_tn_copy;
      }
      mesh_displacement_gf->GetTrueDofs(mesh_displacement_tn);

      while (!converged && iter <= max_iter)
      {
         // Store quantities for error computation and relaxation
         solid_vel_onFluid_prev = solid_vel_onFluid;       // For relaxation step/error computation (Dirichlet or Robin)
         solid_sigmaN_onFluid_prev = solid_sigmaN_onFluid; // For relaxation step (Robin)
         fluid_sigmaN_onSolid_prev = fluid_sigmaN_onSolid; // For error computation
         const Vector datum_tauN_prev(datum_tauN_onFluid_gf), datum_p_prev(datum_p_onFluid_gf); // --face-traction

         // Update gridfunction for error computation
         solid_vel_onFluid_prev_gf.SetFromTrueDofs(solid_vel_onFluid_prev);
         solid_sigmaN_onFluid_prev_gf.SetFromTrueDofs(solid_sigmaN_onFluid_prev);
         fluid_sigmaN_onSolid_prev_gf.SetFromTrueDofs(fluid_sigmaN_onSolid_prev);

         // ---------------------------------------------------------------------------------- //
         //// ----- 1.. Interpolate velocity SOLID --> FLUID -----
         // --------------------------------------------------------------------------------- //

         if (Mpi::Root() && verbose)
            mfem::out << "Interpolating velocity SOLID --> FLUID...";
         finder_fluid_to_solid.InterpolateBackward(solid_vel_gf, solid_vel_onFluid_gf);
   
         if (DD_ctx.coupling != CouplingType::Dirichlet_Neumann && !use_initial_robin_data) // Avoid computing stress for Dirichlet-Neumann coupling
         {
            if (DD_ctx.robin_traction == 0)
            {
               // Consistent Robin-Neumann datum: the traction applied to the solid, i.e. the fluid traction of the
               // previous sub-iterate (at the first sub-iteration, of t_n), as evaluated on the fluid side by the
               // forward transfer.
               solid_sigmaN_onFluid_gf = fluid_sigmaN_onFluid_gf;
               solid_sigmaN_onFluid_gf.Neg(); // NavierSolver's Robin datum is sigma_s.n_s = -sigma_f.n_f (it applies a -1)
               // --face-traction: the datum actually applied is -(tau_n - p n_f) (sign in the coefficient)
               datum_tauN_onFluid_gf = fluid_tauN_onFluid_gf;
               datum_p_onFluid_gf = fluid_p_onFluid_gf;
            }
            else
            {
               finder_fluid_to_solid.InterpolateQoIBackward(SolidBoundaryNormalStress, solid_sigmaN_onFluid_gf, true);
            }
         }
         use_initial_robin_data = false;

         // Relaxation interface variable/s
         // Aitken: the residual is r_k = x~_k - x_{k-1}, with x_{k-1} the *relaxed* iterate used in the previous
         // sub-iteration (the *_prev vectors), see Kuttler & Wall (2008).
         real_t w_sv = 1.0, w_ss = 1.0; // relaxation factors actually used in this sub-iteration
         if (iter > 0) 
         {
            // Relax velocity
            solid_vel_onFluid_gf.GetTrueDofs(solid_vel_onFluid);
            w_sv = DD_ctx.aitken ? aitken_fluid_velocity.ComputeRelaxation(solid_vel_onFluid, solid_vel_onFluid_prev) : DD_ctx.omega;
            solid_vel_onFluid *= w_sv;
            solid_vel_onFluid.Add(1.0 - w_sv, solid_vel_onFluid_prev);
            solid_vel_onFluid_gf.SetFromTrueDofs(solid_vel_onFluid);

            if (DD_ctx.coupling != CouplingType::Dirichlet_Neumann) // Avoid computing stress for Dirichlet-Neumann coupling
            {
               // Relax traction
               solid_sigmaN_onFluid_gf.GetTrueDofs(solid_sigmaN_onFluid);
               w_ss = DD_ctx.aitken ? aitken_solid_sigma.ComputeRelaxation(solid_sigmaN_onFluid, solid_sigmaN_onFluid_prev) : DD_ctx.omega;
               solid_sigmaN_onFluid *= w_ss;
               solid_sigmaN_onFluid.Add(1.0 - w_ss, solid_sigmaN_onFluid_prev);
               solid_sigmaN_onFluid_gf.SetFromTrueDofs(solid_sigmaN_onFluid);
               if (face_robin_datum)
               {
                  // Same relaxation (factor from the nodal sigma.n datum) for the face-normal datum fields
                  datum_tauN_onFluid_gf *= w_ss;
                  datum_tauN_onFluid_gf.Add(1.0 - w_ss, datum_tauN_prev);
                  datum_p_onFluid_gf *= w_ss;
                  datum_p_onFluid_gf.Add(1.0 - w_ss, datum_p_prev);
               }
            }
         }

         if (Mpi::Root() && verbose)
            mfem::out << " done." << std::endl;

         // ---------------------------------------------------------------------------------- //

         // ---------------------------------------------------------------------------------- //
         //// ----- 2.. Solve Navier-Stokes on FLUID -----
         // --------------------------------------------------------------------------------- //

         if (Mpi::Root() && verbose)
            mfem::out << "Solving Navier-Stokes on FLUID...";

         // Resets initial conditions (in time)
         un_next_gf->SetFromTrueDofs(u_fluid_tn);
         p_gf->SetFromTrueDofs(p_fluid_tn);

         // Solve
         navier.Step(current_time, dt, step, true);

         if (Mpi::Root() && verbose)
            mfem::out << " done." << std::endl;

         // --------------------------------------------------------------------------------- //

         // --------------------------------------------------------------------------------- //
         //// ----- 3.. Interpolate traction FLUID --> SOLID -----
         // --------------------------------------------------------------------------------- //

         if (Mpi::Root() && verbose)
            mfem::out << "Interpolating traction FLUID --> SOLID...";
         TransferFluidTraction(); // Uses the InterpolateQoIv2 (to compute normal); also the Robin datum of the next sub-iteration
         fluid_sigmaN_onSolid_gf.GetTrueDofs(fluid_sigmaN_onSolid);
         if (Mpi::Root() && verbose)
            mfem::out << " done." << std::endl;

         // --------------------------------------------------------------------------------- //

         // --------------------------------------------------------------------------------- //
         //// ----- 4.. Solve Elasticity on SOLID -----
         // --------------------------------------------------------------------------------- //

         if (Mpi::Root() && verbose)
            mfem::out << "Solving Elasticity on SOLID...";

         // Resets initial conditions (in time)
         solid_displacement = solid_displacement_tn;
         solid_vel = solid_vel_tn;
         elasticity.ResetState(acceleration_tn);

         // Solve
         elasticity.Step(solid_displacement, solid_vel, current_time, dt, true);
         solid_displacement_gf.SetFromTrueDofs(solid_displacement);
         solid_vel_gf.SetFromTrueDofs(solid_vel);

         if (Mpi::Root() && verbose)
            mfem::out << " done." << std::endl;

         // --------------------------------------------------------------------------------- //

         // --------------------------------------------------------------------------------- //
         //// ----- 5.. Interpolate velocity SOLID --> FLUID -----
         // --------------------------------------------------------------------------------- //

         // Solve the mesh deformation (without actually deforming the mesh) to compute mesh velocity

         if (Mpi::Root() && verbose)
            mfem::out << "Solving Mesh Deformation..." << std::endl;

         Vector mesh_velocity;
         if (fluid_mesh_deformation->IsNonIncremental())
         {
            *mesh_displacement_gf = 0.0;
            finder_solid_to_mesh.Interpolate(solid_displacement_gf, *mesh_displacement_gf);
            fluid_mesh_deformation->Solve(*mesh_displacement_gf, false); // Solve the mesh laplacian for the total displacement
            mesh_displacement_gf->GetTrueDofs(mesh_velocity);            // v = d
            mesh_velocity -= mesh_displacement_tn;                       // v = d - d_prev
            fluid_mesh_displacement_norm = mesh_velocity.Norml2();       // |dx|_2 norm
         }
         else
         { // Compute relative displacement
            Vector rel_displacement;
            solid_displacement_gf.GetTrueDofs(rel_displacement);
            rel_displacement -= solid_displacement_tn;                         // Compute relative displacement
            solid_relative_displacement_gf->SetFromTrueDofs(rel_displacement); // Set the relative displacement field
            *mesh_displacement_gf = 0.0;
            finder_solid_to_mesh.Interpolate(*solid_relative_displacement_gf, *mesh_displacement_gf);
            fluid_mesh_deformation->Solve(*mesh_displacement_gf, false); // Solve the mesh laplacian for the relative displacement
            mesh_displacement_gf->GetTrueDofs(mesh_velocity);            // v = d - d_prev
            fluid_mesh_displacement_norm = mesh_velocity.Norml2();       // |dx|_2 norm
         }

         mesh_velocity *= (1.0 / dt);
         mesh_velocity_gf->SetFromTrueDofs(mesh_velocity);

         if (Mpi::Root() && verbose)
            mfem::out << " done." << std::endl;

         // --------------------------------------------------------------------------------- //

         // --------------------------------------------------------------------------------- //
         //// ----- 6.. Check convergence -----
         // --------------------------------------------------------------------------------- //

         // Compute global L2 error norms
         // The relaxed increment is x_k - x_{k-1} = w r_k: divide by w to test the residual r_k itself (otherwise a
         // small Aitken factor gives a false convergence). Norms: L2 on the same interface elements.
         Vector zero_vec_conv(sdim);
         zero_vec_conv = 0.0;
         VectorConstantCoefficient zero_vcoeff_conv(zero_vec_conv);
         real_t global_norm_fluid = solid_vel_onFluid_gf.ComputeL2Error(zero_vcoeff_conv, irs_vel, &fluid_interfaces_element_idx);
         //real_t global_norm_solid = GlobalLpNorm(2.0, fluid_sigmaN_onSolid_gf.Norml2(), solid_pmesh.GetComm());
         real_t global_abs_err_fluid = solid_vel_onFluid_gf.ComputeL2Error(solid_vel_onFluid_prev_coeff, irs_vel, &fluid_interfaces_element_idx);
         //real_t global_abs_err_solid = fluid_sigmaN_onSolid_gf.ComputeL2Error(fluid_sigmaN_onSolid_prev_coeff, irs_disp, &solid_interfaces_element_idx);
         global_abs_err_fluid /= w_sv;
         real_t global_rel_err_fluid = (global_norm_fluid > 1e-14) ? global_abs_err_fluid / global_norm_fluid : global_abs_err_fluid;
         if (DD_ctx.coupling != CouplingType::Dirichlet_Neumann)
         {
            // Robin data: the traction residual must also be small
            const real_t norm_ss = solid_sigmaN_onFluid_gf.ComputeL2Error(zero_vcoeff_conv, irs_vel, &fluid_interfaces_element_idx);
            const real_t abs_ss = solid_sigmaN_onFluid_gf.ComputeL2Error(solid_sigmaN_onFluid_prev_coeff, irs_vel, &fluid_interfaces_element_idx) / w_ss;
            global_abs_err_fluid = std::max(global_abs_err_fluid, abs_ss);
            global_rel_err_fluid = std::max(global_rel_err_fluid, (norm_ss > 1e-14) ? abs_ss / norm_ss : abs_ss);
         }
         //real_t global_rel_err_solid = global_abs_err_solid / global_norm_solid;
         
         // Check convergence
         //converged_relative = (global_rel_err_solid < rel_tol) && (global_rel_err_fluid < rel_tol) && (iter >= min_iter - 1);
         //converged_absolute = (global_abs_err_solid < abs_tol) && (global_abs_err_fluid < abs_tol) && (iter >= min_iter - 1);
         converged_relative = (global_rel_err_fluid < rel_tol) && (iter >= min_iter - 1);
         converged_absolute = (global_abs_err_fluid < abs_tol) && (iter >= min_iter - 1);
         converged = converged_absolute || converged_relative;
         //converged = converged_relative;

         // Update the iteration counter
         iter++;

         omega_history_fluid.Append(w_sv);
         omega_history_solid.Append(w_ss);

         /*paraview_step++;
         subiter_time += dt;
         pvdc_fluid.SetCycle(paraview_step);
         pvdc_fluid.SetTime(subiter_time);
         pvdc_fluid.Save();
         pvdc_solid.SetCycle(paraview_step);
         pvdc_solid.SetTime(subiter_time);
         pvdc_solid.Save(); */ 

         if (iter == max_iter)
         {
            if (Mpi::Root())
            {
               mfem::out << "\nWARNING: Maximum number of iterations reached in the convergence loop. "
                         << "Interface error (abs) = " << std::scientific << std::setprecision(3) << global_abs_err_fluid
                         << " | Interface error (rel) = " << std::scientific << std::setprecision(3) << global_rel_err_fluid
                         //<< " | Solid error = " << std::scientific << std::setprecision(3) << global_abs_err_solid
                         << std::endl;
            }
         }

         // --------------------------------------------------------------------------------- //
      }

      // --------------------------------------------------------------------------------- //
      //// ----- END OF CONVERGENCE LOOP -----
      // --------------------------------------------------------------------------------- //

      // Reset the residual buffer for the aitken relaxation
      aitken_fluid_velocity.Reset();
      aitken_solid_sigma.Reset();

      // Diagnostic: kinematic mismatch |u_f - u_s|_Gamma / |u_s|_Gamma at the end of the sub-iterations, with u_f the
      // fluid velocity and u_s the final solid velocity on the interface (zero for an exactly coupled solution).
      {
         ParGridFunction us_onFluid(u_gf->ParFESpace());
         us_onFluid = 0.0;
         finder_fluid_to_solid.InterpolateBackward(solid_vel_gf, us_onFluid);
         VectorGridFunctionCoefficient us_coeff(&us_onFluid);
         Vector zero_vec_km(sdim);
         zero_vec_km = 0.0;
         VectorConstantCoefficient zero_km(zero_vec_km);
         // Restrict the fluid velocity to the interface dofs, as us_onFluid (zero elsewhere): the norms below are over
         // the elements adjacent to the interface, so both fields must vanish away from it.
         ParGridFunction uf_onFluid(u_gf->ParFESpace());
         uf_onFluid = 0.0;
         {
            Array<int> iface_vdofs;
            u_gf->ParFESpace()->GetEssentialVDofs(fluid_interface_attr, iface_vdofs);
            for (int i = 0; i < iface_vdofs.Size(); i++)
            {
               if (iface_vdofs[i]) { uf_onFluid(i) = (*un_next_gf)(i); }
            }
         }
         const real_t mismatch = uf_onFluid.ComputeL2Error(us_coeff, irs_vel, &fluid_interfaces_element_idx);
         const real_t us_norm = us_onFluid.ComputeL2Error(zero_km, irs_vel, &fluid_interfaces_element_idx);
         // Pointwise slip at the interface nodes: maximum and its location, and the maximum |u_s| for scale
         struct { double v; int r; } loc_max = {0.0, Mpi::WorldRank()}, glob_max;
         real_t us_max = 0.0, xy_max[2] = {0.0, 0.0};
         {
            ParFiniteElementSpace *vfes_km = u_gf->ParFESpace();
            const FiniteElement *fe0 = vfes_km->GetFE(0);
            MFEM_VERIFY(fe0 && dynamic_cast<const NodalFiniteElement *>(fe0), "slip diagnostic needs a nodal space");
            ParGridFunction xyz(vfes_km);
            VectorFunctionCoefficient id_coeff(sdim, [](const Vector &x, Vector &y) { y = x; });
            xyz.ProjectCoefficient(id_coeff);
            Array<int> iface_vdofs;
            vfes_km->GetEssentialVDofs(fluid_interface_attr, iface_vdofs);
            for (int j = 0; j < vfes_km->GetNDofs(); j++)
            {
               if (!iface_vdofs[vfes_km->DofToVDof(j, 0)]) { continue; }
               real_t s2 = 0.0, u2 = 0.0;
               for (int d = 0; d < sdim; d++)
               {
                  const int vd = vfes_km->DofToVDof(j, d);
                  s2 += std::pow((*un_next_gf)(vd) - us_onFluid(vd), 2);
                  u2 += std::pow(us_onFluid(vd), 2);
               }
               us_max = std::max(us_max, std::sqrt(u2));
               if (std::sqrt(s2) > loc_max.v)
               {
                  loc_max.v = std::sqrt(s2);
                  for (int d = 0; d < 2; d++) { xy_max[d] = xyz(vfes_km->DofToVDof(j, d)); }
               }
            }
         }
         MPI_Allreduce(&loc_max, &glob_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
         MPI_Bcast(xy_max, 2, MPITypeMap<real_t>::mpi_type, glob_max.r, MPI_COMM_WORLD);
         MPI_Allreduce(MPI_IN_PLACE, &us_max, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, MPI_COMM_WORLD);
         if (Mpi::Root())
         {
            mfem::out << "Kinematic mismatch: |u_f - u_s|_Gamma = " << std::scientific << std::setprecision(3)
                      << mismatch << " | |u_s|_Gamma = " << us_norm
                      << " | max nodal slip = " << glob_max.v << " at (" << xy_max[0] << ", " << xy_max[1]
                      << ") | max |u_s| = " << us_max << std::endl;
         }
      }

      // Check if we exit the convergence loop due to retrying the step in Navier-Stokes
      // Compute the CFL based on the provisional velocity
      if (time_adaptivity)
      {
         cfl = navier.ComputeCFL(*un_next_gf, dt);
         error_est = cfl / (cfl_target + cfl_tol);
         if (error_est >= 1.0) // Retry fluid step
         {
            // We will retry the step with a smaller time step
            dt *= 0.125;

            if (Mpi::Root())
               mfem::out << "\n Step " << step << " reached maximum CFL, retrying with smaller step size."
                         << std::setw(8) << " | old dt = " << std::scientific << std::setprecision(3) << previous_dt
                         << " | new dt = " << std::scientific << std::setprecision(3) << dt << "\n"
                         << std::endl;

            // Resets initial conditions and re-try the step and convergence fsi loop
            // Otherwise when we re-enter the convergence loop, we  will assign *_prev to
            // the current values of the grid functions.
            step--; // Decrement step to retry
            // Reset variables for elasticity
            solid_displacement = solid_displacement_tn;
            solid_vel = solid_vel_tn;
            elasticity.ResetState(acceleration_tn);
            solid_displacement_gf.SetFromTrueDofs(solid_displacement);
            solid_vel_gf.SetFromTrueDofs(solid_vel);
            // Reset variables for Navier-Stokes
            u_fluid = u_fluid_tn;
            p_fluid = p_fluid_tn;
            un_next_gf->SetFromTrueDofs(u_fluid);
            p_gf->SetFromTrueDofs(p_fluid);
            continue;
         }

         // Predict new step size
         real_t fac_safety = 2.0;
         real_t eta = pow(1.0 / (fac_safety * error_est), 1.0 / (1.0 + NS_ctx.max_bdf_order));
         real_t fac_min = 0.1;
         real_t fac_max = 2.0;
         dt = dt * std::min(fac_max, std::max(fac_min, eta));
         dt = std::min(dt, dt_max);
      }

      // If not we can update the navier solver and queue new time step in the history array
      current_time += previous_dt;
      navier.UpdateTimestepHistory(previous_dt);
      elasticity.UpdateTimestepHistory(solid_displacement, solid_vel);
      subiter.Append(iter);

      // --------------------------------------------------------------------------------- //
      //// ----- 8.. Deform mesh, Update Navier solver and InterfaceTransfer objects  -----
      // --------------------------------------------------------------------------------- //
      if (Mpi::Root() && verbose)
         mfem::out << "\nDeforming mesh...";

      // If we're here it means that the last step converged
      // Therefore the mesh_displacement doesn't need to be re-computed, but we can just deform the mesh
      if (fluid_mesh_deformation->IsNonIncremental())
      { // mesh_displacement_gf is an absolute displacement
         fluid_mesh_deformation->DeformMeshWithDisplacement(*mesh_displacement_gf);
      }
      else
      { // mesh_displacement_gf is a relative displacement
         fluid_mesh_deformation->DeformMeshWithRelativeDisplacement(*mesh_displacement_gf);
      }

      if (Mpi::Root() && verbose)
         mfem::out << " Moving fluid mesh |dx|_l2 = " << std::scientific << std::setprecision(3)
                   << fluid_mesh_displacement_norm << "" << std::endl;

      if (Mpi::Root() && verbose)
         mfem::out << "Updating Navier-Stokes with new mesh deformation...";

      navier.Update();
      fluid_mesh_deformation->Update();
      // elasticity.Update(); --> Not needed, as we don't deform the solid mesh

      if (Mpi::Root() && verbose)
         mfem::out << "Updating InterfaceTransfer with new mesh deformation...";

      // TODO: check this step as it may give some errors
      // Needed because the the fluid is one of the directions of the transfer in both InterfaceTransfer objects
      // mesh_displacement_gf.Add(previous_dt, solid_vel_gf);
      finder_fluid_to_solid.Update(true, true, mesh_displacement_coeff, nullptr);
      finder_solid_to_mesh.Update();

      // --------------------------------------------------------------------------------- //

      // Output
      if ((step + 1) % save_freq == 0 || last_step)
      {
         if (Mpi::Root() && verbose)
            mfem::out << "\nWriting output to ParaView...";
         paraview_step++;
         pvdc_fluid.SetCycle(paraview_step);
         pvdc_fluid.SetTime(current_time);
         pvdc_fluid.Save();
         pvdc_solid.SetCycle(paraview_step);
         pvdc_solid.SetTime(current_time);
         pvdc_solid.Save();
         if (Mpi::Root() && verbose)
            mfem::out << " done." << std::endl;
      }

      // Output some information
      bool compute_output = true;
      real_t disp_x = 0.0, disp_y = 0.0;
      real_t p_probe1 = 0.0, p_probe2 = 0.0, delta_p = 0.0;
      real_t vel_norm1 = 0.0, vel_norm2 = 0.0;
      if (compute_output) // TODO: change with fixes from test_csm_only
      {
         Vector point1(2), point2(2);
         point1(0) = Mesh_ctx.x0 + 0.15; // Cylinder left vertex (fixed)
         point1(1) = Mesh_ctx.tip_y;
         point2(0) = Mesh_ctx.tip_x; // Beam right vertex (free) --> Note this will move with the mesh
         point2(1) = Mesh_ctx.tip_y;
         DenseMatrix points(dim, 2);
         points.SetCol(0, point1);
         points.SetCol(1, point2);

         Array<int> elem_ids;
         Array<IntegrationPoint> ips;

         fluid_pmesh.FindPoints(points, elem_ids, ips, false);

         // Initialize values to zero
         vel_norm1 = 0.0;
         vel_norm2 = 0.0;
         p_probe1 = 0.0;
         p_probe2 = 0.0;

         Vector val;

         // Only evaluate grid functions if the points were found on this rank
         if (elem_ids[0] >= 0)
         {
            u_gf->GetVectorValue(elem_ids[0], ips[0], val);
            vel_norm1 = val.Norml2();
            p_probe1 = p_gf->GetValue(elem_ids[0], ips[0]);
         }

         if (elem_ids[1] >= 0)
         {
            u_gf->GetVectorValue(elem_ids[1], ips[1], val);
            vel_norm2 = val.Norml2();
            p_probe2 = p_gf->GetValue(elem_ids[1], ips[1]);
         }

         // Reduce values across all ranks to get the result from whichever rank found the point
         real_t global_vals[4] = {vel_norm1, vel_norm2, p_probe1, p_probe2};
         real_t reduced_vals[4];
         MPI_Allreduce(global_vals, reduced_vals, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

         vel_norm1 = reduced_vals[0];
         vel_norm2 = reduced_vals[1];
         p_probe1 = reduced_vals[2];
         p_probe2 = reduced_vals[3];

         delta_p = p_probe1 - p_probe2;

         DenseMatrix points_solid(dim, 1);
         points_solid.SetCol(0, point2);
         solid_pmesh.FindPoints(points_solid, elem_ids, ips);

         // Initialize displacement values
         disp_x = 0.0;
         disp_y = 0.0;

         // Only evaluate grid functions if the point was found on this rank
         if (elem_ids[0] >= 0)
         {
            solid_displacement_gf.GetVectorValue(elem_ids[0], ips[0], val);
            disp_x = val(0);
            disp_y = val(1);
         }

         // Reduce displacement values across all ranks
         real_t global_disp_vals[2] = {disp_x, disp_y};
         real_t reduced_disp_vals[2];
         MPI_Allreduce(global_disp_vals, reduced_disp_vals, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

         disp_x = reduced_disp_vals[0];
         disp_y = reduced_disp_vals[1];

         ComputeDragLift(lift, drag, lift_drag_form);

         // Add to history
         lift_history.Append(lift);
         drag_history.Append(drag);
         delta_p_history.Append(delta_p);
      }

      if (Mpi::Root())
      {
         out << std::left
             << std::setw(8) << step
             << std::setw(16) << std::scientific << std::setprecision(8) << current_time
             << std::setw(16) << std::scientific << std::setprecision(8) << previous_dt
             << std::setw(16) << std::scientific << std::setprecision(8) << lift
             << std::setw(16) << std::scientific << std::setprecision(8) << drag
             << std::setw(16) << std::scientific << std::setprecision(8) << cfl
             << std::setw(16) << std::scientific << std::setprecision(8) << vel_norm1
             << std::setw(16) << std::scientific << std::setprecision(8) << delta_p
             << std::setw(16) << std::scientific << std::setprecision(8) << disp_x
             << std::setw(16) << std::scientific << std::setprecision(8) << disp_y
             << std::setw(16) << std::scientific << std::setprecision(8) << iter
             << std::endl;
      }

   } // END OF TIME-INTEGRATION LOOP

   navier.PrintTimingData();

   // Export the array of sub-iterations
   saveArray(subiter, outfolder, "subiter_count.txt");
   saveArray(omega_history_fluid, outfolder, "omega.txt");
   saveArray(omega_history_solid, outfolder, "omega.txt");

   saveArray(lift_history, outfolder, "lift.txt");
   saveArray(drag_history, outfolder, "drag.txt");
   saveArray(delta_p_history, outfolder, "delta_p.txt");

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 9. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   delete mesh_displacement_coeff;
   delete mesh_displacement_gf;

   if (!fluid_mesh_deformation->IsNonIncremental())
   {
      delete solid_relative_displacement_gf; // Relative solid displacement field
   }

   delete fluid_mesh_deformation;


   return 0;
}

void ComputeDragLift(real_t &drag, real_t &lift, ParLinearForm &lift_drag_form)
{
   // Assuming that
   // - u_gf, p_gf, nu_gf have been already updated with the current velocity, pressure, and viscosity
   // - sigmaN_gf has been already updated with the current stress on the fluid-solid interface

   // NOTE:
   // This function computes the drag and lift forces on the fluid-solid interface.
   // This assumes that we have a 2D problem with the fluid on top and the solid below.
   // This way the normal component is in the y-direction and the tangential component is in the x-direction.

   // Compute drag and lift forces
   // Drag = ∫ Γfs σ·n dx
   // Lift = ∫ Γfs σ·t dx
   // where t is the tangential vector to the boundary

   // CHECK what is the ordering of lift_drag_form?
   lift_drag_form.Assemble();

   lift = 0.0;
   drag = 0.0;
   for (int i = 0; i < lift_drag_form.FESpace()->GetNDofs(); i++)
   {
      int idx_x = Ordering::Map<Ordering::byNODES>(
          lift_drag_form.FESpace()->GetNDofs(),
          lift_drag_form.FESpace()->GetVDim(),
          i,
          0);

      int idx_y = Ordering::Map<Ordering::byNODES>(
          lift_drag_form.FESpace()->GetNDofs(),
          lift_drag_form.FESpace()->GetVDim(),
          i,
          1);

      // Lift is in the x-direction, drag is in the y-direction
      lift += lift_drag_form[idx_x];
      drag += lift_drag_form[idx_y];
   }

   // The local linear-form entries of each rank only contain the contribution
   // of its own boundary elements: sum them over all ranks.
   real_t forces[2] = {lift, drag};
   MPI_Allreduce(MPI_IN_PLACE, forces, 2, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 lift_drag_form.ParFESpace()->GetComm());
   lift = forces[0];
   drag = forces[1];
}

void noSlip_fun(const Vector &x, real_t t, Vector &u)
{
   u = 0.0;
}

real_t zero_pressure(const Vector &x, real_t t)
{
   return 0.0;
}

void inflow_fun(const Vector &x, real_t time, Vector &v)
{
   // NOTE: We include a negative sign because in the FSI coupling the traction is defined as -σF·n, and for
   // the preload we're using the same gridfunction
   const real_t preloadT = NS_ctx.preloadT;
   const real_t preload = (time < preloadT) ? 0.5 * (1.0 - cos(M_PI * time / preloadT)) : 1.0;
   // traction in y-direction
   real_t xi = x[0];
   real_t yi = x[1] - Mesh_ctx.y0; // distance from the bottom wall

   v = 0.0;
   v[0] = preload * (NS_ctx.scaling * NS_ctx.Umax * yi * (Mesh_ctx.D - yi));
   v[1] = 0.0;
}

void gravitational_force(const Vector &x, real_t time, Vector &u)
{
   u(1) = -Elasticity_ctx.g * Elasticity_ctx.density;
   u(0) = 0.0;
};

template <typename T>
void saveArray(const Array<T> &data, const std::string &outfolder, const std::string &name)
{
   // Create the output folder path + name
   std::string outputFolder = outfolder + "/Convergence/";

   if (!fs::is_directory(outputFolder.c_str()) || !fs::exists(outputFolder.c_str()))
   {                                        // Check if folder exists
      fs::create_directories(outputFolder); // create folder
   }

   // Construct the filename
   std::ostringstream filename;
   filename << outputFolder << name;

   // Open the file and save the data
   std::ofstream outFile(filename.str());
   if (outFile.is_open())
   {
      data.Save(outFile);
      outFile.close();
   }
   else
   {
      std::cerr << "Unable to open file: " << filename.str() << std::endl;
   }
}