// =====================================================================================
//          3D FSI Example: Pulsatile Flow in a Deformable Cylinder
// -------------------------------------------------------------------------------------
// Sample runs:
//
// 1. FSI (Robin-Neumann coupling, pulsed inlet pressure)
// mpirun -np 4 ./navier-fsi-cylinder -rs 0 --relaxation 1.0 --time-adaptivity -dt 1e-4 --dt-max 1e-3 --aitken -fo 4 -eo 2 -sf 5 --coupling 1 -of ./Output/FSI-Cylinder
//
// 2. CFD only (Rigid cylinder wall)
// mpirun -np 4 ./navier-fsi-cylinder --cfd-only -rs 0 --time-adaptivity -dt 1e-4 -fo 4 -of ./Output/CFD-Cylinder
//
// 3. CSM only (Cylinder under internal pressure)
// mpirun -np 4 ./navier-fsi-cylinder --csm-only -rs 0 -dt 1e-3 -eo 2 -of ./Output/CSM-Cylinder
//
// 3D FSI Example: Pulsatile flow in a deformable cylindrical tube.
//
// Setup: Fluid fills the inner cylinder. The solid forms the compliant cylindrical wall.
// A pressure pulse is applied at the inlet and propagates as a wave along the tube.
// The Moens-Korteweg wave speed is c = sqrt(E*h / (2*rho_f*R)).
//
// Mesh attributes:
//   Volumes:  1 = fluid (inner cylinder),  2 = solid (cylindrical shell)
//   Surfaces: 1 = inlet (fluid inflow),    2 = outlet (fluid outflow, p=0)
//             3 = wall (no-slip rigid wall, if present in fluid domain)
//             4 = solid rings (fixed ends of the shell)
//             5 = interface (fluid-solid coupling surface)
//             6 = solid outer wall (traction-free outer surface of shell)
//
//
//          z=0                          z=L
//          |                             |
//          |   +=========================+
//          |   |                         |
// ---------|---+                         |
//  Pulse-->|   |  ////// Fluid //////    |---> outflow (p=0)
//  p(t)    |---+                         |
// ---------|   |                         |
//          |   +=========================+
//
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

// Forward declarations
template <typename T>
void saveArray(const Array<T> &data, const std::string &outfolder,
               const std::string &name);

void noSlip_fun(const Vector &x, real_t time, Vector &u);
real_t zero_pressure(const Vector &x, real_t t);
real_t pressure_pulse_fun(const Vector &x, real_t t);
void internal_pressure_bc(const Vector &x, real_t time, Vector &u);

enum class CouplingType : int
{
   Dirichlet_Neumann = 0,
   Robin_Neumann = 1,
   RobinAMP_Neumann = 2
};

struct s_NavierContext
{
   int u_order = 2;
   int max_bdf_order =
      2;             // BDF3 + undamped solid is unstable (added mass)
   real_t density = 1.0;              // ρ_F = 1.0 g/cm³
   real_t dynamic_viscosity = 0.03;   // μ = 0.03 Poise = 0.03 g/(cm·s)
   real_t kinematic_viscosity = 0.03; // ν = μ/ρ = 0.03 cm²/s
   int filter_type = 0;
} NS_ctx;

struct s_ElasticityContext
{
   int order = 2;
   real_t density = 1.2;              // ρ_S = 1.2 g/cm³
   real_t nu = 0.3;
   real_t E = 3.0e6;                  // E = 3·10⁶ dyn/cm²
   real_t mu;                         // = 1.1538e6 dyn/cm²
   real_t lambda;                     // = 1.7308e6 dyn/cm²
   real_t p_internal = 1.3332e5;      // ~100 mmHg in dyn/cm²
   real_t tp_ramp = 5e-3;  // ramp-up duration, ~5 periods of the breathing mode
   real_t tp_stop = 20e-3; // hold pressure until t_final/2 or whatever you choose
} Elasticity_ctx;

struct s_MeshContext
{
   real_t R_inner = 0.5;
   real_t R_outer = 0.6;
   real_t L = 5.0;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   int order = -1;
   real_t E = 1e6;                    // dyn/cm² for mesh deformation
   real_t nu = 0.35;
   int stiffening_type = 1;
   bool enable_stiffening = false;
   MeshDeformationType mesh_deformation_type = MeshDeformationType::Laplacian;
} Mesh_ctx;

struct s_PulseContext
{
   real_t amplitude = 1.3332e4;       // 1.3332·10⁴ dyn/cm²
   real_t ramp_time = 0.0;
   real_t end_time = 3e-3;
} Pulse_ctx;

struct s_DomainDecompositionContext
{
   CouplingType coupling = CouplingType::Dirichlet_Neumann;
   bool aitken = false;
   int robin_traction =
      0;   // Traction in the fluid Robin datum: 0 - fluid traction (consistent), 1 - solid sigma_s(u_h).n
   real_t safety_factor = 1.15;
   real_t omega = 0.5;
} DD_ctx;

const char *outfolder = "./Output/FSI-Cylinder";
int num_procs;
real_t dt;

class BoundaryNormalCoefficient : public VectorCoefficient
{
public:
   BoundaryNormalCoefficient(int dim, real_t scalar_) : VectorCoefficient(dim),
      scalar(scalar_) {}

   /// Evaluate the vector coefficient at @a ip.
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      int sdim = T.GetSpaceDim();

      T.SetIntPoint(&ip);

      // Compute normal
      nor.SetSize(sdim);
      CalcOrtho(T.Jacobian(), nor);
      const real_t scale = nor.Norml2();
      nor *= (1.0 / scale);

      // Return scaled normal
      V.SetSize(sdim);
      V = nor;
      V *= scalar;
   }

   void SetScalar(real_t scalar_) { scalar = scalar_; }

private:
   real_t scalar;
   mutable Vector nor;
};

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
   /// 2. Parse command-line options
   ///////////////////////////////////////////////////////////////////////////////////////////////

   const char *mesh_file = "../../data/cylinder-fsi.e";

   bool cfd_rigid = false;        // CFD only with rigid wall
   bool csm_only = false;         // CSM only with internal pressure load

   // Timestepping
   real_t current_time = 0.0;
   real_t t_final = 0.5;          // ~half a second (multiple pulse cycles)
   dt = 1e-4;
   real_t dt_max = 5e-3;
   real_t cfl_target = 0.8;
   real_t cfl_tol = 1e-1;
   bool time_adaptivity = false;
   real_t relaxation = 1.0;

   // Output
   int save_freq = 1;
   int current_step = 0;
   bool last_step = false;
   bool verbose = false;

   OptionsParser args(argc, argv);
   args.AddOption(&cfd_rigid, "-cfd-only", "--cfd-only", "-no-cfd-only",
                  "--no-cfd-only",
                  "Run only the CFD part (rigid wall, no solid mechanics).");
   args.AddOption(&csm_only, "-csm-only", "--csm-only", "-no-csm-only",
                  "--no-csm-only",
                  "Run only the solid mechanics part (internal pressure load).");
   // FE orders
   args.AddOption(&NS_ctx.u_order, "-fo", "--fluid-order",
                  "Polynomial order for fluid velocity space.");
   args.AddOption(&Elasticity_ctx.order, "-eo", "--elasticity-order",
                  "Polynomial order for solid elasticity space.");
   // Navier-Stokes
   args.AddOption(&NS_ctx.max_bdf_order, "-mbdf", "--max-bdf-order",
                  "Maximum BDF order for fluid time integration.");
   args.AddOption(&NS_ctx.density, "-rho-f", "--rho-fluid",
                  "Density of the fluid [kg/m³] (default: 1e3).");
   args.AddOption(&NS_ctx.dynamic_viscosity, "-visc", "--viscosity",
                  "Dynamic viscosity of the fluid [Pa·s] (default: 4e-3).");
   args.AddOption(&NS_ctx.filter_type, "-filter", "--filter",
                  "Filtering: 0=None, 1=HPFRT_LOS, 2=HPFRT_LPOLY.");
   // Elasticity
   args.AddOption(&Elasticity_ctx.E, "-E", "--young-modulus",
                  "Young's modulus of the solid [Pa] (default: 5e5).");
   args.AddOption(&Elasticity_ctx.nu, "-nu", "--poisson-ratio",
                  "Poisson's ratio of the solid (default: 0.45).");
   args.AddOption(&Elasticity_ctx.density, "-rho-s", "--rho-solid",
                  "Density of the solid [kg/m³] (default: 1.1e3).");
   args.AddOption(&Elasticity_ctx.p_internal, "-p-int", "--internal-pressure",
                  "Internal pressure for CSM-only test [Pa] (default: 1.33e4).");
   // Pulse parameters
   args.AddOption(&Pulse_ctx.amplitude, "-pa", "--pulse-amplitude",
                  "Pressure pulse amplitude [Pa] (default: 1.33e4).");
   args.AddOption(&Pulse_ctx.end_time, "-ptend", "--pulse-end-time",
                  "End time for pulse [s] (default: 3e-3).");
   args.AddOption(&Pulse_ctx.ramp_time, "-rt", "--ramp-time",
                  "Amplitude ramp-up time [s] (default: 0, no ramp).");
   // Time integration
   args.AddOption(&t_final, "-tf", "--t-final", "Final simulation time [s].");
   args.AddOption(&dt, "-dt", "--dt", "Initial time step size [s].");
   args.AddOption(&dt_max, "-dtmax", "--dt-max", "Maximum time step size [s].");
   args.AddOption(&time_adaptivity, "-ta", "--time-adaptivity",
                  "-no-ta", "--no-time-adaptivity",
                  "Enable/disable time adaptivity.");
   // FSI coupling
   args.AddOption((int *)&DD_ctx.coupling, "-c", "--coupling",
                  "Coupling: 0=Dirichlet-Neumann, 1=Robin-Neumann, 2=RobinAMP-Neumann.");
   args.AddOption(&relaxation, "-rel", "--relaxation",
                  "Relaxation factor (default: 1.0).");
   args.AddOption(&DD_ctx.aitken, "-aa", "--aitken", "-no-aa", "--no-aitken",
                  "Enable/disable Aitken relaxation.");
   args.AddOption(&DD_ctx.robin_traction, "-rt", "--robin-traction",
                  "Traction used in the fluid Robin datum (Robin couplings): 0 - fluid traction sigma_f.n of the "
                  "previous sub-iterate, i.e. the load applied to the solid (default), 1 - solid traction "
                  "sigma_s(u_h).n from the displacement gradient.");
   args.AddOption(&DD_ctx.safety_factor, "-sfa", "--safety-factor",
                  "Safety factor for Aitken relaxation (default: 1.15).");
   // Mesh
   args.AddOption(&Mesh_ctx.serial_ref_levels, "-rs", "--rs",
                  "Serial refinement levels.");
   args.AddOption(&Mesh_ctx.parallel_ref_levels, "-pr", "--pr",
                  "Parallel refinement levels.");
   args.AddOption((int *)&Mesh_ctx.mesh_deformation_type, "-mdt",
                  "--mesh-deformation-type",
                  "Mesh deformation: 0=Laplacian, 1=IncrementalLaplacian, 2=Elasticity.");
   args.AddOption(&Mesh_ctx.order, "-mo", "--mesh-order",
                  "Polynomial order of mesh nodes (-1: use elasticity order).");
   args.AddOption(&Mesh_ctx.enable_stiffening, "-es", "--enable-stiffening",
                  "-no-es", "--no-enable-stiffening",
                  "Enable local mesh stiffening near the interface.");
   args.AddOption(&Mesh_ctx.stiffening_type, "-st", "--stiffening-type",
                  "Stiffening type: 0=Jacobian-based, 1=Distance-based.");
   // Output
   args.AddOption(&save_freq, "-sf", "--save-freq", "ParaView save frequency.");
   args.AddOption(&outfolder, "-of", "--output-folder", "Output folder.");
   args.AddOption(&verbose, "-v", "--verbose", "-no-v", "--no-verbose",
                  "Verbose output during time integration.");
   args.ParseCheck();

   // Compute Lame's parameters
   Elasticity_ctx.mu = Elasticity_ctx.E / (2.0 * (1.0 + Elasticity_ctx.nu));
   Elasticity_ctx.lambda = Elasticity_ctx.E * Elasticity_ctx.nu /
                           ((1.0 + Elasticity_ctx.nu) * (1.0 - 2.0 * Elasticity_ctx.nu));

   // Kinematic viscosity
   NS_ctx.kinematic_viscosity = NS_ctx.dynamic_viscosity / NS_ctx.density;

   DD_ctx.omega = relaxation;

   MFEM_VERIFY(!(cfd_rigid && csm_only),
               "Only one of --cfd-only, --csm-only can be selected.");

   // Estimate Moens-Korteweg wave speed for reference
   real_t h_wall = Mesh_ctx.R_outer - Mesh_ctx.R_inner;
   real_t c_MK = std::sqrt(Elasticity_ctx.E * h_wall /
                           (2.0 * NS_ctx.density * Mesh_ctx.R_inner));
   if (Mpi::Root())
   {
      mfem::out << "\nMoens-Korteweg wave speed estimate: c_MK = "
                << std::scientific << std::setprecision(4) << c_MK << " m/s" << std::endl;
      mfem::out << "Expected wave transit time: L/c_MK = "
                << std::scientific << std::setprecision(4) << Mesh_ctx.L / c_MK << " s\n" <<
                std::endl;
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create serial mesh, parallel mesh, and fluid/solid sub-meshes
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
   {
      mfem::out << "Loading mesh and refining in serial...";
   }

   Mesh *parent_mesh = new Mesh(mesh_file);
   const int dim = parent_mesh->Dimension();

   MFEM_VERIFY(dim == 3, "This example requires a 3D mesh.");

   for (int i = 0; i < Mesh_ctx.serial_ref_levels; i++)
   {
      parent_mesh->UniformRefinement();
   }

   Mesh_ctx.order = Mesh_ctx.order < 0 ? Elasticity_ctx.order : Mesh_ctx.order;
   parent_mesh->SetCurvature(Mesh_ctx.order, false);

   if (Mpi::Root())
   {
      mfem::out << " done.\nPartitioning and creating parallel mesh..." << std::endl;
   }

   int partition_type = 1;
   int np = num_procs;
   int *partitioning = parent_mesh->GeneratePartitioning(np, partition_type);

   ParMesh pmesh(MPI_COMM_WORLD, *parent_mesh, partitioning, partition_type);
   ExportMeshwithPartitioning(outfolder, *parent_mesh, partitioning,
                              "Partitioning");
   delete[] partitioning;

   for (int l = 0; l < Mesh_ctx.parallel_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   pmesh.SetAttributes();
   pmesh.EnsureNodes();

   //
   // Volume attributes: 1 = fluid, 2 = solid
   //
   Array<int> fluid_domain_attribute, solid_domain_attribute;
   fluid_domain_attribute.SetSize(1); fluid_domain_attribute[0] = 1;
   solid_domain_attribute.SetSize(1); solid_domain_attribute[0] = 2;

   auto solid_pmesh = ParSubMesh::CreateFromDomain(pmesh, solid_domain_attribute);
   auto fluid_pmesh = ParSubMesh::CreateFromDomain(pmesh, fluid_domain_attribute);

   if (Mpi::Root())
   {
      mfem::out << "done.\nSetting up boundary markers..." << std::endl;
   }

   //
   // Boundary attributes (using numeric IDs):
   //   1 = inlet (fluid)    2 = outlet (fluid)   3 = wall (rigid, fluid side)
   //   4 = solid rings      5 = interface        6 = solid outer wall
   //
   const int n_bdr = pmesh.bdr_attributes.Max();

   Array<int> fluid_inlet(1),    fluid_outlet(1),   fluid_wall(1);
   Array<int> solid_rings(1),    interface_attr(1),  solid_outer_wall(1);
   fluid_inlet[0]      = 1;
   fluid_outlet[0]     = 2;
   fluid_wall[0]       = 3;
   solid_rings[0]      = 4;
   interface_attr[0]   = 5;
   solid_outer_wall[0] = 6;

   // Convert to marker arrays (size = max bdr attribute)
   Array<int> fluid_inlet_marker      = AttributeSets::AttrToMarker(n_bdr,
                                                                    fluid_inlet);
   Array<int> fluid_outlet_marker     = AttributeSets::AttrToMarker(n_bdr,
                                                                    fluid_outlet);
   Array<int> fluid_wall_marker       = AttributeSets::AttrToMarker(n_bdr,
                                                                    fluid_wall);
   Array<int> solid_rings_marker      = AttributeSets::AttrToMarker(n_bdr,
                                                                    solid_rings);
   Array<int> interface_marker        = AttributeSets::AttrToMarker(n_bdr,
                                                                    interface_attr);
   Array<int> solid_outer_wall_marker = AttributeSets::AttrToMarker(n_bdr,
                                                                    solid_outer_wall);

   // The interface marker on the parent mesh is needed by BidirectionalInterfaceTransfer
   Array<int> parent_interface_marker = AttributeSets::AttrToMarker(n_bdr,
                                                                    interface_attr);

   delete parent_mesh;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Create Flow, Solid, and MeshDeformation solvers
   ///////////////////////////////////////////////////////////////////////////////////////////////

   ///////////////
   // FLOW
   ///////////////

   if (Mpi::Root())
   {
      mfem::out << "\nCreating Navier-Stokes solver..." << std::endl;
      mfem::out << "  Fluid density:              " << std::scientific <<
                NS_ctx.density << " kg/m³\n";
      mfem::out << "  Dynamic viscosity:          " << std::scientific <<
                NS_ctx.dynamic_viscosity << " Pa·s\n";
      mfem::out << "  Kinematic viscosity:        " << std::scientific <<
                NS_ctx.kinematic_viscosity << " m²/s\n";
   }

   NavierSolver navier(&fluid_pmesh, NS_ctx.u_order, NS_ctx.kinematic_viscosity,
                       NS_ctx.density, false);
   navier.SetMaxBDFOrder(NS_ctx.max_bdf_order);

   switch (NS_ctx.filter_type)
   {
      case 0: break;
      case 1:
         navier.EnableFilter(NavierSolver::FilterMethod::HPFRT_LOS);
         navier.SetCutoffModes(1);
         navier.SetFilterAlpha(0.5);
         break;
      case 2:
         navier.EnableFilter(NavierSolver::FilterMethod::HPFRT_LPOLY);
         navier.SetCutoffModes(1);
         navier.SetFilterAlpha(10.0);
         break;
      default:
         MFEM_ABORT("Invalid filter type. Use 0=None, 1=HPFRT_LOS, 2=HPFRT_LPOLY.");
   }

   ParGridFunction *u_gf       = navier.GetCurrentVelocity();
   ParGridFunction *un_next_gf = navier.GetProvisionalVelocity();
   ParGridFunction *p_gf       = navier.GetCurrentPressure();
   ParGridFunction *nu_gf      = navier.GetVariableViscosity();

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
   {
      mfem::out << "\nCreating Elasticity solver..." << std::endl;
      mfem::out << "  Young's modulus:            " << std::scientific <<
                Elasticity_ctx.E << " Pa\n";
      mfem::out << "  Poisson's ratio:            " << Elasticity_ctx.nu << "\n";
      mfem::out << "  Density:                    " << std::scientific <<
                Elasticity_ctx.density << " kg/m³\n";
      mfem::out << "  Shear modulus (mu):         " << std::scientific <<
                Elasticity_ctx.mu << " Pa\n";
      mfem::out << "  Lame lambda:                " << std::scientific <<
                Elasticity_ctx.lambda << " Pa\n";
   }

   int solver_type = 15; // HHTAlphaSolver
   ElasticitySolver elasticity(solver_type, &solid_pmesh, Elasticity_ctx.order,
                               Elasticity_ctx.density, Elasticity_ctx.lambda, Elasticity_ctx.mu);

   auto elasticity_fes = elasticity.GetFESpace();

   Vector solid_displacement, solid_displacement_tn;
   Vector solid_vel, solid_vel_tn;

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

   int order_quad_disp = std::max(2, 2 * Elasticity_ctx.order + 2);
   const IntegrationRule *irs_disp[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs_disp[i] = &(IntRules.Get(i, order_quad_disp));
   }

   /////////////////////////
   // MESH DEFORMATION  (skipped for CSM-only)
   /////////////////////////

   MeshDeformation *fluid_mesh_deformation              = nullptr;
   ParFiniteElementSpace *mesh_nodes_fes                = nullptr;
   ParGridFunction *mesh_velocity_gf                    = nullptr;
   ParGridFunction *mesh_displacement_gf                = nullptr;
   VectorGridFunctionCoefficient *mesh_displacement_coeff = nullptr;
   ParGridFunction *solid_relative_displacement_gf      = nullptr;
   Vector mesh_displacement_tn;

   if (!csm_only && !cfd_rigid)
   {
      if (Mpi::Root())
      {
         mfem::out << "\nCreating mesh deformation solver...";
      }

      switch (Mesh_ctx.mesh_deformation_type)
      {
         case MeshDeformationType::Laplacian:
            fluid_mesh_deformation = new MeshLaplacian(&fluid_pmesh);
            break;
         case MeshDeformationType::IncrementalLaplacian:
            fluid_mesh_deformation = new MeshLaplacian(&fluid_pmesh, true);
            break;
         case MeshDeformationType::LinearElasticity:
            fluid_mesh_deformation = new MeshLinearElasticity(&fluid_pmesh);
            break;
         default:
            MFEM_ABORT("Invalid mesh deformation type.");
      }

      std::vector<std::string> mesh_deformation_names = {"Laplacian", "IncrementalLaplacian", "LinearElasticity"};
      if (Mpi::Root())
      {
         mfem::out << " Using: " << mesh_deformation_names[static_cast<int>
                                                           (Mesh_ctx.mesh_deformation_type)] << std::endl;
      }

      mesh_nodes_fes       = fluid_mesh_deformation->GetFESpace();
      mesh_velocity_gf     = new ParGridFunction(mesh_nodes_fes);
      *mesh_velocity_gf    = 0.0;
      // ALE convection (u - w)·grad u: the BDF time derivative is taken at the (moving) mesh nodes, so the fluid
      // solver needs the mesh velocity w. Must be set before navier.Setup(), which builds the convection
      // integrator. Not owned: deleted at the end of main().
      navier.SetMeshVelocity(mesh_velocity_gf, false);
      mesh_displacement_gf = new ParGridFunction(mesh_nodes_fes);
      mesh_displacement_coeff = new VectorGridFunctionCoefficient(
         &solid_displacement_gf);
      *mesh_displacement_gf = 0.0;
      mesh_displacement_gf->GetTrueDofs(mesh_displacement_tn);

      if (!fluid_mesh_deformation->IsNonIncremental())
      {
         solid_relative_displacement_gf = new ParGridFunction(elasticity_fes);
         *solid_relative_displacement_gf = 0.0;
      }
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Add boundary conditions and initial conditions
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
   {
      mfem::out << "\nSetting up boundary conditions..." << std::endl;
   }

   ////////////
   // FLUID  (skipped for CSM-only)
   ////////////

   VectorGridFunctionCoefficient solid_vel_onFluid_coeff(&solid_vel_onFluid_gf);
   VectorGridFunctionCoefficient solid_sigmaN_onFluid_coeff(
      &solid_sigmaN_onFluid_gf);
   real_t cp = 0.0, cs = 0.0, zp = 0.0, zs = 0.0;

   if (!csm_only)
   {
      // - Pressure pulse at inlet (1):      p = P_pulse(t)
      // - Traction-free at outlet (2):      p = 0
      // - No-slip at rigid wall (3):        u = 0
      // - FSI coupling at interface (5):    u = u_solid  (Dirichlet or Robin)

      const int n_bdr_fluid = fluid_pmesh.bdr_attributes.Max();
      Array<int> f_inlet_marker  = AttributeSets::AttrToMarker(n_bdr_fluid,
                                                               fluid_inlet);
      Array<int> f_outlet_marker = AttributeSets::AttrToMarker(n_bdr_fluid,
                                                               fluid_outlet);
      Array<int> f_wall_marker   = AttributeSets::AttrToMarker(n_bdr_fluid,
                                                               fluid_wall);
      Array<int> f_iface_marker  = AttributeSets::AttrToMarker(n_bdr_fluid,
                                                               interface_attr);

      navier.AddPresDirichletBC(pressure_pulse_fun, f_inlet_marker);
      //navier.AddPresDirichletBC(zero_pressure, f_outlet_marker);   -->enforced weakly
      navier.AddVelDirichletBC(noSlip_fun, f_wall_marker);

      cp = std::sqrt((Elasticity_ctx.lambda + 2.0 * Elasticity_ctx.mu) /
                     Elasticity_ctx.density);
      cs = std::sqrt(Elasticity_ctx.mu / Elasticity_ctx.density);
      zp = Elasticity_ctx.density * cp;
      zs = Elasticity_ctx.density * cs;

      switch (DD_ctx.coupling)
      {
         case CouplingType::Dirichlet_Neumann:
            navier.AddVelDirichletBC(&solid_vel_onFluid_coeff, f_iface_marker, false);
            break;
         case CouplingType::Robin_Neumann:
            navier.AddRobinBC(zp, &solid_vel_onFluid_coeff, &solid_sigmaN_onFluid_coeff,
                              f_iface_marker, false);
            break;
         case CouplingType::RobinAMP_Neumann:
            navier.AddAMPRobinBC(zs, zp, &solid_vel_onFluid_coeff,
                                 &solid_sigmaN_onFluid_coeff, f_iface_marker, false);
            break;
         default:
            break;
      }

      navier.Setup(dt);
   }

   ////////////
   // SOLID
   ////////////
   // - Fixed at solid rings (4):           u = 0  (clamped ends)
   // - Traction from fluid at interface (5): Neumann from fluid
   // - Traction-free at outer wall (6):    natural BC (no action needed)

   const int n_bdr_solid = solid_pmesh.bdr_attributes.Max();
   Array<int> s_rings_marker = AttributeSets::AttrToMarker(n_bdr_solid,
                                                           solid_rings);
   Array<int> s_iface_marker = AttributeSets::AttrToMarker(n_bdr_solid,
                                                           interface_attr);

   // Fixed ends: both inlet and outlet rings are clamped
   // If both rings share the same attribute (4), one call suffices
   elasticity.AddFixedConstraint(solid_rings[0]);

   // Traction from fluid on the interface (FSI/CFD only)
   // Sign: -1.0 because n_fluid = -n_solid at the shared interface
   // CSM-only: apply internal pressure on the inner wall (interface surface)
   // representing a static/quasi-static internal pressurization test
   VectorGridFunctionCoefficient fluid_sigmaN_onSolid_coeff(
      &fluid_sigmaN_onSolid_gf);
   BoundaryNormalCoefficient *internal_pressure_coeff = nullptr;
   if (!csm_only)
   {
      elasticity.AddBoundaryLoad(&fluid_sigmaN_onSolid_coeff, interface_attr[0], -1.0,
                                 false);
   }
   else
   {
      internal_pressure_coeff = new BoundaryNormalCoefficient(dim,
                                                              -Elasticity_ctx.p_internal);
      elasticity.AddBoundaryLoad(internal_pressure_coeff, interface_attr[0], true);
   }

   elasticity.Setup();

   ////////////////////
   // MESH DEFORMATION  (skipped for CSM-only or CFD-only)
   ////////////////////

   if (!csm_only && !cfd_rigid)
   {
      // Fix all outer boundaries; the interface displacement is set from the solid
      Array<int> fluid_mesh_deformation_ess_bdr(fluid_pmesh.bdr_attributes.Max());
      fluid_mesh_deformation_ess_bdr = 1;
      fluid_mesh_deformation->SetEssentialBoundary(fluid_mesh_deformation_ess_bdr);

      bool pa_mesh = false;
      int prec_type_mesh = 1;

      if (Mesh_ctx.enable_stiffening)
      {
         fluid_mesh_deformation->EnableLocalStiffening(parent_interface_marker,
                                                       Mesh_ctx.stiffening_type);
      }

      switch (Mesh_ctx.mesh_deformation_type)
      {
         case MeshDeformationType::Laplacian:
         case MeshDeformationType::IncrementalLaplacian:
            static_cast<MeshLaplacian *>(fluid_mesh_deformation)->Setup(1.0, pa_mesh,
                                                                        prec_type_mesh);
            break;
         case MeshDeformationType::LinearElasticity:
            static_cast<MeshLinearElasticity *>(fluid_mesh_deformation)->Setup(Mesh_ctx.E,
                                                                               Mesh_ctx.nu,
                                                                               pa_mesh, prec_type_mesh);
            break;
         default:
            MFEM_ABORT("Invalid mesh deformation type.");
      }
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Setup interface transfer  (skipped for CSM-only)
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Declared at outer scope so the FSI loop can capture them; all null for CSM-only.
   BidirectionalInterfaceTransfer *finder_fluid_to_solid_ptr = nullptr;
   InterfaceTransfer *finder_solid_to_mesh_ptr = nullptr;
   Array<int> solid_interfaces_element_idx;
   Array<int> fluid_interfaces_element_idx;

   // Work vectors / grid functions used only in the FSI convergence loop
   ParGridFunction *solid_vel_onFluid_prev_gf_ptr  = nullptr;
   ParGridFunction *fluid_sigmaN_onSolid_prev_gf_ptr = nullptr;
   ParGridFunction *solid_sigmaN_onFluid_prev_gf_ptr = nullptr;
   Vector solid_vel_onFluid, fluid_sigmaN_onSolid, solid_sigmaN_onFluid;
   Vector solid_vel_onFluid_prev, fluid_sigmaN_onSolid_prev,
          solid_sigmaN_onFluid_prev;

   if (!csm_only && !cfd_rigid)
   {
      if (Mpi::Root())
      {
         mfem::out << "\nSetting up interface transfer..." << std::endl;
      }

      finder_fluid_to_solid_ptr = new BidirectionalInterfaceTransfer(
         u_gf->ParFESpace(), elasticity_fes,
         parent_interface_marker, TransferBackend::GSLIB, pmesh.GetComm(), false, true);

      finder_solid_to_mesh_ptr = new InterfaceTransfer(
         solid_vel_gf.ParFESpace(), mesh_nodes_fes,
         parent_interface_marker, TransferBackend::GSLIB, pmesh.GetComm(), true);

      real_t boundary_distance_tolerance = 1e-4;
      finder_fluid_to_solid_ptr->SetBoundaryDistanceTolerance(
         boundary_distance_tolerance,
         boundary_distance_tolerance);
      finder_solid_to_mesh_ptr->SetBoundaryDistanceTolerance(
         boundary_distance_tolerance);

      finder_fluid_to_solid_ptr->GetElementIdxSrc(fluid_interfaces_element_idx);
      finder_fluid_to_solid_ptr->GetElementIdxDst(solid_interfaces_element_idx);

      // Work vectors for convergence checking and relaxation
      solid_vel_onFluid_prev_gf_ptr     = new ParGridFunction(u_gf->ParFESpace());
      fluid_sigmaN_onSolid_prev_gf_ptr  = new ParGridFunction(elasticity_fes);
      solid_sigmaN_onFluid_prev_gf_ptr  = new ParGridFunction(u_gf->ParFESpace());
      *solid_vel_onFluid_prev_gf_ptr    = 0.0;
      *fluid_sigmaN_onSolid_prev_gf_ptr = 0.0;
      *solid_sigmaN_onFluid_prev_gf_ptr = 0.0;

      solid_vel_onFluid_gf.GetTrueDofs(solid_vel_onFluid);
      solid_sigmaN_onFluid_gf.GetTrueDofs(solid_sigmaN_onFluid);
      fluid_sigmaN_onSolid_gf.GetTrueDofs(fluid_sigmaN_onSolid);
      solid_vel_onFluid_prev_gf_ptr->GetTrueDofs(solid_vel_onFluid_prev);
      fluid_sigmaN_onSolid_prev_gf_ptr->GetTrueDofs(fluid_sigmaN_onSolid_prev);
      solid_sigmaN_onFluid_prev_gf_ptr->GetTrueDofs(solid_sigmaN_onFluid_prev);
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Setup output (ParaView)
   ///////////////////////////////////////////////////////////////////////////////////////////////

   int paraview_step = 0;

   ParaViewDataCollection pvdc_fluid("Fluid", &fluid_pmesh);
   pvdc_fluid.SetPrefixPath(outfolder);
   pvdc_fluid.SetDataFormat(VTKFormat::BINARY);
   pvdc_fluid.SetHighOrderOutput(true);
   pvdc_fluid.SetCompression(true);
   pvdc_fluid.SetCompressionLevel(9);
   pvdc_fluid.SetLevelsOfDetail(NS_ctx.u_order);
   pvdc_fluid.SetCycle(current_step);
   pvdc_fluid.SetTime(0.0);
   pvdc_fluid.RegisterField("velocity",            u_gf);
   pvdc_fluid.RegisterField("provisional_velocity",
                            navier.GetProvisionalVelocity());
   pvdc_fluid.RegisterField("pressure",            p_gf);
   pvdc_fluid.RegisterField("interface_velocity",  &solid_vel_onFluid_gf);
   if (mesh_velocity_gf)
   {
      pvdc_fluid.RegisterField("mesh_velocity",    mesh_velocity_gf);
   }
   ParGridFunction *distance_gf = nullptr;
   if (!csm_only && Mesh_ctx.enable_stiffening && Mesh_ctx.stiffening_type == 1)
   {
      auto stiffening_coeff = fluid_mesh_deformation->GetLocalStiffeningCoefficient();
      distance_gf = static_cast<DistanceBasedStiffeningCoefficient *>
                    (stiffening_coeff)->GetDistanceGf();
      pvdc_fluid.RegisterField("distance", distance_gf);
   }
   if (!csm_only) { pvdc_fluid.Save(); }

   ParaViewDataCollection pvdc_solid("Solid", &solid_pmesh);
   pvdc_solid.SetPrefixPath(outfolder);
   pvdc_solid.SetDataFormat(VTKFormat::BINARY);
   pvdc_solid.SetHighOrderOutput(true);
   pvdc_solid.SetCompression(true);
   pvdc_solid.SetCompressionLevel(9);
   pvdc_solid.SetLevelsOfDetail(Elasticity_ctx.order);
   pvdc_solid.SetCycle(current_step);
   pvdc_solid.SetTime(0.0);
   pvdc_solid.RegisterField("displacement",      &solid_displacement_gf);
   pvdc_solid.RegisterField("velocity",          &solid_vel_gf);
   pvdc_solid.RegisterField("interface_sigmaN",  &fluid_sigmaN_onSolid_gf);
   pvdc_solid.Save();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Solve
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
   {
      out << "\n-----------------------------------------------------------------------"
          "----------------------------------------------------------------------\n";
      out << std::left
          << std::setw(8)  << "Step"
          << std::setw(16) << "Time"
          << std::setw(16) << "dt"
          << std::setw(16) << "p_inlet"
          << std::setw(16) << "p_outlet"
          << std::setw(16) << "CFL"
          << std::setw(16) << "disp_r_mid"
          << std::setw(16) << "subiter"
          << "\n";
      out << "-----------------------------------------------------------------------"
          "----------------------------------------------------------------------\n";
   }

   ////////////////////////////////////////////////////////////////////////////////////////////////
   /// 8.1 ----- CSM-only or CFD-only test
   ////////////////////////////////////////////////////////////////////////////////////////////////

   if (csm_only)
   {
      if (Mpi::Root())
      {
         mfem::out << "\nRunning CSM-only test (internal pressure)..." << std::endl;
      }

      bool load = true;

      for (int step = 0; !last_step; ++step)
      {
         if (current_time + dt >= t_final - dt / 2)
         {
            last_step = true;
         }

         if (load && current_time >= Elasticity_ctx.tp_stop)
         {
            load = false;
            if (Mpi::Root())
            {
               mfem::out << "Stopping internal pressure at t = " << current_time << " s\n";
            }
            internal_pressure_coeff->SetScalar(0.0);
         }
         else if (load)
         {
            real_t ramp = 1.0;
            if (current_time < Elasticity_ctx.tp_ramp)
            {
               ramp = 0.5 * (1.0 - std::cos(M_PI * current_time / Elasticity_ctx.tp_ramp));
            }
            internal_pressure_coeff->SetScalar(-Elasticity_ctx.p_internal * ramp);
         }

         solid_displacement_gf.GetTrueDofs(solid_displacement);
         solid_vel_gf.GetTrueDofs(solid_vel);

         elasticity.Step(solid_displacement, solid_vel, current_time, dt, false);

         solid_displacement_gf.SetFromTrueDofs(solid_displacement);
         solid_vel_gf.SetFromTrueDofs(solid_vel);

         // current_time += dt; --> updated inside elasticity.Step() for HHT solver

         if ((step + 1) % save_freq == 0 || last_step)
         {
            paraview_step++;
            pvdc_solid.SetCycle(paraview_step);
            pvdc_solid.SetTime(current_time);
            pvdc_solid.Save();
         }

         if (Mpi::Root())
         {
            out << std::left
                << std::setw(8)  << step
                << std::setw(16) << std::scientific << std::setprecision(8) << current_time
                << std::setw(16) << std::scientific << std::setprecision(8) << dt
                << std::setw(16) << 0.0
                << std::setw(16) << 0.0
                << std::setw(16) << 0.0
                << std::setw(16) << 0.0  // disp_r_mid (no probe for CSM-only)
                << std::setw(16) << 1
                << "\n";
         }
      }

      if (Mpi::Root())
      {
         mfem::out << "\nCSM-only test done." << std::endl;
      }
      return 0;
   }

   if (cfd_rigid)
   {
      if (Mpi::Root())
      {
         mfem::out << "\nRunning CFD-only test (rigid wall)..." << std::endl;
      }

      for (int step = 0; !last_step; ++step)
      {
         if (current_time + dt >= t_final - dt / 2)
         {
            last_step = true;
         }

         navier.Step(current_time, dt, step, true);

         real_t cfl = 0.0;
         if (time_adaptivity)
         {
            cfl = navier.ComputeCFL(*un_next_gf, dt);
            real_t error_est = cfl / (cfl_target + cfl_tol);
            if (error_est < 1.0)
            {
               real_t eta = pow(1.0 / (2.0 * error_est), 1.0 / 4.0);
               dt = std::min(dt_max, dt * std::min(1.4, std::max(0.1, eta)));
            }
         }

         current_time += dt;
         navier.UpdateTimestepHistory(dt);

         if ((step + 1) % save_freq == 0 || last_step)
         {
            paraview_step++;
            pvdc_fluid.SetCycle(paraview_step);
            pvdc_fluid.SetTime(current_time);
            pvdc_fluid.Save();
         }

         if (Mpi::Root())
         {
            // Pressure probe at inlet center (0, 0, 0) and outlet center (0, 0, L)
            out << std::left
                << std::setw(8)  << step
                << std::setw(16) << std::scientific << std::setprecision(8) << current_time
                << std::setw(16) << std::scientific << std::setprecision(8) << dt
                << std::setw(16) << pressure_pulse_fun(Vector(3), current_time)
                << std::setw(16) << 0.0
                << std::setw(16) << std::scientific << std::setprecision(8) << cfl
                << std::setw(16) << 0.0
                << std::setw(16) << 1
                << "\n";
         }
      }

      if (Mpi::Root())
      {
         mfem::out << "\nCFD-only test done." << std::endl;
      }
      return 0;
   }

   ////////////////////////////////////////////////////////////////////////////////////////////////
   /// 8.2 ----- Coupled FSI
   ////////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
   {
      mfem::out << "\nStarting coupled FSI simulation..." << std::endl;
   }

   const int sdim = fluid_pmesh.Dimension(); // = 3

   // Fluid traction: σ_f·n = (-p I + rho * nu * (∇u + ∇u^T))·n
   auto FluidBoundaryNormalStress = [&](ElementTransformation &Tr,
                                        const IntegrationPoint &ip,
                                        Vector &qoi_loc)
   {
      real_t nu_loc, p_loc;
      Vector nor(sdim);
      DenseMatrix dudx, A(sdim, sdim);
      CalcOrtho(Tr.Jacobian(), nor);
      nor /= nor.Norml2();
      un_next_gf->GetVectorGradient(Tr, dudx);
      p_loc  = p_gf->GetValue(Tr, ip);
      nu_loc = nu_gf->GetValue(Tr, ip);
      for (int i = 0; i < sdim; i++)
         for (int j = 0; j < sdim; j++)
         {
            A(i, j) = -p_loc * (i == j) + NS_ctx.density * nu_loc * (dudx(i, j) + dudx(j,
                                                                                       i));
         }
      for (int i = 0; i < sdim; i++)
      {
         real_t s = 0.0;
         for (int j = 0; j < sdim; j++) { s += A(i, j) * nor(j); }
         qoi_loc(i) = s;
      }
   };

   // Solid traction: σ_s·n = (λ (∇·u) I + μ (∇u + ∇u^T))·n
   auto SolidBoundaryNormalStress = [&](ElementTransformation &Tr,
                                        const IntegrationPoint &ip,
                                        Vector &qoi_loc)
   {
      Vector nor(sdim);
      DenseMatrix dudx, A(sdim, sdim);
      CalcOrtho(Tr.Jacobian(), nor);
      nor /= nor.Norml2();
      solid_displacement_gf.GetVectorGradient(Tr, dudx);
      real_t div_u = dudx.Trace() * Elasticity_ctx.lambda;
      for (int i = 0; i < sdim; i++)
         for (int j = 0; j < sdim; j++)
         {
            A(i, j) = div_u * (i == j) + Elasticity_ctx.mu * (dudx(i, j) + dudx(j, i));
         }
      for (int i = 0; i < sdim; i++)
      {
         real_t s = 0.0;
         for (int j = 0; j < sdim; j++) { s += A(i, j) * nor(j); }
         qoi_loc(i) = s;
      }
   };

   // Convergence coefficient (uses the ptr set in section 6)
   VectorGridFunctionCoefficient solid_vel_onFluid_prev_coeff(
      solid_vel_onFluid_prev_gf_ptr);

   real_t omega_max  = 1.0;
   real_t omega_min  = 1e-6;
   real_t omega_init = DD_ctx.omega;
   int init_steps    = 3;

   AitkenRelaxation aitken_fluid_velocity(fluid_pmesh.GetComm(), omega_init,
                                          omega_min, omega_max, init_steps);
   AitkenRelaxation aitken_solid_sigma(solid_pmesh.GetComm(), omega_init,
                                       omega_min, omega_max, init_steps);
   aitken_fluid_velocity.SetSafetyFactor(DD_ctx.safety_factor);
   aitken_solid_sigma.SetSafetyFactor(DD_ctx.safety_factor);
   aitken_fluid_velocity.SetSize(u_fluid);
   aitken_solid_sigma.SetSize(solid_sigmaN_onFluid);

   Array<int> subiter;
   Array<real_t> omega_history_fluid;
   Array<real_t> p_inlet_history, p_outlet_history, disp_r_history;

   real_t fluid_mesh_displacement_norm = 0.0;

   // Fluid traction on the fluid interface at t_0 (as computed by the forward transfer): Robin datum of the first
   // sub-iteration with --robin-traction 0.
   ParGridFunction fluid_sigmaN_onFluid_gf(u_gf->ParFESpace());
   fluid_sigmaN_onFluid_gf = 0.0;
   if (DD_ctx.coupling != CouplingType::Dirichlet_Neumann &&
       DD_ctx.robin_traction == 0)
   {
      finder_fluid_to_solid_ptr->InterpolateQoIForward(FluidBoundaryNormalStress,
                                                       fluid_sigmaN_onSolid_gf, true);
      fluid_sigmaN_onFluid_gf = finder_fluid_to_solid_ptr->GetForwardSourceQoI();
   }

   for (int step = current_step; !last_step; ++step)
   {
      if (current_time + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      real_t previous_dt = dt;
      real_t cfl = 0.0;

      bool converged = false;
      int iter = 0;
      const int max_iter = 500;
      const int min_iter = 3;
      const real_t rel_tol = 1e-4;
      const real_t abs_tol = 1e-7;

      // Store state at t_n
      solid_displacement_gf.GetTrueDofs(solid_displacement_tn);
      solid_vel_gf.GetTrueDofs(solid_vel_tn);
      u_gf->GetTrueDofs(u_fluid_tn);
      p_gf->GetTrueDofs(p_fluid_tn);
      fluid_sigmaN_onSolid_gf.GetTrueDofs(fluid_sigmaN_onSolid);
      u_fluid              = u_fluid_tn;
      p_fluid              = p_fluid_tn;
      solid_displacement   = solid_displacement_tn;
      solid_vel            = solid_vel_tn;
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
         solid_vel_onFluid_prev     = solid_vel_onFluid;
         solid_sigmaN_onFluid_prev  = solid_sigmaN_onFluid;
         fluid_sigmaN_onSolid_prev  = fluid_sigmaN_onSolid;

         solid_vel_onFluid_prev_gf_ptr->SetFromTrueDofs(solid_vel_onFluid_prev);
         solid_sigmaN_onFluid_prev_gf_ptr->SetFromTrueDofs(solid_sigmaN_onFluid_prev);
         fluid_sigmaN_onSolid_prev_gf_ptr->SetFromTrueDofs(fluid_sigmaN_onSolid_prev);

         // --- Step 1: Interpolate solid velocity/traction SOLID --> FLUID ---
         if (Mpi::Root() && verbose)
         {
            mfem::out << "  Interpolating SOLID -> FLUID...";
         }

         finder_fluid_to_solid_ptr->InterpolateBackward(solid_vel_gf,
                                                        solid_vel_onFluid_gf);

         if (DD_ctx.coupling != CouplingType::Dirichlet_Neumann)
         {
            if (DD_ctx.robin_traction == 0)
            {
               // Consistent Robin-Neumann datum: the traction applied to the solid, i.e. the fluid traction of the
               // previous sub-iterate (at the first sub-iteration, of t_n), as evaluated on the fluid side by the
               // forward transfer.
               solid_sigmaN_onFluid_gf = fluid_sigmaN_onFluid_gf;
               solid_sigmaN_onFluid_gf.Neg(); // NavierSolver's Robin datum is sigma_s.n_s = -sigma_f.n_f (it applies a -1)
            }
            else
            {
               finder_fluid_to_solid_ptr->InterpolateQoIBackward(SolidBoundaryNormalStress,
                                                                 solid_sigmaN_onFluid_gf, true);
            }
         }

         // Relaxation
         // Aitken: the residual is r_k = x~_k - x_{k-1}, with x_{k-1} the *relaxed* iterate used in the previous
         // sub-iteration (the *_prev vectors), see Kuttler & Wall (2008).
         real_t w_sv = 1.0, w_ss =
                          1.0; // relaxation factors actually used in this sub-iteration
         if (iter > 0)
         {
            solid_vel_onFluid_gf.GetTrueDofs(solid_vel_onFluid);
            w_sv = DD_ctx.aitken ?
                   aitken_fluid_velocity.ComputeRelaxation(solid_vel_onFluid,
                                                           solid_vel_onFluid_prev) : DD_ctx.omega;
            solid_vel_onFluid *= w_sv;
            solid_vel_onFluid.Add(1.0 - w_sv, solid_vel_onFluid_prev);
            solid_vel_onFluid_gf.SetFromTrueDofs(solid_vel_onFluid);

            if (DD_ctx.coupling != CouplingType::Dirichlet_Neumann)
            {
               solid_sigmaN_onFluid_gf.GetTrueDofs(solid_sigmaN_onFluid);
               w_ss = DD_ctx.aitken ?
                      aitken_solid_sigma.ComputeRelaxation(solid_sigmaN_onFluid,
                                                           solid_sigmaN_onFluid_prev) : DD_ctx.omega;
               solid_sigmaN_onFluid *= w_ss;
               solid_sigmaN_onFluid.Add(1.0 - w_ss, solid_sigmaN_onFluid_prev);
               solid_sigmaN_onFluid_gf.SetFromTrueDofs(solid_sigmaN_onFluid);
            }
         }

         if (Mpi::Root() && verbose) { mfem::out << " done.\n"; }

         // --- Step 2: Solve Navier-Stokes on FLUID ---
         if (Mpi::Root() && verbose)
         {
            mfem::out << "  Solving Navier-Stokes...";
         }

         un_next_gf->SetFromTrueDofs(u_fluid_tn);
         p_gf->SetFromTrueDofs(p_fluid_tn);

         navier.Step(current_time, dt, step, true);

         if (Mpi::Root() && verbose) { mfem::out << " done.\n"; }

         // --- Step 3: Interpolate fluid traction FLUID --> SOLID ---
         if (Mpi::Root() && verbose)
         {
            mfem::out << "  Interpolating FLUID -> SOLID (traction)...";
         }

         finder_fluid_to_solid_ptr->InterpolateQoIForward(FluidBoundaryNormalStress,
                                                          fluid_sigmaN_onSolid_gf, true);
         fluid_sigmaN_onSolid_gf.GetTrueDofs(fluid_sigmaN_onSolid);
         fluid_sigmaN_onFluid_gf =
            finder_fluid_to_solid_ptr->GetForwardSourceQoI(); // Robin datum of the next sub-iteration

         if (Mpi::Root() && verbose) { mfem::out << " done.\n"; }

         // --- Step 4: Solve Elasticity on SOLID ---
         if (Mpi::Root() && verbose)
         {
            mfem::out << "  Solving Elasticity...";
         }

         solid_displacement = solid_displacement_tn;
         solid_vel          = solid_vel_tn;
         elasticity.ResetState(acceleration_tn);

         elasticity.Step(solid_displacement, solid_vel, current_time, dt, true);
         solid_displacement_gf.SetFromTrueDofs(solid_displacement);
         solid_vel_gf.SetFromTrueDofs(solid_vel);

         if (Mpi::Root() && verbose) { mfem::out << " done.\n"; }

         // --- Step 5: Compute mesh velocity (without deforming yet) ---
         if (Mpi::Root() && verbose)
         {
            mfem::out << "  Computing mesh displacement...";
         }

         Vector mesh_velocity;
         if (fluid_mesh_deformation->IsNonIncremental())
         {
            *mesh_displacement_gf = 0.0;
            finder_solid_to_mesh_ptr->Interpolate(solid_displacement_gf,
                                                  *mesh_displacement_gf);
            fluid_mesh_deformation->Solve(*mesh_displacement_gf, false);
            mesh_displacement_gf->GetTrueDofs(mesh_velocity);
            mesh_velocity -= mesh_displacement_tn;
            fluid_mesh_displacement_norm = mesh_velocity.Norml2();
         }
         else
         {
            Vector rel_displacement;
            solid_displacement_gf.GetTrueDofs(rel_displacement);
            rel_displacement -= solid_displacement_tn;
            solid_relative_displacement_gf->SetFromTrueDofs(rel_displacement);
            *mesh_displacement_gf = 0.0;
            finder_solid_to_mesh_ptr->Interpolate(*solid_relative_displacement_gf,
                                                  *mesh_displacement_gf);
            fluid_mesh_deformation->Solve(*mesh_displacement_gf, false);
            mesh_displacement_gf->GetTrueDofs(mesh_velocity);
            fluid_mesh_displacement_norm = mesh_velocity.Norml2();
         }

         mesh_velocity *= (1.0 / dt);
         mesh_velocity_gf->SetFromTrueDofs(mesh_velocity);

         if (Mpi::Root() && verbose) { mfem::out << " done.\n"; }

         // --- Step 6: Check convergence ---
         // The relaxed increment is x_k - x_{k-1} = w r_k: divide by w to test the residual r_k itself (otherwise a
         // small Aitken factor gives a false convergence). Norms: L2 on the same interface elements.
         Vector zero_vec_conv(sdim);
         zero_vec_conv = 0.0;
         VectorConstantCoefficient zero_vcoeff_conv(zero_vec_conv);
         real_t global_norm_fluid = solid_vel_onFluid_gf.ComputeL2Error(
                                       zero_vcoeff_conv, irs_vel, &fluid_interfaces_element_idx);
         real_t global_abs_err_fluid = solid_vel_onFluid_gf.ComputeL2Error(
                                          solid_vel_onFluid_prev_coeff, irs_vel, &fluid_interfaces_element_idx) / w_sv;
         real_t global_rel_err_fluid = global_abs_err_fluid /
                                       (global_norm_fluid + std::numeric_limits<real_t>::epsilon());
         if (DD_ctx.coupling != CouplingType::Dirichlet_Neumann)
         {
            // Robin data: the traction residual must also be small
            VectorGridFunctionCoefficient solid_sigmaN_onFluid_prev_coeff(
               solid_sigmaN_onFluid_prev_gf_ptr);
            const real_t norm_ss = solid_sigmaN_onFluid_gf.ComputeL2Error(
                                      zero_vcoeff_conv, irs_vel, &fluid_interfaces_element_idx);
            const real_t abs_ss = solid_sigmaN_onFluid_gf.ComputeL2Error(
                                     solid_sigmaN_onFluid_prev_coeff, irs_vel, &fluid_interfaces_element_idx) / w_ss;
            global_abs_err_fluid = std::max(global_abs_err_fluid, abs_ss);
            global_rel_err_fluid = std::max(global_rel_err_fluid,
                                            abs_ss / (norm_ss + std::numeric_limits<real_t>::epsilon()));
         }

         bool conv_rel = (global_rel_err_fluid < rel_tol) && (iter >= min_iter - 1);
         bool conv_abs = (global_abs_err_fluid < abs_tol) && (iter >= min_iter - 1);
         converged = conv_abs || conv_rel;

         iter++;
         omega_history_fluid.Append(w_sv);

         if (iter == max_iter && Mpi::Root())
         {
            mfem::out << "\nWARNING: Max iterations reached. "
                      << "abs_err=" << std::scientific << std::setprecision(3) << global_abs_err_fluid
                      << "  rel_err=" << global_rel_err_fluid << std::endl;
         }
      } // convergence loop

      aitken_fluid_velocity.Reset();
      aitken_solid_sigma.Reset();

      // Time adaptivity
      if (time_adaptivity)
      {
         cfl = navier.ComputeCFL(*un_next_gf, dt);
         real_t error_est = cfl / (cfl_target + cfl_tol);
         if (error_est >= 1.0)
         {
            dt *= 0.125;
            if (Mpi::Root())
               mfem::out << "\nStep " << step << ": CFL exceeded, retrying with dt = "
                         << std::scientific << std::setprecision(3) << dt << std::endl;

            solid_displacement = solid_displacement_tn;
            solid_vel          = solid_vel_tn;
            elasticity.ResetState(acceleration_tn);
            solid_displacement_gf.SetFromTrueDofs(solid_displacement);
            solid_vel_gf.SetFromTrueDofs(solid_vel);
            u_fluid = u_fluid_tn;
            p_fluid = p_fluid_tn;
            un_next_gf->SetFromTrueDofs(u_fluid);
            p_gf->SetFromTrueDofs(p_fluid);
            step--;
            continue;
         }

         real_t eta = pow(1.0 / (2.0 * error_est), 1.0 / (1.0 + NS_ctx.max_bdf_order));
         dt = std::min(dt_max, dt * std::min(2.0, std::max(0.1, eta)));
      }

      current_time += previous_dt;
      navier.UpdateTimestepHistory(previous_dt);
      elasticity.UpdateTimestepHistory(solid_displacement, solid_vel);
      subiter.Append(iter);

      // Deform the fluid mesh
      if (fluid_mesh_deformation->IsNonIncremental())
      {
         fluid_mesh_deformation->DeformMeshWithDisplacement(*mesh_displacement_gf);
      }
      else
      {
         fluid_mesh_deformation->DeformMeshWithRelativeDisplacement(
            *mesh_displacement_gf);
      }

      navier.Update();
      fluid_mesh_deformation->Update();
      finder_fluid_to_solid_ptr->Update(true, true, mesh_displacement_coeff, nullptr);
      finder_solid_to_mesh_ptr->Update();

      // QoI: pressure probes and radial displacement at midpoint
      real_t p_inlet_val = 0.0, p_outlet_val = 0.0, disp_r_mid = 0.0;
      {
         // Pressure probe at inlet center (x=-L/2, y=0, z=0) and outlet center (x=L/2, y=0, z=0)
         DenseMatrix pts(3, 2);
         pts(0,0) = -Mesh_ctx.L / 2.0; pts(1,0) = 0.0;
         pts(2,0) = 0.0;          // inlet center
         pts(0,1) = Mesh_ctx.L / 2.0; pts(1,1) = 0.0; pts(2,1) = 0.0;   // outlet center
         Array<int> elem_ids;
         Array<IntegrationPoint> ips;
         fluid_pmesh.FindPoints(pts, elem_ids, ips);

         if (elem_ids[0] >= 0) { p_inlet_val  = p_gf->GetValue(elem_ids[0], ips[0]); }
         if (elem_ids[1] >= 0) { p_outlet_val = p_gf->GetValue(elem_ids[1], ips[1]); }

         real_t global_p[2] = {p_inlet_val, p_outlet_val}, reduced_p[2];
         MPI_Allreduce(global_p, reduced_p, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         p_inlet_val  = reduced_p[0];
         p_outlet_val = reduced_p[1];

         // Radial displacement probe at (0, 0, R_inner)
         DenseMatrix pts_s(3, 1);
         pts_s(0,0) = 0.0; pts_s(1,0) = 0.0; pts_s(2,0) = Mesh_ctx.R_inner;
         solid_pmesh.FindPoints(pts_s, elem_ids, ips);
         if (elem_ids[0] >= 0)
         {
            Vector val;
            solid_displacement_gf.GetVectorValue(elem_ids[0], ips[0], val);
            disp_r_mid = val(0); // x-displacement ~ radial for a point on +x axis
         }
         real_t global_dr;
         MPI_Allreduce(&disp_r_mid, &global_dr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         disp_r_mid = global_dr;
      }
      p_inlet_history.Append(p_inlet_val);
      p_outlet_history.Append(p_outlet_val);
      disp_r_history.Append(disp_r_mid);

      // Output
      if ((step + 1) % save_freq == 0 || last_step)
      {
         paraview_step++;
         pvdc_fluid.SetCycle(paraview_step);
         pvdc_fluid.SetTime(current_time);
         pvdc_fluid.Save();
         pvdc_solid.SetCycle(paraview_step);
         pvdc_solid.SetTime(current_time);
         pvdc_solid.Save();
      }

      if (Mpi::Root())
      {
         out << std::left
             << std::setw(8)  << step
             << std::setw(16) << std::scientific << std::setprecision(8) << current_time
             << std::setw(16) << std::scientific << std::setprecision(8) << previous_dt
             << std::setw(16) << std::scientific << std::setprecision(8) << p_inlet_val
             << std::setw(16) << std::scientific << std::setprecision(8) << p_outlet_val
             << std::setw(16) << std::scientific << std::setprecision(8) << cfl
             << std::setw(16) << std::scientific << std::setprecision(8) << disp_r_mid
             << std::setw(16) << std::scientific << std::setprecision(8) << iter
             << "\n";
      }

   } // FSI time-integration loop

   navier.PrintTimingData();

   saveArray(subiter,           outfolder, "subiter.txt");
   saveArray(omega_history_fluid, outfolder, "omega.txt");
   saveArray(p_inlet_history,   outfolder, "p_inlet.txt");
   saveArray(p_outlet_history,  outfolder, "p_outlet.txt");
   saveArray(disp_r_history,    outfolder, "disp_r_mid.txt");

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 9. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (mesh_displacement_coeff) { delete mesh_displacement_coeff; }
   if (mesh_displacement_gf) { delete mesh_displacement_gf; }
   if (mesh_velocity_gf) { delete mesh_velocity_gf; }
   if (fluid_mesh_deformation) { delete fluid_mesh_deformation; }
   if (solid_relative_displacement_gf) { delete solid_relative_displacement_gf; }
   if (finder_fluid_to_solid_ptr) { delete finder_fluid_to_solid_ptr; }
   if (finder_solid_to_mesh_ptr) { delete finder_solid_to_mesh_ptr; }
   if (solid_vel_onFluid_prev_gf_ptr) { delete solid_vel_onFluid_prev_gf_ptr; }
   if (fluid_sigmaN_onSolid_prev_gf_ptr) { delete fluid_sigmaN_onSolid_prev_gf_ptr; }
   if (solid_sigmaN_onFluid_prev_gf_ptr) { delete solid_sigmaN_onFluid_prev_gf_ptr; }

   return 0;
}

// =====================================================================================
// Helper functions
// =====================================================================================

void noSlip_fun(const Vector &x, real_t /*t*/, Vector &u)
{
   u = 0.0;
}

real_t zero_pressure(const Vector & /*x*/, real_t /*t*/)
{
   return 0.0;
}

real_t pressure_pulse_fun(const Vector & /*x*/, real_t t)
{
   // Benchmark: constant traction for 3e-3 s, then zero

   if (t <= 0.0 || t >= Pulse_ctx.end_time)
   {
      return 0.0;
   }

   // Optional smooth ramp-up of the amplitude
   real_t ramp = 1.0;
   if (Pulse_ctx.ramp_time > 0.0 && t < Pulse_ctx.ramp_time)
   {
      ramp = 0.5 * (1.0 - std::cos(M_PI * t / Pulse_ctx.ramp_time));
   }

   return Pulse_ctx.amplitude * ramp;
}


template <typename T>
void saveArray(const Array<T> &data, const std::string &outfolder,
               const std::string &name)
{
   std::string outputFolder = outfolder + "/Convergence/";
   if (!fs::is_directory(outputFolder) || !fs::exists(outputFolder))
   {
      fs::create_directories(outputFolder);
   }

   std::ostringstream filename;
   filename << outputFolder << name;
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
