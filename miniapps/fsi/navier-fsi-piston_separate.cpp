// =====================================================================================
//          2D FSI Example: Fluid + Elastic Solid
// -------------------------------------------------------------------------------------
// Sample runs:
//
// The type of FSI coupling is controlled by the --coupling flag (0: Dirichlet-Neumann, 1: Robin-Neumann, 2: RobinAMP-Neumann)
//
// Longitudinal problem (elastic piston manufactured solution)
//
// Assumptions:
//   - Geometry: fluid domain H=1, solid domain H_bar=1 (both unit height)
//   - Benchmark parameters: lambda_bar = mu_bar = rho_bar = delta (Lame = density = delta)
//   - Wave speed: cp = sqrt((lambda+2mu)/rho_s) = sqrt(3) = 1.732 m/s (same for all delta)
//   - Manufactured solution: F_bar(tau) = alpha*cos(omega*tau)
//   - Interface amplitude: a = 2*alpha*sin(omega*H_bar/cp) = 0.183 m (same for all delta)
//   - omega=2 chosen so that omega*H_bar/cp = 1.155 rad < pi/2, giving a monotonic
//     solid velocity profile from 0 at the fixed wall to max at the interface
//   - alpha=0.1 (positive) gives upward interface motion (a > 0)
//   - Fluid: rho_f=1, viscosity=1e-2 (same for all delta)
//   - Coupling: Dirichlet-Neumann (fluid=Dirichlet sender, solid=Neumann receiver)
//   - Initial conditions: solid velocity = 2*alpha*omega*sin(omega*(y_bar+H_bar)/cp),
//                         fluid velocity = 2*alpha*omega*sin(omega*H_bar/cp) (uniform),
//                         solid displacement = 0, fluid pressure = 0
//   - t_final = pi ~ one half period of the oscillation
//
// DIRICHLET-NEUMANN COUPLING (coupling = 0):
//
// delta = 1e3 (heavy solid, rho_bar/rho_f = 1e3)
// mpirun -np 4 ./navier-fsi-two-domains   --coupling 0   -p 0   -rs 1   -fo 4   -eo 2   --viscosity 1e-2   -rho-f 1   -rho-s 1e3   --young-modulus 2500   --poisson-ratio 0.25   --relaxation 1   -dt 1e-4  -ta  --dt-max 1e-1  -tf 3.14159   --periodic   -pa 0.1   -pw 2   -of /home/shared/Output/FSI-LANL/Piston-FSI/Debug-DN/delta1e3/
//
// delta = 1e2
// mpirun -np 4 ./navier-fsi-two-domains   --coupling 0   -p 0   -rs 1   -fo 4   -eo 2   --viscosity 1e-2   -rho-f 1   -rho-s 1e2   --young-modulus 250   --poisson-ratio 0.25   --relaxation 0.5   -dt 1e-4  -ta  --dt-max 1e-1  -tf 3.14159   --periodic   -pa 0.1   -pw 2   -of /home/shared/Output/FSI-LANL/Piston-FSI/Debug-DN/delta1e2/
//
// delta = 1 (moderate solid, rho_bar/rho_f = 1)
// mpirun -np 4 ./navier-fsi-two-domains   --coupling 0   -p 0   -rs 1   -fo 4   -eo 2   --viscosity 1e-2   -rho-f 1   -rho-s 1   --young-modulus 2.5   --poisson-ratio 0.25   --relaxation 0.01   -dt 1e-4  -ta  --dt-max 1e-1  -tf 3.14159   --periodic   -pa 0.1   -pw 2   -of /home/shared/Output/FSI-LANL/Piston-FSI/Debug-DN/delta1/
//
// The following mesh is used:
//
//               7
//         -------------
//        |             |
//    5   |    Fluid    | 6
//        |   Domain    |
//        |     (2)     |
//        |             |
//         --4---3---4--  Interface
//        |             |
//        |    Solid    |
//     2  |   Domain    | 2
//        |     (1)     |
//        |             |
//         -------------
//               1
//
// The mesh is linear but the order of the Nodes gridfunction can be selected with the flag -mo.
// If not provided, the mesh will have same order as the elasticity solver.
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
ParMesh CreateSubmeshMesh(Mesh *parent_mesh, int domain_attr, std::vector<Vector> translations, int mesh_order = 1);
template <typename T>
void saveArray(const Array<T> &data, const std::string &outfolder, const std::string &name);

real_t pressure_piston_top(const Vector &x, double time);
real_t solid_disp_analytic(const Vector &x, double time);
real_t solid_vel_analytic(const Vector &x, double time);
void   solid_disp_vec_analytic(const Vector &x, double time, Vector &u);
void   solid_vel_vec_analytic(const Vector &x, double time, Vector &v);
void   solid_acc_vec_analytic(const Vector &x, double time, Vector &a);
void   fluid_vel_vec_analytic(const Vector &x, double time, Vector &v);
real_t fluid_pressure_analytic(const Vector &x, double time);
real_t fluid_interface_pressure_analytic(const Vector &x, double time);

// Analytic interface traction quantities (for separate-problem debugging)
// solid_sigma_n_vec_analytic : σ_s · n_s at the FS interface (n_s = (0,+1))
// fluid_sigma_n_vec_analytic : σ_f · n_f at the FS interface (n_f = (0,-1))
void   solid_sigma_n_vec_analytic(const Vector &x, double time, Vector &v);
void   fluid_sigma_n_vec_analytic(const Vector &x, double time, Vector &v);

void noSlip_fun(const Vector &x, double time, Vector &u);
real_t zero_pressure(const Vector &x, double t);

void ComputeDragLift(real_t &drag, real_t &lift, ParLinearForm &lift_drag_form);

struct s_NavierContext // Navier Stokes params
{
   int u_order = 2;                   // polynomial order of the velocity space
   int max_bdf_order = 3;             // maximum BDF order for time integration
   real_t density = 1;                // Density of the fluid
   real_t dynamic_viscosity = 1e-3;   // Dynamic viscosity of the fluid
   real_t kinematic_viscosity = 1e-6; // Kinematic viscosity of the fluid
   real_t Umax = 0.2;                 // Maximum velocity at the inlet
   real_t preloadT = 2.0;             // Pressure preload
} NS_ctx;

struct s_ElasticityContext // elasticity params
{
   int order = 2;        // Polynomial order of the elasticity space
   real_t density = 1e3; // Density
   real_t nu = 0.4;      // Poisson's ratio
   real_t E = 2.0e6;     // Young's modulus
   real_t mu;            // G, Shear modulus, Lame's second parameter
   real_t lambda;        // Lame's first parameter
   real_t traction = 2.5e5;
} Elasticity_ctx;

struct s_MeshContext // mesh
{
   real_t L = 1.0; // length of the mesh
   real_t H = 1.0; // height of the mesh
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   int order = -1;  // polynomial order of the mesh nodes
   real_t E = 1e6;  // Young's modulus for the mesh deformation
   real_t nu = 0.4; // Poisson's ratio for the mesh deformation
} Mesh_ctx;

struct s_DomainDecompositionContext // domain decomposition params
{
   bool aitken = false;         // Enable Aitken relaxation
   real_t safety_factor = 1.15; // Safety factor for the Aitken relaxation method
   real_t omega_solid_velocity = 1.0;
   real_t omega_solid_sigma = 1.0;
   real_t omega_fluid_velocity = 1.0;
   real_t alphaP = -1.0; 
   real_t alphaS = -1.0;
   real_t alphaSolid = -1.0; 
   bool relative_convergence = true; // Use absolute or relative convergence criterion
} DD_ctx;

struct s_ElasticPistonContext // elastic piston verification (section 7.1)
{
   real_t delta = 1e3;
   real_t alpha = 0.01;       // Amplitude of F̄(τ) = α·cos(ωτ)
   real_t omega = 2.0 * M_PI; // Frequency parameter in F̄(τ)
   real_t a = 0.0;            // Interface oscillation amplitude a = 2α·sin(ω·H̄/c̄p) [computed]
   real_t cp_bar = 0.0;       // Solid longitudinal wave speed c̄p = sqrt((λ̄+2μ̄)/ρ̄s) [computed]
   real_t H_fluid = 0.0;      // Initial fluid domain height in y [from mesh bounding box]
   real_t H_bar_solid = 0.0;  // Solid reference domain height in y [from mesh bounding box]
} ElasticPiston_ctx;

enum class CouplingType : int
{
   Dirichlet_Neumann = 0,
   Robin_Neumann = 1,
   RobinAMP_Neumann = 2,
   Neumann_Dirichlet = 3,
   Robin_Robin = 4
};

const char *outfolder = "./Output/FSI";
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
   const char *mesh_file = "../../data/multidomain2d-quad.e";
   bool periodic = false; // periodic mesh

   // Timestepping parameters
   real_t current_time = 0.0;
   // Max-in-time L2 errors against the analytic elastic piston solution:
   // solid displacement, solid velocity, fluid velocity, fluid pressure.
   real_t piston_max_err[4] = {0.0, 0.0, 0.0, 0.0};
   real_t t_final = 0.5;
   real_t dt = 1e-5;
   real_t dt_max = 1e-1;
   real_t cfl_target = 0.5;
   real_t cfl_tol = 1e-4;
   bool time_adaptivity = false; // Enable time adaptivity

   // Coupling parameters
   real_t relaxation = 1.0;
   CouplingType coupling = CouplingType::Dirichlet_Neumann; // Default coupling type

   // Deformation parameters
   MeshDeformationType mesh_deformation_type = MeshDeformationType::Laplacian; // Default is Laplacian

   // Output parameters
   int save_freq = 1;
   int save_freq_preload = 1;

   int current_step = 0;
   bool last_step = false;

   bool verbose = false; // verbose output during time integration

   OptionsParser args(argc, argv);
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
   // Elasticity parameters
   args.AddOption(&ElasticPiston_ctx.delta, "-delta", "--delta",
                  "Density and Lame parameter ratio delta = rho_bar/rho_f = lambda_bar/mu_bar for elastic piston verification (default: 1e3).");
   // Navier Stokes parameters
   args.AddOption(&NS_ctx.dynamic_viscosity, "-visc", "--viscosity",
                  "Dynamic viscosity of the fluid (default: 1e-6).");
   // Time integration parameters
   args.AddOption(&t_final, "-tf", "--t_final", "t_final");
   args.AddOption(&dt, "-dt", "--dt", "dt.");
   args.AddOption(&dt_max, "-dtmax", "--dt-max", "Maximum time step size.");
   // FSI coupling parameters
   args.AddOption((int *)&coupling, "-c", "--coupling",
                  "Coupling type: 0 - Dirichlet_Neumann, 1 - Robin_Neumann, 2 - RobinAMP_Neumann, 3 - Neumann_Dirichlet (default: 0).");
   args.AddOption(&relaxation, "-rel", "--relaxation",
                  "Relaxation factor for the fluid and elastic problem (default: 1.0).");
   args.AddOption(&DD_ctx.aitken, "-aa", "--aitken", "-no-aa", "--no-aitken",
                  "Enable or disable Aitken relaxation.");
   args.AddOption(&DD_ctx.safety_factor, "-sfa", "--safety-factor",
                  "Safety factor for the Aitken relaxation method (default: 1.15). ");
   args.AddOption(&DD_ctx.relative_convergence, "-rc", "--relative-convergence", "-ac", "--absolute-convergence",
                  "Use relative or absolute convergence criterion (default: false, i.e. absolute convergence).");
   args.AddOption(&DD_ctx.alphaP, "-ap", "--alpha-p",
                  "Alpha_p parameter for Robin or RobinAMP coupling (default: -1, i.e. use impedance-based value).");
   args.AddOption(&DD_ctx.alphaS, "-as", "--alpha-s",
                  "Alpha_s parameter for RobinAMP coupling (default: -1, i.e. use impedance-based value).");
   args.AddOption(&DD_ctx.alphaSolid, "-asolid", "--alpha-solid",
                  "Alpha_solid parameter for Robin-Robin coupling (default: -1, i.e. use impedance-based value).");
   // Mesh parameters
   args.AddOption(&periodic, "-pm", "--periodic", "-no-pm", "--no-periodic",
                  "Periodic mesh (default: false).");
   args.AddOption(&Mesh_ctx.serial_ref_levels, "-rs", "--rs", "serial_ref_levels");
   args.AddOption(&Mesh_ctx.parallel_ref_levels, "-rp", "--rp", "parallel_ref_levels");
   args.AddOption((int *)&mesh_deformation_type, "-mdt", "--mesh-deformation-type",
                  "Type of mesh deformation to use (0 - Laplacian, 1 - IncrementalLaplacian, 2 - Elasticity). "
                  "Default is 0 (Laplacian).");
   args.AddOption(&Mesh_ctx.order, "-mo", "--mesh-order",
                  "Polynomial order of the mesh nodes (default: 2).");
   // Elastic piston verification (section 7.1)
   args.AddOption(&ElasticPiston_ctx.alpha, "-pa", "--piston-alpha",
                  "Alpha parameter in F̄(τ)=alpha*cos(omega*tau) for elastic-piston verification.");
   args.AddOption(&ElasticPiston_ctx.omega, "-pw", "--piston-omega",
                  "Omega parameter in F̄(τ)=alpha*cos(omega*tau) for elastic-piston verification.");
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

   // Compute Lame's parameters
   Elasticity_ctx.mu = ElasticPiston_ctx.delta;                                                 // G, Shear modulus, Lame's second parameter
   Elasticity_ctx.lambda = ElasticPiston_ctx.delta; // Lame's first parameter
   Elasticity_ctx.E = Elasticity_ctx.lambda + 2 * Elasticity_ctx.mu; // Young's modulus
   Elasticity_ctx.nu = Elasticity_ctx.lambda / (2 * (Elasticity_ctx.lambda + Elasticity_ctx.mu)); // Poisson's ratio

   Elasticity_ctx.density = ElasticPiston_ctx.delta; // Density


   // Compute the kinematic viscosity
   NS_ctx.kinematic_viscosity = NS_ctx.dynamic_viscosity / NS_ctx.density;

   // Set the relaxation factors for the fluid and solid problems
   // If Aitken relaxation is enabled, these factors will be used as the initial omega values at each timestep
   DD_ctx.omega_solid_velocity = relaxation; // Relaxation factor for the fluid problem
   DD_ctx.omega_solid_sigma = relaxation; // Relaxation factor for the solid problem


   // Based on the coupling type, set some booleans for interpolation and transfer 
   bool need_solid_velocity_on_fluid = (coupling != CouplingType::Neumann_Dirichlet);
   bool need_solid_traction_on_fluid = (coupling != CouplingType::Dirichlet_Neumann);
   bool need_fluid_velocity_on_solid = (coupling == CouplingType::Neumann_Dirichlet || coupling == CouplingType::Robin_Robin );
   bool need_fluid_traction_on_solid = (coupling != CouplingType::Neumann_Dirichlet);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create serial Mesh, parallel and fluid/solid sub-meshes
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\nLoading mesh and refining in serial...";

   Mesh *parent_mesh = new Mesh(mesh_file);
   const int dim = parent_mesh->Dimension();

   for (int i = 0; i < Mesh_ctx.serial_ref_levels; i++)
   {
      parent_mesh->UniformRefinement();
   }

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
      mfem::out << "Creating sub-meshes...";
   }

   // Create the fluid and solid sub-meshes
   std::vector<Vector> translations_fluid;
   std::vector<Vector> translations_solid;
   if (periodic)
   {
      translations_fluid.push_back(Vector({Mesh_ctx.L, 0.0})); // Periodic in the x-direction
      translations_solid.push_back(Vector({Mesh_ctx.L, 0.0})); // Periodic in the x-direction
   }

   if (Mesh_ctx.order < 0)
   {
      Mesh_ctx.order = Elasticity_ctx.order; // Use the elasticity order as the mesh order if not specified
   }
   auto solid_pmesh = CreateSubmeshMesh(parent_mesh, 1, translations_solid, 1);
   auto fluid_pmesh = CreateSubmeshMesh(parent_mesh, 2, translations_fluid, Mesh_ctx.order);

   Vector fluid_bb_min, fluid_bb_max, solid_bb_min, solid_bb_max;
   fluid_pmesh.GetBoundingBox(fluid_bb_min, fluid_bb_max, std::max(Mesh_ctx.order, 1));
   solid_pmesh.GetBoundingBox(solid_bb_min, solid_bb_max, std::max(Elasticity_ctx.order, 1));
   ElasticPiston_ctx.H_fluid = fluid_bb_max(1) - fluid_bb_min(1);
   ElasticPiston_ctx.H_bar_solid = solid_bb_max(1) - solid_bb_min(1);

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
      mfem::out << "Fluid height H_fluid = " << ElasticPiston_ctx.H_fluid << std::endl;
      mfem::out << "Solid height H_bar_solid = " << ElasticPiston_ctx.H_bar_solid << std::endl;
   }

   //////////////////
   // SET MARKERS
   //////////////////

   // Setup markers for the solid-fluid interface #(3, 4)
   Array<int> fluid_fs_interface_attr(fluid_pmesh.bdr_attributes.Max());
   fluid_fs_interface_attr = 0;
   fluid_fs_interface_attr[2] = 1;
   fluid_fs_interface_attr[3] = 1;

   Array<int> solid_fs_interface_attr(solid_pmesh.bdr_attributes.Max());
   solid_fs_interface_attr = 0;
   solid_fs_interface_attr[2] = 1;
   solid_fs_interface_attr[3] = 1;

   // Interface attributes for the solid-fluid interface in parent mesh, used in  BidirectionalInterfaceTransfer
   // @NOTE: if we don't use marker from the parent mesh there might be compatibility issues with the markers depending on the direction of the transfer
   Array<int> parent_interface_attr(parent_mesh->bdr_attributes.Max());
   parent_interface_attr = 0;
   parent_interface_attr[2] = 1;
   parent_interface_attr[3] = 1;
   delete parent_mesh; // we don't need the parent mesh anymore

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

   ParGridFunction *u_gf = navier.GetCurrentVelocity();
   ParGridFunction *un_next_gf = navier.GetProvisionalVelocity();
   ParGridFunction *p_gf = navier.GetCurrentPressure();
   ParGridFunction *nu_gf = navier.GetVariableViscosity();

   Vector u_fluid, p_fluid, solid_sigmaN_onFluid, solid_vel_onFluid;
   Vector u_fluid_tn, p_fluid_tn;
   u_gf->GetTrueDofs(u_fluid);
   p_gf->GetTrueDofs(p_fluid);
   u_gf->GetTrueDofs(u_fluid_tn);
   p_gf->GetTrueDofs(p_fluid_tn);

   // ParGridFunction sigmaN_gf(u_gf->ParFESpace()); sigmaN_gf = 0.0;
   ParGridFunction solid_vel_onFluid_gf(u_gf->ParFESpace());
   solid_vel_onFluid_gf = 0.0;
   solid_vel_onFluid_gf.GetTrueDofs(solid_vel_onFluid);
   ParGridFunction solid_sigmaN_onFluid_gf(u_gf->ParFESpace());
   solid_sigmaN_onFluid_gf = 0.0;
   solid_sigmaN_onFluid_gf.GetTrueDofs(solid_sigmaN_onFluid);

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
   Vector solid_vel, solid_vel_tn, fluid_sigmaN_onSolid, fluid_vel_onSolid;

   auto elasticity_fes = elasticity.GetFESpace();

   ParGridFunction solid_displacement_gf(elasticity_fes);
   solid_displacement_gf = 0.0;
   solid_displacement_gf.GetTrueDofs(solid_displacement);
   solid_displacement_gf.GetTrueDofs(solid_displacement_tn);
   ParGridFunction solid_displacement_relaxed_gf(elasticity_fes);
   solid_displacement_relaxed_gf = 0.0;

   ParGridFunction solid_vel_gf(elasticity_fes);
   solid_vel_gf = 0.0;
   solid_vel_gf.GetTrueDofs(solid_vel);
   solid_vel_gf.GetTrueDofs(solid_vel_tn);

   ParGridFunction fluid_sigmaN_onSolid_gf(elasticity_fes);
   fluid_sigmaN_onSolid_gf = 0.0;
   fluid_sigmaN_onSolid_gf.GetTrueDofs(fluid_sigmaN_onSolid);
   VectorGridFunctionCoefficient fluid_sigmaN_onSolid_coeff(&fluid_sigmaN_onSolid_gf);

   ParGridFunction fluid_vel_onSolid_gf(elasticity_fes);
   fluid_vel_onSolid_gf = 0.0;
   fluid_vel_onSolid_gf.GetTrueDofs(fluid_vel_onSolid);
   VectorGridFunctionCoefficient fluid_vel_onSolid_coeff(&fluid_vel_onSolid_gf); 

   int order_quad_disp = std::max(2, 2 * Elasticity_ctx.order + 2);
   const IntegrationRule *irs_disp[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs_disp[i] = &(IntRules.Get(i, order_quad_disp));
   }

   if (Mpi::Root())
      mfem::out << "done." << std::endl;

   /////////////////////////
   // MESH DEFORMATION
   /////////////////////////

   if (Mpi::Root())
      mfem::out << "\nCreating mesh deformation solver...";

   auto vfec = H1_FECollection(2, fluid_pmesh.Dimension());
   auto vfes = ParFiniteElementSpace(&fluid_pmesh, &vfec, fluid_pmesh.Dimension());

   MeshDeformation *fluid_mesh_deformation = nullptr;

   switch (mesh_deformation_type)
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
      mfem::out << "\nUsing mesh deformation type: " << mesh_deformation_names[static_cast<int>(mesh_deformation_type)] << std::endl;

   ParFiniteElementSpace *mesh_nodes_fes = fluid_mesh_deformation->GetFESpace();

   ParGridFunction *mesh_velocity_gf = new ParGridFunction(mesh_nodes_fes); // Mesh velocity field

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
      mfem::out << "done." << std::flush;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Add boundary conditions and initial conditions
   ///////////////////////////////////////////////////////////////////////////////////////////////

   MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes have finished setting up the solvers before printing

   if (Mpi::Root())
      mfem::out << "\nSetting up boundary conditions and initial conditions..." << std::endl;

   ////////////
   // FLUID
   ////////////
   // - Dirichlet on inlet Γ = 5 (parabolic profile)
   // - Pressure on outlet Γ = 6 (p = 0)
   // - No slip on walls Γ = 7 (u = 0)
   // - Dirichlet on interface Γfs = 3, 4 (u = u_solid)

   Array<int> no_slip(fluid_pmesh.bdr_attributes.Max());
   no_slip = 0;
   Array<int> inlet(fluid_pmesh.bdr_attributes.Max());
   inlet = 0;
   Array<int> outlet(fluid_pmesh.bdr_attributes.Max());
   outlet = 0;

   // Longitudinal problem fluid BCs
   if (Mpi::Root())
      mfem::out << "Selected longitudinal problem." << std::endl;

   outlet[6] = 1; // Outlet boundary
   navier.AddPresDirichletBC(pressure_piston_top, outlet);
   if (!periodic)
   {
      no_slip[4] = 1;
      no_slip[5] = 1;
      navier.AddVelDirichletBC(noSlip_fun, no_slip);
   }

   // Add dirichlet at the interface
   VectorGridFunctionCoefficient solid_vel_onFluid_coeff(&solid_vel_onFluid_gf); // Holds the solid velocity projected onto the fluid mesh
   VectorGridFunctionCoefficient solid_sigmaN_onFluid_coeff(&solid_sigmaN_onFluid_gf);
   // navier.AddVelDirichletBC(solid_vel_onFluid_coeff, fluid_fs_interface_attr);
   real_t cp = std::sqrt((Elasticity_ctx.lambda + 2 * Elasticity_ctx.mu) / Elasticity_ctx.density); // compression wave speed
   real_t cs = std::sqrt(Elasticity_ctx.mu / Elasticity_ctx.density);                               // shear wave speed
   real_t zp = Elasticity_ctx.density * cp;
   real_t zs = Elasticity_ctx.density * cs;
   real_t zf = NS_ctx.density * ElasticPiston_ctx.H_fluid / dt;

   if (DD_ctx.alphaP < 0.0) 
      DD_ctx.alphaP = zp;

   if (DD_ctx.alphaS < 0.0)
      DD_ctx.alphaS = zs;

   if (DD_ctx.alphaSolid < 0.0)
      DD_ctx.alphaSolid = zp;

   ElasticPiston_ctx.cp_bar = cp;
   if (ElasticPiston_ctx.cp_bar > 0.0)
   {
      const real_t phase = ElasticPiston_ctx.omega * ElasticPiston_ctx.H_bar_solid / ElasticPiston_ctx.cp_bar;
      ElasticPiston_ctx.a = 2.0 * ElasticPiston_ctx.alpha * std::sin(phase);
   }

   if (Mpi::Root())
   {
      mfem::out << "Elastic piston exact-pressure longitudinal mode:" << std::endl;
      mfem::out << "  - alpha: " << std::scientific << ElasticPiston_ctx.alpha << std::endl;
      mfem::out << "  - omega: " << std::scientific << ElasticPiston_ctx.omega << std::endl;
      mfem::out << "  - H_fluid: " << std::scientific << ElasticPiston_ctx.H_fluid << std::endl;
      mfem::out << "  - H_bar_solid: " << std::scientific << ElasticPiston_ctx.H_bar_solid << std::endl;
      mfem::out << "  - cp_bar: " << std::scientific << ElasticPiston_ctx.cp_bar << std::endl;
      mfem::out << "  - interface amplitude a: " << std::scientific << ElasticPiston_ctx.a << std::endl;
   }

   VectorCoefficient *neumann_coeff = nullptr;
   //Coefficient *neumann_coeff = nullptr;
   switch (coupling)
   {
   case CouplingType::Dirichlet_Neumann:
      navier.AddVelDirichletBC(&solid_vel_onFluid_coeff, fluid_fs_interface_attr, false);
      break;
   case CouplingType::Robin_Neumann:
      navier.AddRobinBC(DD_ctx.alphaP, &solid_vel_onFluid_coeff, &solid_sigmaN_onFluid_coeff, fluid_fs_interface_attr, false);
      break;
   case CouplingType::RobinAMP_Neumann:
      navier.AddAMPRobinBC(DD_ctx.alphaS, DD_ctx.alphaP, &solid_vel_onFluid_coeff, &solid_sigmaN_onFluid_coeff, fluid_fs_interface_attr, false);
      break;
   case CouplingType::Neumann_Dirichlet:
   { // NavierSolver::AddNeumannBC expects the fluid traction σ_f·n_f.
      // We store σ_s·n_s in solid_sigmaN_onFluid_gf, so convert it using
      // σ_f·n_f = -σ_s·n_s before passing it to the fluid solver.
      neumann_coeff = new ScalarVectorProductCoefficient(-1.0, solid_sigmaN_onFluid_coeff);
      bool use_shear_stress = false;
      navier.AddNeumannBC(neumann_coeff, fluid_fs_interface_attr, false, use_shear_stress);
      // neumann_coeff = new FunctionCoefficient(fluid_interface_pressure_analytic);
      // navier.AddPresDirichletBC(neumann_coeff, fluid_fs_interface_attr, false);
   }
   break;
   case CouplingType::Robin_Robin:
      navier.AddRobinBC(DD_ctx.alphaP, &solid_vel_onFluid_coeff, &solid_sigmaN_onFluid_coeff, fluid_fs_interface_attr, false);
      break;
   default:
      break;
   }

   Array<int> fluid_dirichlet_bdr(fluid_pmesh.bdr_attributes.Max());
   fluid_dirichlet_bdr = 1; 
   ConstantCoefficient zero_coeff(0.0);
   navier.AddVelDirichletBC(&zero_coeff, fluid_dirichlet_bdr, 0, false);

   MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes have finished setting up the solvers before printing

   if (Mpi::Root())
   {
      mfem::out << "Impedance values:" << std::endl;
      mfem::out << "  - zp = " << zp << std::endl;
      mfem::out << "  - zs = " << zs << std::endl;
      mfem::out << "  - zf = " << zf << std::endl;
   }

   // Setup solver

   // For the longitudinal problem, we need to initialize the fluid velocity
   // This needs to be done before Setup is called
   // because it defines un based on u_gf

   // -----------------------------------------------------------------------
   // Analytic initial conditions at t=0 for the elastic piston problem.
   // See elastic_piston_analytic.hpp for function definitions.
   //
   // Solid:
   //   u2(y_bar, 0)   = 0                                      (trivial)
   //   u2_t(y_bar, 0) = 2*alpha*omega*sin(omega*(y_bar+H)/cp)  (non-zero)
   //
   // Fluid:
   //   v2(0)  = 2*alpha*omega*sin(omega*H_bar/cp)  (uniform in y)
   //   p(0)   = 0                                  (trivial)
   // -----------------------------------------------------------------------

   // Solid displacement: zero everywhere at t=0, no projection needed
   solid_displacement_gf = 0.0;

   // Solid velocity: spatially varying, must be projected
   VectorFunctionCoefficient solid_vel_ic_coeff(dim, solid_vel_vec_analytic);
   solid_vel_ic_coeff.SetTime(0.0);
   solid_vel_gf.ProjectCoefficient(solid_vel_ic_coeff);

   // Fluid velocity: uniform in y at t=0, must be projected
   VectorFunctionCoefficient fluid_vel_ic_coeff(dim, fluid_vel_vec_analytic);
   fluid_vel_ic_coeff.SetTime(0.0);
   un_next_gf->ProjectCoefficient(fluid_vel_ic_coeff);
   u_gf->ProjectCoefficient(fluid_vel_ic_coeff);

   // Fluid pressure: zero at t=0, no projection needed
   *p_gf = 0.0;

   // Sync true dof vectors with the projected grid functions
   solid_displacement_gf.GetTrueDofs(solid_displacement);
   solid_vel_gf.GetTrueDofs(solid_vel);
   un_next_gf->GetTrueDofs(u_fluid);

   p_gf->GetTrueDofs(p_fluid);

   navier.Setup(dt);

   ////////////
   // SOLID
   ////////////
   // - Neumann (traction) on interface Γfs = 3, 4 (σ·n = σ_f · n)

   Array<int> lateral_bdr(solid_pmesh.bdr_attributes.Max()); // #1
   lateral_bdr = 0;
   lateral_bdr[1] = 1;                                      // Lateral sides (left and right)
   Array<int> bottom_bdr(solid_pmesh.bdr_attributes.Max()); // #2
   bottom_bdr = 0;
   bottom_bdr[0] = 1; // Bottom side

   // Longitudinal problem solid BCs
   elasticity.AddFixedConstraint(bottom_bdr); // u(ȳ=-H̄,t)=0 for elastic-piston exact-pressure verification
   if (!periodic)
   {
      elasticity.AddFixedConstraint(lateral_bdr, 0); // x displacement fixed on the lateral sides
   }


   // Grid functions for PrescribedVelocity BC (Neumann-Dirichlet coupling)
   // - fluid_vel_onSolid_gf : fluid velocity at t_{n+1}, updated in the convergence loop
   // - nd_prev_vel_gf       : solid velocity at t_n, updated after each converged time step
   // - nd_prev_acc_gf       : solid acceleration at t_n, updated after each converged time step
   // DerivedAccelerationCoefficient (inside ElasticityOperator) reads these three
   // GridFunctions to compute the consistent boundary acceleration.
   ParGridFunction nd_prev_vel_gf(elasticity_fes); nd_prev_vel_gf = 0.0;
   ParGridFunction nd_prev_acc_gf(elasticity_fes); nd_prev_acc_gf = 0.0;

   VectorCoefficient *robin_coeff = nullptr; // Coefficient for Robin BC, used in Robin-Robin coupling
   switch (coupling)
   {
   case CouplingType::Dirichlet_Neumann:
   case CouplingType::Robin_Neumann:
   case CouplingType::RobinAMP_Neumann:
      elasticity.AddBoundaryLoad(&fluid_sigmaN_onSolid_coeff, solid_fs_interface_attr, -1.0, false);
      break;
   case CouplingType::Neumann_Dirichlet:
      elasticity.AddPrescribedVelocity(&fluid_vel_onSolid_gf, &nd_prev_vel_gf, &nd_prev_acc_gf, solid_fs_interface_attr);
      break;
   case CouplingType::Robin_Robin:
      robin_coeff = new VectorSumCoefficient(fluid_vel_onSolid_coeff, fluid_sigmaN_onSolid_coeff, DD_ctx.alphaSolid, -1.0);
      elasticity.AddRobinBC(DD_ctx.alphaSolid, robin_coeff, solid_fs_interface_attr, false);
      break;
   default:
      break;
   }

   elasticity.Setup();

   ////////////////////
   // MESH DEFORMATION
   ////////////////////

   Array<int> fluid_mesh_deformation_ess_bdr(fluid_pmesh.bdr_attributes.Max());
   // fluid_mesh_deformation_ess_bdr = 1;
   fluid_mesh_deformation_ess_bdr = no_slip;
   fluid_mesh_deformation_ess_bdr[2] = 1; // Interface attributes
   fluid_mesh_deformation_ess_bdr[3] = 1;

   fluid_mesh_deformation->SetEssentialBoundary(fluid_mesh_deformation_ess_bdr);

   bool pa_mesh = false;
   int prec_type_mesh = 1;
   switch (mesh_deformation_type)
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
   BidirectionalInterfaceTransfer finder_fluid_to_solid(u_gf->ParFESpace(), elasticity_fes, parent_interface_attr, TransferBackend::GSLIB, fluid_pmesh.GetComm(), false, true);
   InterfaceTransfer finder_solid_to_mesh(solid_vel_gf.ParFESpace(), mesh_nodes_fes, parent_interface_attr, TransferBackend::GSLIB, solid_pmesh.GetComm(), true);

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

   // GridFunctionCoefficients for error evaluation and relaxation steps
   ParGridFunction *solid_disp_prev_gf = new ParGridFunction(elasticity_fes);
   *solid_disp_prev_gf = 0.0;
   ParGridFunction *solid_vel_prev_gf = new ParGridFunction(elasticity_fes);
   *solid_vel_prev_gf = 0.0;
   ParGridFunction *fluid_sigmaN_onSolid_prev_gf = new ParGridFunction(elasticity_fes);
   *fluid_sigmaN_onSolid_prev_gf = 0.0;
   ParGridFunction *u_fluid_prev_gf = new ParGridFunction(u_gf->ParFESpace());
   *u_fluid_prev_gf = 0.0;
   ParGridFunction *p_fluid_prev_gf = new ParGridFunction(p_gf->ParFESpace());
   *p_fluid_prev_gf = 0.0;
   ParGridFunction *solid_sigmaN_onFluid_prev_gf = new ParGridFunction(u_gf->ParFESpace());
   *solid_sigmaN_onFluid_prev_gf = 0.0;
   ParGridFunction *solid_vel_onFluid_prev_gf = new ParGridFunction(u_gf->ParFESpace());
   *solid_vel_onFluid_prev_gf = 0.0;
   ParGridFunction *fluid_vel_onSolid_prev_gf = new ParGridFunction(elasticity_fes);
   *fluid_vel_onSolid_prev_gf = 0.0;
   VectorGridFunctionCoefficient solid_disp_prev_coeff(solid_disp_prev_gf);
   VectorGridFunctionCoefficient fluid_sigmaN_onSolid_prev_coeff(fluid_sigmaN_onSolid_prev_gf);
   VectorGridFunctionCoefficient solid_sigmaN_onFluid_prev_coeff(solid_sigmaN_onFluid_prev_gf); 
   VectorGridFunctionCoefficient solid_vel_onFluid_prev_coeff(solid_vel_onFluid_prev_gf);
   VectorGridFunctionCoefficient fluid_vel_onSolid_prev_coeff(fluid_vel_onSolid_prev_gf);
   VectorGridFunctionCoefficient u_fluid_prev_coeff(u_fluid_prev_gf);

   Vector p_fluid_prev, u_fluid_prev, solid_displacement_prev, solid_vel_prev, fluid_sigmaN_onSolid_prev, solid_sigmaN_onFluid_prev, solid_vel_onFluid_prev, fluid_vel_onSolid_prev;
   p_fluid_prev_gf->GetTrueDofs(p_fluid_prev);
   u_fluid_prev_gf->GetTrueDofs(u_fluid_prev);
   solid_disp_prev_gf->GetTrueDofs(solid_displacement_prev);
   solid_vel_prev_gf->GetTrueDofs(solid_vel_prev);
   solid_vel_onFluid_prev_gf->GetTrueDofs(solid_vel_onFluid_prev);
   fluid_sigmaN_onSolid_prev_gf->GetTrueDofs(fluid_sigmaN_onSolid_prev);
   solid_sigmaN_onFluid_prev_gf->GetTrueDofs(solid_sigmaN_onFluid_prev);
   fluid_vel_onSolid_prev_gf->GetTrueDofs(fluid_vel_onSolid_prev);

   // Extract the indices of elements at the interface and convert them to markers
   // Useful to restrict the computation of the L2 error to the elements on the interface
   Array<int> solid_interfaces_element_idx;
   Array<int> fluid_interfaces_element_idx;
   finder_fluid_to_solid.GetElementIdxSrc(fluid_interfaces_element_idx);
   finder_fluid_to_solid.GetElementIdxDst(solid_interfaces_element_idx);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Set up coefficients and output
   ///////////////////////////////////////////////////////////////////////////////////////////////

   ParLinearForm lift_drag_form(u_gf->ParFESpace());
   auto stress_integ = new BoundaryNormalStressIntegrator(*u_gf, *p_gf, *nu_gf, NS_ctx.density);
   stress_integ->SetIntRule(&navier.gll_ir_face);
   lift_drag_form.AddBoundaryIntegrator(stress_integ,
                                        fluid_fs_interface_attr);

   // auto sigmaN_coeff = VectorGridFunctionCoefficient(&sigmaN_gf);
   // ParLinearForm lift_drag_form2(u_gf->ParFESpace());
   // auto stress_integ2 = new VectorBoundaryLFIntegrator(sigmaN_coeff);
   // stress_integ2->SetIntRule(&navier.gll_ir_face);
   // lift_drag_form2.AddBoundaryIntegrator(stress_integ2,
   //                                      fluid_fs_interface_attr);

   ParGridFunction *normal_shear_stress_gf = new ParGridFunction(u_gf->ParFESpace()); *normal_shear_stress_gf = 0.0;
   VectorCoefficient *normal_shear_stress_coeff = navier.GetShearStressNormalTractionCoefficient();
   normal_shear_stress_gf->ProjectBdrCoefficient(*normal_shear_stress_coeff, fluid_fs_interface_attr);

   ParaViewDataCollection pvdc_fluid("Fluid", &fluid_pmesh);
   pvdc_fluid.SetPrefixPath(outfolder);
   pvdc_fluid.SetDataFormat(VTKFormat::BINARY);
   pvdc_fluid.SetHighOrderOutput(true);
   pvdc_fluid.SetLevelsOfDetail(NS_ctx.u_order);
   pvdc_fluid.SetCycle(current_step);
   pvdc_fluid.SetTime(0.0);
   pvdc_fluid.RegisterField("velocity", u_gf);
   pvdc_fluid.RegisterField("provisional_velocity", navier.GetProvisionalVelocity());
   pvdc_fluid.RegisterField("pressure", p_gf);
   pvdc_fluid.RegisterField("solid_velocity_onFluid", &solid_vel_onFluid_gf);
   pvdc_fluid.RegisterField("sigmaS_onFluid", &solid_sigmaN_onFluid_gf);
   pvdc_fluid.RegisterField("mesh_velocity", mesh_velocity_gf); // CHECK --> Can we store the mesh velocity even tho it has a different fes?
   pvdc_fluid.RegisterField("normal_shear_stress", normal_shear_stress_gf);
   pvdc_fluid.Save();

   ParaViewDataCollection pvdc_solid("Solid", &solid_pmesh);
   pvdc_solid.SetPrefixPath(outfolder);
   pvdc_solid.SetDataFormat(VTKFormat::BINARY);
   pvdc_solid.SetHighOrderOutput(true);
   pvdc_solid.SetLevelsOfDetail(Elasticity_ctx.order);
   pvdc_solid.SetCycle(current_step);
   pvdc_solid.SetTime(0.0);
   pvdc_solid.RegisterField("displacement", &solid_displacement_gf);
   pvdc_solid.RegisterField("velocity", &solid_vel_gf);
   pvdc_solid.RegisterField("sigmaN", &fluid_sigmaN_onSolid_gf);
   pvdc_solid.RegisterField("fluid_vel_onSolid", &fluid_vel_onSolid_gf);
   pvdc_solid.Save();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Solve FSI problem
   ///////////////////////////////////////////////////////////////////////////////////////////////

   Array<real_t> lift_history, drag_history, delta_p_history;


   ////////////////////////////////////////////////////////////////////////////////////////////////
   /// 8.2 ----- Coupled FSI
   ////////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\nStarting coupled FSI problem..." << std::endl;

   // Setup Aitken acceleration
   real_t omega_max = 1.0;                       // Maximum relaxation factor
   real_t omega_min = 1e-1;                      // Minimum relaxation factor
   real_t omega_solid_velocity_init = DD_ctx.omega_solid_velocity; // Relaxation factor for the fluid problem
   real_t omega_solid_sigma_init = DD_ctx.omega_solid_sigma; // Relaxation factor for the solid problem
   int init_steps = 3;                           // Number of initial steps with constant relaxation factor

   AitkenRelaxation aitken_solid_velocity(solid_pmesh.GetComm(), omega_solid_velocity_init, omega_min, omega_max, init_steps);
   AitkenRelaxation aitken_solid_sigma(fluid_pmesh.GetComm(), omega_solid_sigma_init, omega_min, omega_max, init_steps);
   AitkenRelaxation aitken_fluid_velocity(fluid_pmesh.GetComm(), 1.0, omega_min, omega_max, init_steps);
   aitken_solid_velocity.SetSafetyFactor(DD_ctx.safety_factor);
   aitken_solid_sigma.SetSafetyFactor(DD_ctx.safety_factor);
   aitken_fluid_velocity.SetSafetyFactor(DD_ctx.safety_factor);

   aitken_solid_velocity.SetSize(solid_vel);
   aitken_solid_sigma.SetSize(solid_sigmaN_onFluid);
   aitken_fluid_velocity.SetSize(u_fluid);

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
          << std::setw(16) << "CFL"
          << std::setw(16) << "VelNorm"
          << std::setw(16) << "p_probe"
          << std::setw(16) << "disp_x"
          << std::setw(16) << "disp_y"
          << std::setw(16) << "subiter"
          << std::setw(16) << "w_solid"
          << std::setw(16) << "w_fluid"
          << std::endl;
      out << "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
   }

   // Store initial velocity and displacement for the elasticity problem
   // CHECK: is this doing a deep copy?
   solid_displacement_gf.GetTrueDofs(solid_displacement_tn);
   solid_vel_gf.GetTrueDofs(solid_vel_tn);
   u_gf->GetTrueDofs(u_fluid_tn);
   p_gf->GetTrueDofs(p_fluid_tn);

   // Initialize PrescribedVelocity BC state for Neumann-Dirichlet coupling
   if (coupling == CouplingType::Neumann_Dirichlet)
   {
      // Project from analytic initial conditions
      VectorFunctionCoefficient analytic_solid_vel_coeff(dim, solid_vel_vec_analytic);
      VectorFunctionCoefficient analytic_solid_acc_coeff(dim, solid_acc_vec_analytic);
      analytic_solid_vel_coeff.SetTime(0.0); 
      analytic_solid_acc_coeff.SetTime(0.0);
      nd_prev_vel_gf.ProjectCoefficient(analytic_solid_vel_coeff); 
      nd_prev_acc_gf.ProjectCoefficient(analytic_solid_acc_coeff);
   }

   last_step = false;                                                           // Reset last_step for the coupled FSI problem
   current_step = 0;                                                            // Reset current step for the coupled FSI problem, since preload was only for CSM and NS is not initialized
   bool initial_csm_step = true; // For the longitudinal problem this is the first csm step
   real_t fluid_mesh_displacement_norm = 0.0;
   int paraview_step = 0; // ParaView step counter

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
      bool converged_fluid = false;
      bool converged_solid = false;

      int iter = 0;
      const int max_iter = 1;
      const int min_iter = 1;
      real_t rel_tol = 1e-5;
      real_t abs_tol = 1e-7;

      // Retrieve the computed provisional velocity which will be used in the convergence loop

      // Store initial conditions
      solid_displacement_gf.GetTrueDofs(solid_displacement_tn);
      solid_displacement_gf.GetTrueDofs(solid_displacement_prev);
      solid_vel_gf.GetTrueDofs(solid_vel_tn);
      u_gf->GetTrueDofs(u_fluid_tn);
      p_gf->GetTrueDofs(p_fluid_tn);
      fluid_sigmaN_onSolid_gf.GetTrueDofs(fluid_sigmaN_onSolid);
      solid_vel_onFluid_gf.GetTrueDofs(solid_vel_onFluid);
      solid_sigmaN_onFluid_gf.GetTrueDofs(solid_sigmaN_onFluid);
      fluid_vel_onSolid_gf.GetTrueDofs(fluid_vel_onSolid);
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
      solid_displacement_prev = solid_displacement;


      real_t debug_time = 0.0;

      //while (!converged && iter <= max_iter)
     // {
         // Store the previous iterates for variables used in relaxation
         u_fluid_prev = u_fluid;
         p_fluid_prev = p_fluid;
         solid_vel_prev = solid_vel;
         fluid_vel_onSolid_prev = fluid_vel_onSolid;
         u_fluid_prev_gf->SetFromTrueDofs(u_fluid_prev);
         p_fluid_prev_gf->SetFromTrueDofs(p_fluid_prev);
         solid_vel_prev_gf->SetFromTrueDofs(solid_vel_prev);
         fluid_vel_onSolid_prev_gf->SetFromTrueDofs(fluid_vel_onSolid_prev);

         // Store the previous iterates for variables used in relaxation
         fluid_sigmaN_onSolid_prev = fluid_sigmaN_onSolid;
         solid_sigmaN_onFluid_prev = solid_sigmaN_onFluid;
         solid_vel_onFluid_prev = solid_vel_onFluid;
         fluid_sigmaN_onSolid_prev_gf->SetFromTrueDofs(fluid_sigmaN_onSolid_prev);
         solid_sigmaN_onFluid_prev_gf->SetFromTrueDofs(solid_sigmaN_onFluid_prev);
         solid_vel_onFluid_prev_gf->SetFromTrueDofs(solid_vel_onFluid_prev);

         // NOTE: solid_displacement_prev is intentionally NOT updated here.
         // It must still hold u^{n+1,k-1} when the mesh deformation is solved,
         // so that the relaxation omega*u^{n+1,k} + (1-omega)*u^{n+1,k-1} is correct.
         // It is updated to u^{n+1,k} at the end of the mesh deformation block.


         // --------------------------------------------------------------------------------- //
         //// ----- 1. Interpolate SOLID --> FLUID and relax interface quantities -----
         // For all coupling types: interpolate solid velocity as Dirichlet BC for fluid.
         // For RR/NN only: also interpolate solid traction (Robin BC for fluid).
         // --------------------------------------------------------------------------------- //

         if (Mpi::Root() && verbose)
            mfem::out << "Interpolating velocity SOLID --> FLUID...";

         ///<--- Solid velocity
         if (need_solid_velocity_on_fluid)
         { // Interpolate
            // Prescribe exact solid velocity at t_{n+1} directly (no interpolation from solid solver)
            VectorFunctionCoefficient exact_solid_vel_coeff(dim, fluid_vel_vec_analytic);
            exact_solid_vel_coeff.SetTime(current_time + dt);
            solid_vel_onFluid_gf.ProjectBdrCoefficient(exact_solid_vel_coeff, fluid_fs_interface_attr);
            solid_vel_onFluid_gf.GetTrueDofs(solid_vel_onFluid);
         }

         ///<--- Solid traction
         if (need_solid_traction_on_fluid)
         {
            // Prescribe exact σ_s · n_s at the interface (no interpolation from solid solver)
            VectorFunctionCoefficient exact_solid_sigma_coeff(dim, solid_sigma_n_vec_analytic);
            exact_solid_sigma_coeff.SetTime(current_time + dt);
            solid_sigmaN_onFluid_gf.ProjectBdrCoefficient(exact_solid_sigma_coeff, fluid_fs_interface_attr);
            solid_sigmaN_onFluid_gf.GetTrueDofs(solid_sigmaN_onFluid);
         }

         if (Mpi::Root() && verbose)
            mfem::out << " done." << std::endl;


         // --------------------------------------------------------------------------------- //
         //// ----- 2. Solve Mesh Deformation and compute mesh velocity -----
         // --------------------------------------------------------------------------------- //

         // This way, the mesh velocity in ALE is consistent with the BC being applied from the solid side
         if (Mpi::Root() && verbose)
            mfem::out << "Solving Mesh Deformation..." << std::endl;

         // In separate mode, pre-load solid_displacement_gf with the exact analytic
         // displacement at t_{n+1} so that the mesh deformation is driven by
         // the exact solution rather than by the (not-yet-run) solid solver.
         {
            VectorFunctionCoefficient exact_disp_coeff(dim, solid_disp_vec_analytic);
            exact_disp_coeff.SetTime(current_time + dt);
            solid_displacement_gf.ProjectCoefficient(exact_disp_coeff);
            solid_displacement_gf.GetTrueDofs(solid_displacement);
         }

         Vector mesh_velocity;
         if (fluid_mesh_deformation->IsNonIncremental())
         { 
            Vector solid_displacement_relaxed;
            solid_displacement_gf.GetTrueDofs(solid_displacement_relaxed);
            solid_displacement_relaxed *= DD_ctx.omega_solid_velocity;
            solid_displacement_relaxed.Add(1.0 - DD_ctx.omega_solid_velocity, solid_displacement_prev);
            solid_displacement_relaxed_gf.SetFromTrueDofs(solid_displacement_relaxed);
          
            *mesh_displacement_gf = 0.0;
            finder_solid_to_mesh.Interpolate(solid_displacement_relaxed_gf, *mesh_displacement_gf);
            fluid_mesh_deformation->Solve(*mesh_displacement_gf, false);
            mesh_displacement_gf->GetTrueDofs(mesh_velocity);
            mesh_velocity -= mesh_displacement_tn;
            fluid_mesh_displacement_norm = mesh_velocity.Norml2();
         }
         else
         {
            Vector rel_displacement;
            solid_displacement_gf.GetTrueDofs(rel_displacement);
            rel_displacement *= DD_ctx.omega_solid_velocity;
            rel_displacement.Add(1.0 - DD_ctx.omega_solid_velocity, solid_displacement_prev);

            rel_displacement -= solid_displacement_tn;
            solid_relative_displacement_gf->SetFromTrueDofs(rel_displacement);
            *mesh_displacement_gf = 0.0;
            finder_solid_to_mesh.Interpolate(*solid_relative_displacement_gf,
                                              *mesh_displacement_gf);
            fluid_mesh_deformation->Solve(*mesh_displacement_gf, false);
            mesh_displacement_gf->GetTrueDofs(mesh_velocity);
            fluid_mesh_displacement_norm = mesh_velocity.Norml2();
         }
         mesh_velocity *= (1.0 / dt);
         mesh_velocity_gf->SetFromTrueDofs(mesh_velocity);

         // Now that we have the relaxed mesh velocity, we can update the previous solid_displacement
         // Otherwise, if updated before we'd have had solid_displacement_prev = solid_displacement
         solid_displacement_prev = solid_displacement;
         solid_disp_prev_gf->SetFromTrueDofs(solid_displacement_prev);

         if (Mpi::Root() && verbose)
            mfem::out << " done." << std::endl;

         // --------------------------------------------------------------------------------- //
         //// ----- 3. Solve Navier-Stokes on FLUID -----
         // Reset to t_n state each iteration (partitioned scheme).
         // --------------------------------------------------------------------------------- //

         if (Mpi::Root() && verbose)
            mfem::out << "Solving Navier-Stokes on FLUID...";

         un_next_gf->SetFromTrueDofs(u_fluid_tn);
         p_gf->SetFromTrueDofs(p_fluid_tn);

         navier.Step(current_time, dt, step, true);

         un_next_gf->GetTrueDofs(u_fluid);
         p_gf->GetTrueDofs(p_fluid);

         normal_shear_stress_gf->ProjectBdrCoefficient(*normal_shear_stress_coeff, fluid_fs_interface_attr);

         if (Mpi::Root() && verbose)
            mfem::out << " done." << std::endl;

         // --------------------------------------------------------------------------------- //
         //// ----- 4. Interpolate FLUID --> SOLID -----
         // D-N / Robin: compute fluid traction and interpolate to solid mesh (Neumann BC).
         // N-D: interpolate fluid velocity to solid mesh (Dirichlet BC via PrescribedVelocity).
         // --------------------------------------------------------------------------------- //

         if (need_fluid_velocity_on_solid)
         {
            if (Mpi::Root() && verbose)
               mfem::out << "Interpolating velocity FLUID --> SOLID...";

            // Prescribe exact fluid velocity at the interface (no interpolation from fluid solver)
            VectorFunctionCoefficient exact_fluid_vel_coeff(dim, fluid_vel_vec_analytic);
            exact_fluid_vel_coeff.SetTime(current_time + dt);
            fluid_vel_onSolid_gf.ProjectBdrCoefficient(exact_fluid_vel_coeff, solid_fs_interface_attr);
            fluid_vel_onSolid_gf.GetTrueDofs(fluid_vel_onSolid);

            if (Mpi::Root() && verbose)
               mfem::out << " done." << std::endl;
         }

         if (need_fluid_traction_on_solid)
         {
            if (Mpi::Root() && verbose)
               mfem::out << "Interpolating traction FLUID --> SOLID...";

            // Prescribe exact σ_f · n_f at the interface (no interpolation from fluid solver).
            // Note: AddBoundaryLoad applies a -1 scale, so the load on the solid is -fluid_sigmaN_onSolid.
            VectorFunctionCoefficient exact_fluid_sigma_coeff(dim, fluid_sigma_n_vec_analytic);
            exact_fluid_sigma_coeff.SetTime(current_time + dt);
            fluid_sigmaN_onSolid_gf.ProjectBdrCoefficient(exact_fluid_sigma_coeff, solid_fs_interface_attr);
            fluid_sigmaN_onSolid_gf.GetTrueDofs(fluid_sigmaN_onSolid);

            if (Mpi::Root() && verbose)
               mfem::out << " done." << std::endl;
         }

         // --------------------------------------------------------------------------------- //
         //// ----- 4. Solve Elasticity on SOLID -----
         // Reset to t_n state each iteration (partitioned scheme).
         // Driven by fluid traction from step 3.
         // --------------------------------------------------------------------------------- //

         if (Mpi::Root() && verbose)
            mfem::out << "Solving Elasticity on SOLID...";

         solid_displacement = solid_displacement_tn;
         solid_vel = solid_vel_tn;
         elasticity.ResetState(acceleration_tn);

         elasticity.Step(solid_displacement, solid_vel, current_time, dt, true);

         solid_vel_gf.SetFromTrueDofs(solid_vel);
         solid_displacement_gf.SetFromTrueDofs(solid_displacement);

         if (Mpi::Root() && verbose)
            mfem::out << " done." << std::endl;

         // --------------------------------------------------------------------------------- //
         //// ----- 5. Check convergence (coupling-aware) -----
         //
         // SOLID side: check quantities transferred FLUID -> SOLID
         //   - D-N / R-N / R-AMP-N : fluid traction on solid
         //   - N-D                 : fluid velocity on solid
         //
         // FLUID side: check quantities transferred SOLID -> FLUID
         //   - D-N                 : solid velocity on fluid
         //   - R-N / R-AMP-N       : solid velocity + solid traction on fluid
         //   - N-D                 : solid traction on fluid
         // --------------------------------------------------------------------------------- //

         real_t global_abs_err_solid = 0.0, global_rel_err_solid = 0.0;
         real_t global_abs_err_fluid = 0.0, global_rel_err_fluid = 0.0;

         constexpr real_t rel_eps = 1e-14;
         auto safe_rel = [&](real_t abs_err, real_t norm) -> real_t
         {
            return (norm > rel_eps) ? (abs_err / norm) : abs_err;
         };

         // ---------- SOLID convergence metrics (FLUID -> SOLID) ----------
         if (need_fluid_traction_on_solid)
         {
            real_t norm_sigmaF_onSolid = GlobalLpNorm(
                2.0, fluid_sigmaN_onSolid_gf.Norml2(), solid_pmesh.GetComm());

            real_t abs_sigmaF_onSolid = fluid_sigmaN_onSolid_gf.ComputeL2Error(
                fluid_sigmaN_onSolid_prev_coeff, irs_disp, &solid_interfaces_element_idx);

            real_t rel_sigmaF_onSolid = safe_rel(abs_sigmaF_onSolid, norm_sigmaF_onSolid);

            global_abs_err_solid = std::max(global_abs_err_solid, abs_sigmaF_onSolid);
            global_rel_err_solid = std::max(global_rel_err_solid, rel_sigmaF_onSolid);
         }

         if (need_fluid_velocity_on_solid)
         {
            real_t norm_uF_onSolid = GlobalLpNorm(
                2.0, fluid_vel_onSolid_gf.Norml2(), solid_pmesh.GetComm());

            real_t abs_uF_onSolid = fluid_vel_onSolid_gf.ComputeL2Error(
                fluid_vel_onSolid_prev_coeff, irs_disp, &solid_interfaces_element_idx);

            real_t rel_uF_onSolid = safe_rel(abs_uF_onSolid, norm_uF_onSolid);

            global_abs_err_solid = std::max(global_abs_err_solid, abs_uF_onSolid);
            global_rel_err_solid = std::max(global_rel_err_solid, rel_uF_onSolid);
         }

         // ---------- FLUID convergence metrics (SOLID -> FLUID) ----------
         if (need_solid_velocity_on_fluid)
         {
            real_t norm_uS_onFluid = GlobalLpNorm(
                2.0, solid_vel_onFluid_gf.Norml2(), fluid_pmesh.GetComm());

            real_t abs_uS_onFluid = solid_vel_onFluid_gf.ComputeL2Error(
                solid_vel_onFluid_prev_coeff, irs_vel, &fluid_interfaces_element_idx);

            real_t rel_uS_onFluid = safe_rel(abs_uS_onFluid, norm_uS_onFluid);

            global_abs_err_fluid = std::max(global_abs_err_fluid, abs_uS_onFluid);
            global_rel_err_fluid = std::max(global_rel_err_fluid, rel_uS_onFluid);
         }

         if (need_solid_traction_on_fluid)
         {
            real_t norm_sigmaS_onFluid = GlobalLpNorm(
                2.0, solid_sigmaN_onFluid_gf.Norml2(), fluid_pmesh.GetComm());

            real_t abs_sigmaS_onFluid = solid_sigmaN_onFluid_gf.ComputeL2Error(
                solid_sigmaN_onFluid_prev_coeff, irs_vel, &fluid_interfaces_element_idx);

            real_t rel_sigmaS_onFluid = safe_rel(abs_sigmaS_onFluid, norm_sigmaS_onFluid);

            global_abs_err_fluid = std::max(global_abs_err_fluid, abs_sigmaS_onFluid);
            global_rel_err_fluid = std::max(global_rel_err_fluid, rel_sigmaS_onFluid);
         }

         if (DD_ctx.relative_convergence)
         {
            converged_fluid = (global_rel_err_fluid < rel_tol);
            converged_solid = (global_rel_err_solid < rel_tol);
         }
         else
         {
            converged_fluid = (global_abs_err_fluid < abs_tol);
            converged_solid = (global_abs_err_solid < abs_tol);
         }

         converged = converged_fluid && converged_solid && (iter >= min_iter - 1);

         iter++;

         omega_history_fluid.Append(DD_ctx.omega_solid_velocity);
         omega_history_solid.Append(DD_ctx.omega_solid_sigma);


         // DEBUG --> output paraview every iteration of the convergence loop
         /*debug_time += previous_dt; // Update current time for visualization purposes only (not used in the solver)
         pvdc_fluid.SetCycle(iter);
         pvdc_fluid.SetTime(debug_time); // fictitious time just for visualization purposes
         pvdc_fluid.Save();
         pvdc_solid.SetCycle(iter);
         pvdc_solid.SetTime(debug_time); // fictitious time just for visualization purposes
         pvdc_solid.Save();*/
      //}
      // --------------------------------------------------------------------------------- //
      //// ----- END OF CONVERGENCE LOOP -----
      // --------------------------------------------------------------------------------- //

      // Reset the residual buffer for the aitken relaxation
      aitken_solid_velocity.Reset();
      aitken_solid_sigma.Reset();

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

      // Elastic piston verification: L2 errors against the analytic solution
      // at the new time level.
      {
         ParGridFunction us_gf(elasticity_fes), vs_gf(elasticity_fes);
         us_gf.SetFromTrueDofs(solid_displacement);
         vs_gf.SetFromTrueDofs(solid_vel);

         VectorFunctionCoefficient us_ex(dim, solid_disp_vec_analytic);
         VectorFunctionCoefficient vs_ex(dim, solid_vel_vec_analytic);
         VectorFunctionCoefficient uf_ex(dim, fluid_vel_vec_analytic);
         FunctionCoefficient pf_ex(fluid_pressure_analytic);
         us_ex.SetTime(current_time);
         vs_ex.SetTime(current_time);
         uf_ex.SetTime(current_time);
         pf_ex.SetTime(current_time);

         const real_t errs[4] = {us_gf.ComputeL2Error(us_ex),
                                 vs_gf.ComputeL2Error(vs_ex),
                                 u_gf->ComputeL2Error(uf_ex),
                                 p_gf->ComputeL2Error(pf_ex)
                                };
         for (int i = 0; i < 4; i++)
         {
            piston_max_err[i] = std::max(piston_max_err[i], errs[i]);
         }
         if (Mpi::Root())
         {
            out << "Piston L2 errors (t = " << std::scientific
                << std::setprecision(4) << current_time << "): "
                << "solid disp = " << errs[0] << " | solid vel = " << errs[1]
                << " | fluid vel = " << errs[2] << " | fluid pres = " << errs[3]
                << std::endl;
         }
      }


      // Update PrescribedVelocity BC state for next time step (Neumann-Dirichlet)
      if (need_fluid_velocity_on_solid)
      {
         nd_prev_vel_gf.SetFromTrueDofs(solid_vel);
         Vector *acc_state = elasticity.GetLatestState();
         if (acc_state != nullptr)
            nd_prev_acc_gf.SetFromTrueDofs(*acc_state);
      }

      if (initial_csm_step)
      {
         initial_csm_step = false; // We are done with the initial CSM step
      }

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
      bool compute_output = false;
      real_t disp_x = 0.0, disp_y = 0.0;
      real_t p_probe1 = 0.0, p_probe2 = 0.0, delta_p = 0.0;
      real_t vel_norm1 = 0.0, vel_norm2 = 0.0;
      if (compute_output) // TODO: change with fixes from test_csm_only
      {
         Vector point1(2), point2(2);
         point1(0) = 0.15; // Cylinder left vertex (fixed)
         point1(1) = 0.2;
         point2(0) = 0.6; // Beam right vertex (free) --> Note this will move with the mesh
         point2(1) = 0.2;
         DenseMatrix points(dim, 2);
         points.SetCol(0, point1);
         points.SetCol(1, point2);

         Array<int> elem_ids;
         Array<IntegrationPoint> ips;

         fluid_pmesh.FindPoints(points, elem_ids, ips);

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
             //<< std::setw(16) << std::scientific << std::setprecision(8) << lift
             //<< std::setw(16) << std::scientific << std::setprecision(8) << drag
             << std::setw(16) << std::scientific << std::setprecision(8) << cfl
             << std::setw(16) << std::scientific << std::setprecision(8) << vel_norm1
             << std::setw(16) << std::scientific << std::setprecision(8) << delta_p
             << std::setw(16) << std::scientific << std::setprecision(8) << disp_x
             << std::setw(16) << std::scientific << std::setprecision(8) << disp_y
             << std::setw(16) << std::scientific << std::setprecision(8) << iter
             << std::setw(16) << std::scientific << std::setprecision(8) << DD_ctx.omega_solid_sigma
               << std::setw(16) << std::scientific << std::setprecision(8) << DD_ctx.omega_solid_velocity
             << std::endl;
      }

   } // END OF TIME-INTEGRATION LOOP

   if (Mpi::Root())
   {
      out << "Piston max L2 errors over time: "
          << std::scientific << std::setprecision(4)
          << "solid disp = " << piston_max_err[0]
          << " | solid vel = " << piston_max_err[1]
          << " | fluid vel = " << piston_max_err[2]
          << " | fluid pres = " << piston_max_err[3] << std::endl;
   }

   navier.PrintTimingData();

   // Export the array of sub-iterations
   saveArray(subiter, outfolder, "subiter_count.txt");
   saveArray(omega_history_fluid, outfolder, "omega_solid_velocity.txt");
   saveArray(omega_history_solid, outfolder, "omega_solid_sigma.txt");

   saveArray(lift_history, outfolder, "lift.txt");
   saveArray(drag_history, outfolder, "drag.txt");
   saveArray(delta_p_history, outfolder, "delta_p.txt");

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 9. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   delete mesh_displacement_coeff;
   delete mesh_displacement_gf;
   delete normal_shear_stress_gf;

   if (!fluid_mesh_deformation->IsNonIncremental())
   {
      delete solid_relative_displacement_gf; // Relative solid displacement field
   }

   delete fluid_mesh_deformation;

   delete neumann_coeff;

   return 0;
}

ParMesh CreateSubmeshMesh(Mesh *parent_mesh, int domain_attr, std::vector<Vector> translations, int mesh_order)
{
   // Extract serial submeshes
   Array<int> domain_attributes(1);

   std::string filename;
   if (domain_attr == 1)
   {
      // Solid domain
      filename = "SolidPartitioning";
      domain_attributes[0] = 1;
   }
   else if (domain_attr == 2)
   {
      // Fluid domain
      filename = "FluidPartitioning";
      domain_attributes[0] = 2;
   }

   bool periodic = translations.size() > 0;

   Mesh *submesh = nullptr;
   if (periodic)
   {
      auto submesh_original = SubMesh::CreateFromDomain(*parent_mesh, domain_attributes);
      submesh_original.SetCurvature(mesh_order, true);
      // Make periodic
      submesh = new Mesh(Mesh::MakePeriodic(
          submesh_original,
          submesh_original.CreatePeriodicVertexMapping(translations)));
      submesh_original.Clear(); // Clear the original submesh to free memory
   }
   else
   {
      submesh = new Mesh(SubMesh::CreateFromDomain(*parent_mesh, domain_attributes));
      submesh->SetCurvature(mesh_order, false);
   }

   // Generate mesh partitioning and create a parallel mesh.
   // Partition type:
   // 0) METIS_PartGraphRecursive (sorted neighbor lists)
   // 1) METIS_PartGraphKway      (sorted neighbor lists) (default)
   // 2) METIS_PartGraphVKway     (sorted neighbor lists)
   // 3) METIS_PartGraphRecursive
   // 4) METIS_PartGraphKway
   // 5) METIS_PartGraphVKway
   int partition_type = 1;
   int np = num_procs;

   int *partitioning = submesh->GeneratePartitioning(np, partition_type);

   // Create parallel mesh
   ParMesh pmesh = ParMesh(MPI_COMM_WORLD, *submesh, partitioning, partition_type);
   ExportMeshwithPartitioning(outfolder, *submesh, partitioning, filename);
   delete[] partitioning;
   delete submesh;

   for (int l = 0; l < Mesh_ctx.parallel_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   pmesh.SetAttributes();
   pmesh.EnsureNodes();

   return pmesh;
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


// ============================================================================
// Elastic Piston Manufactured Solution
// ============================================================================
// Reference: Section 7.1, Eqs. (7.1)-(7.7)
//
// Manufactured solution with F_bar(tau) = alpha * cos(omega * tau):
//
//   u2(y_bar, t) = 2*alpha*sin(omega*(y_bar+H_bar)/cp)*sin(omega*t)
//
// Initial conditions (t=0):
//   u2(y_bar, 0)   = 0
//   u2_t(y_bar, 0) = 2*alpha*omega*sin(omega*(y_bar+H_bar)/cp)
//
//   v2(0)  = 2*alpha*omega*sin(omega*H_bar/cp)   [uniform in y]
//   p(y,0) = 0
//
// All functions below use the same context structs already defined
// in your solver (ElasticPiston_ctx, Elasticity_ctx, NS_ctx).
// ============================================================================

// ----------------------------------------------------------------------------
// SOLID: vertical displacement  u2(y_bar, t)
// ----------------------------------------------------------------------------
real_t solid_disp_analytic(const Vector &x, double time)
{
   const real_t omega  = ElasticPiston_ctx.omega;
   const real_t alpha  = ElasticPiston_ctx.alpha;
   const real_t cp     = ElasticPiston_ctx.cp_bar;
   const real_t H_bar  = ElasticPiston_ctx.H_bar_solid;

   // x(1) is the vertical coordinate y_bar in the solid reference domain
   const real_t y_bar  = x(1);
   const real_t phase  = omega * (y_bar + H_bar) / cp;

   return 2.0 * alpha * std::sin(phase) * std::sin(omega * time);
}

// ----------------------------------------------------------------------------
// SOLID: vertical velocity  u2_t(y_bar, t)
// ----------------------------------------------------------------------------
real_t solid_vel_analytic(const Vector &x, double time)
{
   const real_t omega  = ElasticPiston_ctx.omega;
   const real_t alpha  = ElasticPiston_ctx.alpha;
   const real_t cp     = ElasticPiston_ctx.cp_bar;
   const real_t H_bar  = ElasticPiston_ctx.H_bar_solid;

   const real_t y_bar  = x(1);
   const real_t phase  = omega * (y_bar + H_bar) / cp;

   return 2.0 * alpha * omega * std::sin(phase) * std::cos(omega * time);
}

// ----------------------------------------------------------------------------
// SOLID: displacement vector coefficient (2D: x-component = 0, y = u2)
// ----------------------------------------------------------------------------
void solid_disp_vec_analytic(const Vector &x, double time, Vector &u)
{
   u.SetSize(x.Size());
   u = 0.0;
   u(1) = solid_disp_analytic(x, time);
}

// ----------------------------------------------------------------------------
// SOLID: velocity vector coefficient (2D: x-component = 0, y = u2_t)
// ----------------------------------------------------------------------------
void solid_vel_vec_analytic(const Vector &x, double time, Vector &v)
{
   v.SetSize(x.Size());
   v = 0.0;
   v(1) = solid_vel_analytic(x, time);
}

// ----------------------------------------------------------------------------
// SOLID:  acceleration vector coefficient (2D: x-component = 0, y = u2_tt)
// ----------------------------------------------------------------------------
void solid_acc_vec_analytic(const Vector &x, double time, Vector &a)
{
   const real_t omega  = ElasticPiston_ctx.omega;
   const real_t alpha  = ElasticPiston_ctx.alpha;  
   const real_t cp     = ElasticPiston_ctx.cp_bar;
   const real_t H_bar  = ElasticPiston_ctx.H_bar_solid;
   const real_t y_bar  = x(1);
   const real_t phase  = omega * (y_bar + H_bar) / cp;
   const real_t u2_tt  = -2.0 * alpha * omega * omega * std::sin(phase) * std::sin(omega * time);

   a.SetSize(x.Size());
   a = 0.0;
   a(1) = u2_tt;
}

// ----------------------------------------------------------------------------
// FLUID: vertical velocity  v2(t)  — uniform in y, depends on t only
// ----------------------------------------------------------------------------
void fluid_vel_vec_analytic(const Vector &x, double time, Vector &v)
{
   const real_t omega  = ElasticPiston_ctx.omega;
   const real_t alpha  = ElasticPiston_ctx.alpha;
   const real_t cp     = ElasticPiston_ctx.cp_bar;
   const real_t H_bar  = ElasticPiston_ctx.H_bar_solid;

   // v2(t) = u2_t(0,t) = 2*alpha*omega*sin(omega*H_bar/cp)*cos(omega*t)
   const real_t phase  = omega * H_bar / cp;

   v.SetSize(x.Size());
   v = 0.0;
   v(1) = 2.0 * alpha * omega * std::sin(phase) * std::cos(omega * time);
}

// ----------------------------------------------------------------------------
// FLUID: pressure  p(y, t)  — Eq. (7.1)
//   p(y,t) = [(H-y)*pI(t) + (y-yI(t))*pH(t)] / (H - yI(t))
// ----------------------------------------------------------------------------
real_t fluid_pressure_analytic(const Vector &x, double time)
{
   const real_t omega      = ElasticPiston_ctx.omega;
   const real_t alpha      = ElasticPiston_ctx.alpha;
   const real_t cp         = ElasticPiston_ctx.cp_bar;
   const real_t H_bar      = ElasticPiston_ctx.H_bar_solid;
   const real_t H          = ElasticPiston_ctx.H_fluid;
   const real_t rho        = NS_ctx.density;
   const real_t lambda_2mu = Elasticity_ctx.lambda + 2.0 * Elasticity_ctx.mu;

   // y is the vertical coordinate in the fluid domain
   const real_t y      = x(1);
   const real_t phase  = omega * H_bar / cp;

   // Interface position yI(t) = u2(0,t)
   const real_t yI     = 2.0 * alpha * std::sin(phase) * std::sin(omega * time);

   // u2_tt(0,t) = -omega^2 * yI(t)
   const real_t yI_tt  = -omega * omega * yI;

   // u2_ybar(0,t)
   const real_t u2_yb  = (2.0 * alpha * omega / cp)
                         * std::sin(omega * time)
                         * std::cos(phase);

   // pI(t) = -(lambda+2mu)*u2_ybar(0,t)   [Eq. 7.6b]
   const real_t pI     = -lambda_2mu * u2_yb;

   // pH(t) from Eq. (7.7)
   const real_t pH     = -rho * (H - yI) * yI_tt - lambda_2mu * u2_yb;

   // Guard against interface being at the top wall (degenerate case)
   const real_t denom  = H - yI;
   if (std::abs(denom) < 1e-14) { return pI; }

   // p(y,t) from Eq. (7.1)
   return ((H - y) * pI + (y - yI) * pH) / denom;
}

real_t fluid_interface_pressure_analytic(const Vector &x, double time)
{
   const real_t omega      = ElasticPiston_ctx.omega;
   const real_t alpha      = ElasticPiston_ctx.alpha;
   const real_t cp         = ElasticPiston_ctx.cp_bar;
   const real_t H_bar      = ElasticPiston_ctx.H_bar_solid;
   const real_t H          = ElasticPiston_ctx.H_fluid;
   const real_t rho        = NS_ctx.density;
   const real_t lambda_2mu = Elasticity_ctx.lambda + 2.0 * Elasticity_ctx.mu;

   // y is the vertical coordinate in the fluid domain
   const real_t y      = x(1);
   const real_t phase  = omega * H_bar / cp;

   // Interface position yI(t) = u2(0,t)
   const real_t yI     = 2.0 * alpha * std::sin(phase) * std::sin(omega * time);

   // u2_tt(0,t) = -omega^2 * yI(t)
   const real_t yI_tt  = -omega * omega * yI;

   // u2_ybar(0,t)
   const real_t u2_yb  = (2.0 * alpha * omega / cp)
                         * std::sin(omega * time)
                         * std::cos(phase);

   // pI(t) = -(lambda+2mu)*u2_ybar(0,t)   [Eq. 7.6b]
   const real_t pI     = -lambda_2mu * u2_yb;

   return pI; 
}

real_t pressure_piston_top(const Vector &x, double time)
{
   const real_t omega = ElasticPiston_ctx.omega;
   const real_t alpha = ElasticPiston_ctx.alpha;
   const real_t cp_bar = ElasticPiston_ctx.cp_bar;
   const real_t H_bar = ElasticPiston_ctx.H_bar_solid;
   const real_t H = ElasticPiston_ctx.H_fluid;

   if (cp_bar <= 0.0)
   {
      return 0.0;
   }

   // yI(t) = ū2(0,t) = α cos(ω(t - H̄/c̄p)) - α cos(ω(t + H̄/c̄p)) = 2 α sin(ω H̄/c̄p) sin(ω t)
   const real_t phase = omega * H_bar / cp_bar;
   const real_t yI = 2.0 * alpha * std::sin(phase) * std::sin(omega * time);
   const real_t yI_tt = -omega * omega * yI;

   // ū2,ȳ(0,t) from ū2(ȳ,t) = F̄(t-(ȳ+H̄)/c̄p)-F̄(t+(ȳ+H̄)/c̄p), F̄(τ)=α cos(ωτ)
   const real_t uy_interface = (2.0 * alpha * omega / cp_bar) * std::sin(omega * time) * std::cos(phase);

   // From (7.7): ρ (H - ū2(0,t)) ū2,tt(0,t) + (λ̄+2μ̄) ū2,ȳ(0,t) = -pH(t)
   const real_t lambda_2mu = Elasticity_ctx.lambda + 2.0 * Elasticity_ctx.mu;
   //const real_t pH = -NS_ctx.density * (H - yI) * yI_tt - lambda_2mu * uy_interface;
   const real_t pH = -NS_ctx.density * H * yI_tt - lambda_2mu * uy_interface;

   return pH;
}

// ----------------------------------------------------------------------------
// INTERFACE: solid normal stress  σ_s · n_s  for the FLUID side  (separate mode)
//
// For the 1-D longitudinal manufactured solution the solid stress tensor has
// only one non-zero component:
//
//   σ_s,22 = (λ+2μ) ∂u2/∂ȳ
//
// At ȳ = 0 (the interface), with n_s = (0, +1):
//
//   σ_s · n_s = (0, (λ+2μ)(2α ω/c̄p) sin(ω·H̄/c̄p) cos(ω t))
//             = (0, −p_I(t))     [where p_I = −(λ+2μ) u2_ȳ(0,t)]
//
// This quantity is what the Robin/Neumann terms on the FLUID side expect
// (the solid_sigmaN_onFluid_gf grid function).
// ----------------------------------------------------------------------------
void solid_sigma_n_vec_analytic(const Vector &x, double time, Vector &v)
{
   const real_t omega      = ElasticPiston_ctx.omega;
   const real_t alpha      = ElasticPiston_ctx.alpha;
   const real_t cp         = ElasticPiston_ctx.cp_bar;
   const real_t H_bar      = ElasticPiston_ctx.H_bar_solid;
   const real_t lambda_2mu = Elasticity_ctx.lambda + 2.0 * Elasticity_ctx.mu;

   // ∂u2/∂ȳ at ȳ = 0: derivative of 2α sin(ω(ȳ+H̄)/c̄p) sin(ωt) w.r.t. ȳ
   const real_t phase    = omega * H_bar / cp;
   const real_t u2_ybar  = (2.0 * alpha * omega / cp) * std::sin(omega * time) * std::cos(phase);

   v.SetSize(x.Size());
   v = 0.0;
   v(1) = lambda_2mu * u2_ybar;  // = −p_I(t)
}

// ----------------------------------------------------------------------------
// INTERFACE: fluid normal stress  σ_f · n_f  for the SOLID side  (separate mode)
//
// For the 1-D longitudinal manufactured solution the fluid velocity is uniform
// so viscous contributions vanish. With n_f = (0, −1) (outward from fluid):
//
//   σ_f · n_f = (−p I) · n_f = p_I(t) · (0, +1)  [since n_f,2 = −1]
//
// p_I(t) = −(λ+2μ) u2_ȳ(0,t)   [Eq. 7.6b in the paper]
//
// This quantity feeds the fluid_sigmaN_onSolid_gf grid function.
// The elasticity solver applies it with a −1 scale factor
// (AddBoundaryLoad(..., -1.0, ...)), so the net traction on the solid is
//   t_s = −σ_f · n_f = (0, −p_I(t)).
// ----------------------------------------------------------------------------
void fluid_sigma_n_vec_analytic(const Vector &x, double time, Vector &v)
{
   const real_t omega      = ElasticPiston_ctx.omega;
   const real_t alpha      = ElasticPiston_ctx.alpha;
   const real_t cp         = ElasticPiston_ctx.cp_bar;
   const real_t H_bar      = ElasticPiston_ctx.H_bar_solid;
   const real_t lambda_2mu = Elasticity_ctx.lambda + 2.0 * Elasticity_ctx.mu;

   const real_t phase    = omega * H_bar / cp;
   const real_t u2_ybar  = (2.0 * alpha * omega / cp) * std::sin(omega * time) * std::cos(phase);

   // p_I = −(λ+2μ) u2_ȳ
   const real_t pI = -lambda_2mu * u2_ybar;

   // σ_f · n_f = (0, p_I)   [n_f = (0,-1), so −p*n_f,2 = +p_I]
   v.SetSize(x.Size());
   v = 0.0;
   v(1) = pI;
}

void noSlip_fun(const Vector &x, double t, Vector &u)
{
   u = 0.0;
}

real_t zero_pressure(const Vector &x, double t)
{
   return 0.0;
}

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