// =====================================================================================
//          Test Solid with Manufactured Solutions
// -------------------------------------------------------------------------------------
// Sample runs:
//
//     mpirun -np 4 ./test-csm-convergence -rs 3 -of ./Output/CSM-Convergence
//
// Manufactured solutions for the solid mechanics problem (linear elasticity) on a square domain.
//
//
// u = [cos(y)*sin(t)*sin(x), cos(x)*sin(t)*sin(y)]
//
// f = [2*lambda*cos(y)*sin(t)*sin(x) + 4*mu*cos(y)*sin(t)*sin(x) - rho*cos(y)*sin(t)*sin(x),
//      2*lambda*cos(x)*sin(t)*sin(y) + 4*mu*cos(x)*sin(t)*sin(y) - rho*cos(x)*sin(t)*sin(y)]
//

// Solid mechanics module
#include "structure/elasticity_solver.hpp"

// Mesh utilities
#include "fsi_utils.hpp"

// Output
#include <fstream>

using namespace mfem;

struct s_ElasticityContext // elasticity params
{
   int order = 2;        // polynomial order of the elasticity space
   real_t density = 1e3; // density of the solid
   real_t nu = 0.4;      // Poisson's ratio
   real_t mu = 2.0e6;    // Shear modulus/rigidity
   real_t lambda;
   real_t freq = 1.0;    // spatial frequency factor
   int solver_type = 15; // HHTAlphaSolver
} Elasticity_ctx;

struct s_MeshContext // mesh
{
   bool hex = true; // use quad elements, otherwise tri mesh
   int n = 5;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
} Mesh_ctx;

struct s_ProblemContext // problem context
{
   real_t t_final = 0.005;
   real_t dt = 1e-4; // time step
} Problem_ctx;

// Forward declarations
void prescribed_displacement_mms(const Vector &x, real_t t, Vector &v);
void velocity_mms(const Vector &x, real_t t, Vector &v);
void acceleration_mms(const Vector &x, real_t t, Vector &v);
void forcing_mms(const Vector &x, real_t t, Vector &v);

int main(int argc, char *argv[])
{

   /* { // Uncomment this block to ease MPI debuggings
      int i = 0;
      while (0 == i)
         sleep(5);
   } */

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 1. Initialize MPI and Hypre
   ///////////////////////////////////////////////////////////////////////////////////////////////
   Mpi::Init(argc, argv);
   Hypre::Init();

   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 2. Parse command-line options.
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // Output parameters
   const char *outfolder = "./Output/FSI";
   int save_freq = 1;

   int total_refinements = 3; // total number of serial refinements

   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&Elasticity_ctx.order, "-eo", "--elasticity-order",
                  "polynomial_order for solid elasticity space");
   args.AddOption(&Elasticity_ctx.mu, "-sh", "--shear_modulus",
                  "Shear modulus/rigidity");
   args.AddOption(&Elasticity_ctx.solver_type, "-st", "--solver-type",
                  "Time integrator for the elasticity solver.");

   args.AddOption(&Problem_ctx.dt, "-dt", "--dt",
                  "dt.");
   args.AddOption(&Problem_ctx.t_final, "-tf", "--t-final",
                  "t_final.");

   args.AddOption(&total_refinements, "-rs", "--rs", "serial_ref_levels");

   args.AddOption(&outfolder, "-of", "--output-folder",
                  "Output folder.");
   args.AddOption(&save_freq, "-sf", "--save-freq",
                  "Save frequency for output files.");

   args.ParseCheck();

   Elasticity_ctx.lambda = 2 * Elasticity_ctx.mu * Elasticity_ctx.nu /
                           (1 - 2 * Elasticity_ctx.nu);

   real_t l2_err_u_prev = 0.0;
   real_t h_prev = 0.0;

   if (Mpi::Root())
   {
      std::cout <<
                "----------------------------------------------------------------------------------------"
                << std::endl;
      std::cout << std::left << std::setw(16) << "DOFs " << std::setw(
                   16) << "h " << std::setw(16) << "L^2 error " << std::setw(
                   16) << "L^2 rate " << std::endl;
      std::cout <<
                "----------------------------------------------------------------------------------------"
                << std::endl;
   }

   // Refinement loop
   for (int serial_ref_levels = 0; serial_ref_levels < total_refinements;
        serial_ref_levels++)
   {

      ///////////////////////////////////////////////////////////////////////////////////////////////
      /// 3. Create serial Mesh, parallel and fluid/solid sub-meshes
      ///////////////////////////////////////////////////////////////////////////////////////////////

      Element::Type type;
      type = Mesh_ctx.hex ? Element::QUADRILATERAL : Element::TRIANGLE;

      Mesh mesh;
      mesh = Mesh::MakeCartesian2D(Mesh_ctx.n, Mesh_ctx.n, type, true);
      mesh.EnsureNodes();
      GridFunction *nodes = mesh.GetNodes();
      *nodes *= 2.0;
      *nodes -= 1.0;

      for (int l = 0; l < serial_ref_levels; l++)
      {
         mesh.UniformRefinement();
      }

      ParMesh solid_mesh(MPI_COMM_WORLD, mesh);
      mesh.Clear();

      ///////////////////////////////////////////////////////////////////////////////////////////////
      /// 4. Create Solid Solver
      ///////////////////////////////////////////////////////////////////////////////////////////////

      // Create the elasticity solver
      elasticity::ElasticitySolver elasticity(Elasticity_ctx.solver_type, &solid_mesh,
                                              Elasticity_ctx.order, Elasticity_ctx.density, Elasticity_ctx.lambda,
                                              Elasticity_ctx.mu, false);

      // Grid functions and vectors for the solid
      Vector solid_displacement;
      Vector solid_vel;

      auto elasticity_fes = elasticity.GetFESpace();

      ParGridFunction solid_displacement_gf(elasticity_fes);
      solid_displacement_gf = 0.0;
      solid_displacement_gf.GetTrueDofs(solid_displacement);

      ParGridFunction solid_vel_gf(elasticity_fes);
      solid_vel_gf = 0.0;
      solid_vel_gf.GetTrueDofs(solid_vel);

      ParGridFunction solid_displacement_exact_gf(elasticity_fes);
      solid_displacement_exact_gf = 0.0;
      ParGridFunction solid_vel_exact_gf(elasticity_fes);
      solid_vel_exact_gf = 0.0;
      ParGridFunction solid_acc_exact_gf(elasticity_fes);
      solid_acc_exact_gf = 0.0;
      ParGridFunction solid_forcing_exact_gf(elasticity_fes);
      solid_forcing_exact_gf = 0.0;

      VectorFunctionCoefficient prescribed_displacement_exact(2,
                                                              prescribed_displacement_mms);
      VectorFunctionCoefficient velocity_exact(2, velocity_mms);
      VectorFunctionCoefficient acceleration_exact(2, acceleration_mms);
      VectorFunctionCoefficient forcing_exact(2, forcing_mms);

      ///////////////////////////////////////////////////////////////////////////////////////////////
      /// 5. Set BCs
      ///////////////////////////////////////////////////////////////////////////////////////////////

      // Markers
      Array<int> domain_marker(solid_mesh.attributes.Max());
      domain_marker = 1;
      Array<int> solid_bdr(solid_mesh.bdr_attributes.Max());
      solid_bdr = 1;
      elasticity.AddPrescribedDisplacement(prescribed_displacement_mms, velocity_mms,
                                           acceleration_mms, solid_bdr);
      elasticity.AddBodyForce(forcing_mms, domain_marker);

      // After bcs have been set we can set up the elasticity operator
      elasticity.Setup();

      ///////////////////////////////////////////////////////////////////////////////////////////////
      /// 6. Set up output
      ///////////////////////////////////////////////////////////////////////////////////////////////

      ///////////////////////////////////////////////////////////////////////////////////////////////
      /// 6. Set up output
      ///////////////////////////////////////////////////////////////////////////////////////////////

      int step = 0;

      std::string name = "Solid-MMS-ref" + std::to_string(serial_ref_levels);
      ParaViewDataCollection pvdc_solid(name, &solid_mesh);
      pvdc_solid.SetPrefixPath(outfolder);
      pvdc_solid.SetDataFormat(VTKFormat::BINARY32);
      pvdc_solid.SetHighOrderOutput(true);
      pvdc_solid.SetLevelsOfDetail(Elasticity_ctx.order);
      pvdc_solid.SetCycle(step);
      pvdc_solid.SetTime(0.0);
      pvdc_solid.RegisterField("velocity", &solid_vel_gf);
      pvdc_solid.RegisterField("displacement", &solid_displacement_gf);
      pvdc_solid.RegisterField("exact_acceleration", &solid_acc_exact_gf);
      pvdc_solid.RegisterField("exact_velocity", &solid_vel_exact_gf);
      pvdc_solid.RegisterField("exact_displacement", &solid_displacement_exact_gf);
      pvdc_solid.RegisterField("exact_forcing", &solid_forcing_exact_gf);
      pvdc_solid.Save();

      ///////////////////////////////////////////////////////////////////////////////////////////////
      /// 7.  Solve problem
      ///////////////////////////////////////////////////////////////////////////////////////////////

      // Initial conditions
      prescribed_displacement_exact.SetTime(0.0);
      velocity_exact.SetTime(0.0);
      acceleration_exact.SetTime(0.0);
      forcing_exact.SetTime(0.0);
      solid_displacement_gf.ProjectCoefficient(prescribed_displacement_exact);
      solid_vel_gf.ProjectCoefficient(velocity_exact);
      solid_vel_exact_gf.ProjectCoefficient(velocity_exact);
      solid_displacement_exact_gf.ProjectCoefficient(prescribed_displacement_exact);
      solid_acc_exact_gf.ProjectCoefficient(acceleration_exact);
      solid_forcing_exact_gf.ProjectCoefficient(forcing_exact);
      pvdc_solid.SetCycle(0);
      pvdc_solid.SetTime(0.0);
      pvdc_solid.Save();

      real_t current_time = 0.0;
      bool last_step = false;
      for (int step = 0; !last_step; ++step) // TIME-INTEGRATION LOOP
      {
         if (current_time + Problem_ctx.dt >= Problem_ctx.t_final - Problem_ctx.dt / 2)
         {
            last_step = true;
         }

         // --------------------------------------------------------------------------------- //
         //// ----- Solve Elasticity -----
         // --------------------------------------------------------------------------------- //

         solid_displacement_gf.GetTrueDofs(solid_displacement);
         solid_vel_gf.GetTrueDofs(solid_vel);

         elasticity.Step(solid_displacement, solid_vel, current_time, Problem_ctx.dt);

         solid_displacement_gf.SetFromTrueDofs(solid_displacement);
         solid_vel_gf.SetFromTrueDofs(solid_vel);

         prescribed_displacement_exact.SetTime(current_time);
         velocity_exact.SetTime(current_time);
         acceleration_exact.SetTime(current_time);
         forcing_exact.SetTime(current_time);

         // --------------------------------------------------------------------------------- //
         //// ----- Output
         // --------------------------------------------------------------------------------- //

         // Compute exact solution
         solid_displacement_exact_gf.ProjectCoefficient(prescribed_displacement_exact);
         solid_vel_exact_gf.ProjectCoefficient(velocity_exact);
         solid_acc_exact_gf.ProjectCoefficient(acceleration_exact);
         solid_forcing_exact_gf.ProjectCoefficient(forcing_exact);

         // Export to paraview
         if (step % save_freq == 0 || last_step)
         {
            pvdc_solid.SetCycle(step + 1);
            pvdc_solid.SetTime(current_time);
            pvdc_solid.Save();
         }
      }

      // --------------------------------------------------------------------------------- //
      //// ----- Compute and print the L^2 and H^1 norms of the error.
      // --------------------------------------------------------------------------------- //

      real_t l2_err_u = 0.0;
      prescribed_displacement_exact.SetTime(current_time);
      l2_err_u = solid_displacement_gf.ComputeL2Error(prescribed_displacement_exact);

      real_t h_min = 0.0;
      real_t h_max = 0.0;
      real_t kappa_min = 0.0;
      real_t kappa_max = 0.0;
      solid_mesh.GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

      real_t l2_rate_u = (serial_ref_levels != 0) ? log(l2_err_u / l2_err_u_prev) /
                         log(h_min / h_prev) : 0.0;

      l2_err_u_prev = l2_err_u;
      h_prev = h_min;

      if (Mpi::Root())
      {
         std::cout << std::setw(16) << elasticity.GetProblemSize() << std::setw(
                      16) << h_min << std::setw(16) << l2_err_u << std::setw(16) << l2_rate_u <<
                   std::endl;
      }

      ///////////////////////////////////////////////////////////////////////////////////////////////
      /// 9. Cleanup
      ///////////////////////////////////////////////////////////////////////////////////////////////
   }

   return 0;
}

void prescribed_displacement_mms(const Vector &x, real_t t, Vector &v)
{
   real_t freq = Elasticity_ctx.freq;
   v = 0.0;
   v[0] = cos(2 * M_PI * x[1] * freq) * sin(2 * M_PI * x[0] * freq) * sin(
             t); // cos(2π*y*freq)*sin(2π*x*freq)*sin(t)
   v[1] = cos(2 * M_PI * x[0] * freq) * sin(2 * M_PI * x[1] * freq) * sin(
             t); // cos(2π*x*freq)*sin(2π*y*freq)*sin(t)
}

void velocity_mms(const Vector &x, real_t t, Vector &v)
{
   real_t freq = Elasticity_ctx.freq;
   v = 0.0;
   v[0] = cos(2 * M_PI * x[1] * freq) * sin(2 * M_PI * x[0] * freq) * cos(
             t); // cos(2π*y*freq)*sin(2π*x*freq)*cos(t)
   v[1] = cos(2 * M_PI * x[0] * freq) * sin(2 * M_PI * x[1] * freq) * cos(
             t); // cos(2π*x*freq)*sin(2π*y*freq)*cos(t)
}

void acceleration_mms(const Vector &x, real_t t, Vector &v)
{
   real_t freq = Elasticity_ctx.freq;
   v = 0.0;
   v[0] = -cos(2 * M_PI * x[1] * freq) * sin(2 * M_PI * x[0] * freq) * sin(
             t); // -cos(2π*y*freq)*sin(2π*x*freq)*sin(t)
   v[1] = -cos(2 * M_PI * x[0] * freq) * sin(2 * M_PI * x[1] * freq) * sin(
             t); // -cos(2π*x*freq)*sin(2π*y*freq)*sin(t)
}

void forcing_mms(const Vector &x, real_t t, Vector &v)
{
   v = 0.0;
   real_t freq = Elasticity_ctx.freq;
   real_t freq_sq = Elasticity_ctx.freq * Elasticity_ctx.freq;
   real_t pi_sq = M_PI * M_PI;
   real_t spatial_factor_lambda = 8 * freq_sq * pi_sq * Elasticity_ctx.lambda;
   real_t spatial_factor_mu = 16 * freq_sq * pi_sq * Elasticity_ctx.mu;

   v[0] = (spatial_factor_lambda + spatial_factor_mu - Elasticity_ctx.density) *
          cos(2 * M_PI * x[1] * freq) * sin(2 * M_PI * x[0] * freq) * sin(t); // f_x
   v[1] = (spatial_factor_lambda + spatial_factor_mu - Elasticity_ctx.density) *
          cos(2 * M_PI * x[0] * freq) * sin(2 * M_PI * x[1] * freq) * sin(t); // f_y
}
