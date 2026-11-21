// =====================================================================================
//          Test Solid with Manufactured Solutions — Prescribed Velocity BC
// -------------------------------------------------------------------------------------
// Sample runs:
//
//     mpirun -np 4 ./test-csm-mms-prescribedvel -rs 1 -eo 2 -dt 1e-4 -tf 1.0 -st 15 -of ./Output/CSM-MMS-VEL
//
// Manufactured solutions for the solid mechanics problem (linear elasticity) on a
// square domain, using AddPrescribedVelocity as the boundary condition.
// The MMS velocity is projected onto a GridFunction at t+dt before each step.
// The consistent boundary acceleration is derived internally by
// DerivedAccelerationCoefficient using the time integration scheme.
//
// u = [cos(2π*y*f)*sin(2π*x*f)*sin(t), cos(2π*x*f)*sin(2π*y*f)*sin(t)]
// =====================================================================================

#include "structure/elasticity_solver.hpp"
#include "fsi_utils.hpp"
#include <fstream>
#include <iomanip>

using namespace mfem;

struct s_ElasticityContext
{
   int order = 2;
   real_t density = 1e3;
   real_t nu = 0.4;
   real_t mu = 2.0e6;
   real_t lambda;
   real_t freq = 1.0;
   int solver_type = 15; // HHTAlphaSolver
} Elasticity_ctx;

struct s_MeshContext
{
   bool hex = true;
   int n = 10;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
} Mesh_ctx;

struct s_ProblemContext
{
   real_t t_final = 5.0;
   real_t dt = 1e-3;
} Problem_ctx;

// Forward declarations
void displacement_mms(const Vector &x, real_t t, Vector &v);
void velocity_mms(const Vector &x, real_t t, Vector &v);
void forcing_mms(const Vector &x, real_t t, Vector &v);

int main(int argc, char *argv[])
{
   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 1. Initialize MPI and Hypre
   ///////////////////////////////////////////////////////////////////////////////////////////////
   Mpi::Init(argc, argv);
   Hypre::Init();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 2. Parse command-line options
   ///////////////////////////////////////////////////////////////////////////////////////////////
   real_t current_time = 0.0;
   const char *outfolder = "./Output/CSM-MMS-VEL";
   int save_freq = 1;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&Elasticity_ctx.order, "-eo", "--elasticity-order",
                  "Polynomial order for solid elasticity space.");
   args.AddOption(&Elasticity_ctx.mu, "-sh", "--shear_modulus",
                  "Shear modulus/rigidity.");
   args.AddOption(&Elasticity_ctx.solver_type, "-st", "--solver-type",
                  "Time integrator for the elasticity solver.");
   args.AddOption(&Problem_ctx.dt, "-dt", "--dt", "Time step dt.");
   args.AddOption(&Problem_ctx.t_final, "-tf", "--t-final", "Final time.");
   args.AddOption(&Mesh_ctx.serial_ref_levels, "-rs", "--rs", "Serial refinement levels.");
   args.AddOption(&Mesh_ctx.parallel_ref_levels, "-rp", "--rp", "Parallel refinement levels.");
   args.AddOption(&outfolder, "-of", "--output-folder", "Output folder.");
   args.AddOption(&save_freq, "-sf", "--save-freq", "Save frequency for output files.");
   args.ParseCheck();

   Elasticity_ctx.lambda = 2.0 * Elasticity_ctx.mu * Elasticity_ctx.nu
                           / (1.0 - 2.0 * Elasticity_ctx.nu);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create mesh
   ///////////////////////////////////////////////////////////////////////////////////////////////
   if (Mpi::Root())
      mfem::out << "\nLoading mesh and refining in serial...";

   Element::Type type = Mesh_ctx.hex ? Element::QUADRILATERAL : Element::TRIANGLE;
   Mesh mesh = Mesh::MakeCartesian2D(Mesh_ctx.n, Mesh_ctx.n, type, true);
   mesh.EnsureNodes();
   GridFunction *nodes = mesh.GetNodes();
   *nodes *= 2.0;
   *nodes -= 1.0;

   for (int l = 0; l < Mesh_ctx.serial_ref_levels; l++)
      mesh.UniformRefinement();

   ParMesh solid_mesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   for (int l = 0; l < Mesh_ctx.parallel_ref_levels; l++)
      solid_mesh.UniformRefinement();

   if (Mpi::Root())
      mfem::out << "done." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Create solver
   ///////////////////////////////////////////////////////////////////////////////////////////////
   if (Mpi::Root())
      mfem::out << "\nCreating Elasticity solver..." << std::endl;

   elasticity::ElasticitySolver elasticity(
      Elasticity_ctx.solver_type, &solid_mesh,
      Elasticity_ctx.order, Elasticity_ctx.density,
      Elasticity_ctx.lambda, Elasticity_ctx.mu);

   auto fes = elasticity.GetFESpace();

   // Solution state vectors and grid functions
   Vector solid_displacement, solid_vel;
   ParGridFunction solid_displacement_gf(fes); solid_displacement_gf = 0.0;
   ParGridFunction solid_vel_gf(fes);          solid_vel_gf          = 0.0;

   // -----------------------------------------------------------------------
   // Grid functions for the PrescribedVelocity BC
   //
   //   prescribed_vel_gf : MMS velocity at t_{n+1} — updated before each Step
   //   prev_vel_gf       : velocity at t_n         — updated after each Step
   //   prev_acc_gf       : acceleration at t_n     — updated after each Step
   //
   // DerivedAccelerationCoefficient (inside ElasticityOperator) reads these
   // three GridFunctions to compute the consistent boundary acceleration.
   // -----------------------------------------------------------------------
   ParGridFunction prescribed_vel_gf(fes); prescribed_vel_gf = 0.0;
   ParGridFunction prev_vel_gf(fes);        prev_vel_gf       = 0.0;
   ParGridFunction prev_acc_gf(fes);        prev_acc_gf       = 0.0;

   // Exact solution GFs for output and error computation
   ParGridFunction exact_displacement_gf(fes); exact_displacement_gf = 0.0;
   ParGridFunction exact_vel_gf(fes);          exact_vel_gf          = 0.0;

   // Coefficients
   VectorFunctionCoefficient disp_exact_coeff(2, displacement_mms);
   VectorFunctionCoefficient vel_exact_coeff(2, velocity_mms);

   if (Mpi::Root())
      mfem::out << "done." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Set boundary conditions and setup
   ///////////////////////////////////////////////////////////////////////////////////////////////
   if (Mpi::Root())
      mfem::out << "\nSetting up boundary conditions..." << std::endl;

   Array<int> domain_marker(solid_mesh.attributes.Max()); domain_marker = 1;
   Array<int> solid_bdr(solid_mesh.bdr_attributes.Max()); solid_bdr = 1;

   // PrescribedVelocity BC on all boundaries:
   //   prescribed_vel_gf is set to the MMS velocity at t+dt before each Step call.
   //   prev_vel_gf / prev_acc_gf are updated after each Step.
   elasticity.AddPrescribedVelocity(
      &prescribed_vel_gf, &prev_vel_gf, &prev_acc_gf, solid_bdr);
   elasticity.AddBodyForce(forcing_mms, domain_marker);
   elasticity.Setup();

   if (Mpi::Root())
      mfem::out << "done." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Set up ParaView output
   ///////////////////////////////////////////////////////////////////////////////////////////////
   ParaViewDataCollection pvdc("Solid", &solid_mesh);
   pvdc.SetPrefixPath(outfolder);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   pvdc.SetHighOrderOutput(true);
   pvdc.SetLevelsOfDetail(Elasticity_ctx.order);
   pvdc.RegisterField("displacement",        &solid_displacement_gf);
   pvdc.RegisterField("velocity",            &solid_vel_gf);
   pvdc.RegisterField("prescribed_velocity", &prescribed_vel_gf);
   pvdc.RegisterField("prev_velocity",       &prev_vel_gf);
   pvdc.RegisterField("prev_acceleration",   &prev_acc_gf);
   pvdc.RegisterField("exact_displacement",  &exact_displacement_gf);
   pvdc.RegisterField("exact_velocity",      &exact_vel_gf);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Initial conditions (t = 0)
   ///////////////////////////////////////////////////////////////////////////////////////////////
   disp_exact_coeff.SetTime(0.0);
   vel_exact_coeff.SetTime(0.0);

   solid_displacement_gf.ProjectCoefficient(disp_exact_coeff);
   solid_vel_gf.ProjectCoefficient(vel_exact_coeff);
   solid_displacement_gf.GetTrueDofs(solid_displacement);
   solid_vel_gf.GetTrueDofs(solid_vel);

   ///<--- Initialize GridFunctions
   //   PrescribedVelocity BC:
   //   prev_vel_gf = MMS velocity at t=0
   //   prev_acc_gf = 0  (acceleration is zero at t=0 since sin(0) = 0)
   prev_vel_gf.ProjectCoefficient(vel_exact_coeff);
   prev_acc_gf = 0.0;

   exact_displacement_gf.ProjectCoefficient(disp_exact_coeff);
   exact_vel_gf.ProjectCoefficient(vel_exact_coeff);

   pvdc.SetCycle(0);
   pvdc.SetTime(0.0);
   pvdc.Save();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Time integration loop
   ///////////////////////////////////////////////////////////////////////////////////////////////
   if (Mpi::Root())
   {
      mfem::out << "-------------------------------------------------------" << std::endl;
      mfem::out << std::left
          << std::setw(8)  << "Step"
          << std::setw(16) << "Time"
          << std::setw(16) << "dt"
          << std::setw(16) << "L^2 error (u)"
          << std::endl;
      mfem::out << "-------------------------------------------------------" << std::endl;
   }

   bool last_step = false;
   for (int step = 0; !last_step; ++step)
   {
      real_t dt = Problem_ctx.dt;

      if (current_time + dt >= Problem_ctx.t_final - dt / 2)
         last_step = true;

      // 1. Update prescribed_vel_gf to the MMS velocity at t_{n+1} = current_time + dt.
      //    DerivedAccelerationCoefficient reads this during ImplicitSolve to compute
      //    the consistent boundary acceleration.
      vel_exact_coeff.SetTime(current_time + dt);
      prescribed_vel_gf.ProjectCoefficient(vel_exact_coeff);

      // 2. Advance the solver
      solid_displacement_gf.GetTrueDofs(solid_displacement);
      solid_vel_gf.GetTrueDofs(solid_vel);

      elasticity.Step(solid_displacement, solid_vel, current_time, dt);

      solid_displacement_gf.SetFromTrueDofs(solid_displacement);
      solid_vel_gf.SetFromTrueDofs(solid_vel);

      // 3. Update prev_vel_gf and prev_acc_gf for the next step's
      //    DerivedAccelerationCoefficient.
      prev_vel_gf.SetFromTrueDofs(solid_vel);
      {
         Vector *acc_state = elasticity.GetLatestState();
         if (acc_state != nullptr)
            prev_acc_gf.SetFromTrueDofs(*acc_state);
      }

      //current_time += dt;

      // Compute and print L^2 displacement error
      disp_exact_coeff.SetTime(current_time);
      const real_t l2_err = solid_displacement_gf.ComputeL2Error(disp_exact_coeff);

      if (Mpi::Root())
      {
         mfem::out << std::left
             << std::setw(8)  << step
             << std::setw(16) << std::scientific << std::setprecision(8) << current_time
             << std::setw(16) << std::scientific << std::setprecision(8) << dt
             << std::setw(16) << std::scientific << std::setprecision(8) << l2_err
             << std::endl;
      }

      // Update exact GFs and save
      vel_exact_coeff.SetTime(current_time);
      exact_displacement_gf.ProjectCoefficient(disp_exact_coeff);
      exact_vel_gf.ProjectCoefficient(vel_exact_coeff);

      if (step % save_freq == 0 || last_step)
      {
         pvdc.SetCycle(step + 1);
         pvdc.SetTime(current_time);
         pvdc.Save();
      }
   }

   return 0;
}

// =====================================================================================
// MMS functions
// =====================================================================================

void displacement_mms(const Vector &x, real_t t, Vector &v)
{
   const real_t freq = Elasticity_ctx.freq;
   v = 0.0;
   v[0] = cos(2*M_PI*x[1]*freq) * sin(2*M_PI*x[0]*freq) * sin(t);
   v[1] = cos(2*M_PI*x[0]*freq) * sin(2*M_PI*x[1]*freq) * sin(t);
}

void velocity_mms(const Vector &x, real_t t, Vector &v)
{
   const real_t freq = Elasticity_ctx.freq;
   v = 0.0;
   v[0] = cos(2*M_PI*x[1]*freq) * sin(2*M_PI*x[0]*freq) * cos(t);
   v[1] = cos(2*M_PI*x[0]*freq) * sin(2*M_PI*x[1]*freq) * cos(t);
}

void forcing_mms(const Vector &x, real_t t, Vector &v)
{
   const real_t freq    = Elasticity_ctx.freq;
   const real_t freq_sq = freq * freq;
   const real_t pi_sq   = M_PI * M_PI;
   const real_t spatial_factor_lambda = 8.0  * freq_sq * pi_sq * Elasticity_ctx.lambda;
   const real_t spatial_factor_mu     = 16.0 * freq_sq * pi_sq * Elasticity_ctx.mu;
   v = 0.0;
   v[0] = (spatial_factor_lambda + spatial_factor_mu - Elasticity_ctx.density)
          * cos(2*M_PI*x[1]*freq) * sin(2*M_PI*x[0]*freq) * sin(t);
   v[1] = (spatial_factor_lambda + spatial_factor_mu - Elasticity_ctx.density)
          * cos(2*M_PI*x[0]*freq) * sin(2*M_PI*x[1]*freq) * sin(t);
}
