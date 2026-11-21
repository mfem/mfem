// =====================================================================================
//          Test Solid Robin BC with Manufactured Solutions
// -------------------------------------------------------------------------------------
// Sample runs:
//
//     mpirun -np 4 ./test-csm-robin-mms -rs 2 -eo 2 -dt 1e-3 -tf 1.0 -st 15
//
// Manufactured solution on [0,1]^2:
//
//   u(x,y,t) = [ x * sin(omega*t),  0 ]
//
// Strain (symmetric gradient):
//   eps_11 = sin(omega*t),  eps_22 = 0,  eps_12 = 0
//
// Stress (linear elasticity):
//   sigma_11 = (lambda + 2*mu) * sin(omega*t)
//   sigma_22 = lambda * sin(omega*t)
//   sigma_12 = 0
//
// Since div(sigma) = 0, the body force is purely inertial:
//   f = [ -rho * omega^2 * x * sin(omega*t),  0 ]
//
// Boundary conditions:
//   Left  (x=0, attr=4):  Dirichlet   u = [0, 0]
//   Right (x=1, attr=2):  Robin       alpha * du/dt + sigma*n = g
//                           n = [1,0]  =>  sigma*n = [(lambda+2*mu)*sin(omega*t), 0]
//                           g = [ alpha*omega*cos(omega*t) + (lambda+2*mu)*sin(omega*t),  0 ]
//   Bottom (y=0, attr=1): Neumann     sigma*n = [0, -lambda*sin(omega*t)]    (n=[0,-1])
//   Top    (y=1, attr=3): Neumann     sigma*n = [0,  lambda*sin(omega*t)]    (n=[0, 1])
//
// Initial conditions:
//   u(x,y,0) = [0, 0]
//   du/dt(x,y,0) = [x * omega, 0]
// =====================================================================================
//
// Sample run:
//    mpirun -np 4 ./test-csm-robin-mms -rs 2 -eo 2 -dt 1e-3 -tf 1.0 -st 15
//

#include "structure/elasticity_solver.hpp"
#include "fsi_utils.hpp"

#include <fstream>
#include <iomanip>

using namespace mfem;

struct s_ElasticityContext
{
   int    order       = 2;
   real_t density     = 1e3;
   real_t nu          = 0.4;
   real_t mu          = 2.0e6;
   real_t lambda;           // computed from nu and mu
   int    solver_type = 15; // HHTAlphaSolver
   real_t omega       = 1.0; // temporal frequency
   real_t alpha       = 1.0; // Robin coefficient
} Elasticity_ctx;

struct s_MeshContext
{
   bool hex                  = true;
   int  n                    = 8;
   int  serial_ref_levels    = 0;
   int  parallel_ref_levels  = 0;
} Mesh_ctx;

struct s_ProblemContext
{
   real_t t_final = 1.0;
   real_t dt      = 1e-3;
} Problem_ctx;

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
void disp_mms     (const Vector &x, real_t t, Vector &v);
void vel_mms      (const Vector &x, real_t t, Vector &v);
void acc_mms      (const Vector &x, real_t t, Vector &v);
void forcing_mms  (const Vector &x, real_t t, Vector &v);
void robin_g_mms  (const Vector &x, real_t t, Vector &v);
void traction_top_mms   (const Vector &x, real_t t, Vector &v);
void traction_bot_mms   (const Vector &x, real_t t, Vector &v);
void disp_left_mms(const Vector &x, real_t t, Vector &v);
void vel_left_mms (const Vector &x, real_t t, Vector &v);
void acc_left_mms (const Vector &x, real_t t, Vector &v);

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
   const char *outfolder   = "./Output/CSM-Robin-MMS";
   const char *device_config = "cpu";
   int save_freq = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string.");
   args.AddOption(&Elasticity_ctx.order, "-eo", "--elasticity-order",
                  "Polynomial order for elasticity space.");
   args.AddOption(&Elasticity_ctx.mu, "-sh", "--shear-modulus",
                  "Shear modulus mu.");
   args.AddOption(&Elasticity_ctx.nu, "-nu", "--poisson-ratio",
                  "Poisson ratio nu.");
   args.AddOption(&Elasticity_ctx.density, "-rho", "--density",
                  "Solid density.");
   args.AddOption(&Elasticity_ctx.solver_type, "-st", "--solver-type",
                  "SecondOrderODESolver type.");
   args.AddOption(&Elasticity_ctx.omega, "-w", "--omega",
                  "Temporal frequency of manufactured solution.");
   args.AddOption(&Elasticity_ctx.alpha, "-a", "--alpha",
                  "Robin coefficient alpha.");
   args.AddOption(&Problem_ctx.dt, "-dt", "--dt", "Time step.");
   args.AddOption(&Problem_ctx.t_final, "-tf", "--t-final", "Final time.");
   args.AddOption(&Mesh_ctx.serial_ref_levels, "-rs", "--rs",
                  "Serial refinement levels.");
   args.AddOption(&Mesh_ctx.parallel_ref_levels, "-rp", "--rp",
                  "Parallel refinement levels.");
   args.AddOption(&outfolder, "-of", "--output-folder", "Output folder.");
   args.AddOption(&save_freq, "-sf", "--save-freq", "ParaView save frequency.");
   args.ParseCheck();

   Elasticity_ctx.lambda =
      2.0 * Elasticity_ctx.mu * Elasticity_ctx.nu / (1.0 - 2.0 * Elasticity_ctx.nu);

   if (Mpi::Root())
   {
      mfem::out << "Robin MMS test\n"
                << "  lambda = " << Elasticity_ctx.lambda << "\n"
                << "  mu     = " << Elasticity_ctx.mu     << "\n"
                << "  rho    = " << Elasticity_ctx.density << "\n"
                << "  omega  = " << Elasticity_ctx.omega  << "\n"
                << "  alpha  = " << Elasticity_ctx.alpha  << "\n";
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Build mesh on [0,1]^2
   ///////////////////////////////////////////////////////////////////////////////////////////////
   // MakeCartesian2D boundary attributes on [0,1]^2 (generate_edges=true):
   //   attr 1 = bottom (y=0),  attr 2 = right (x=1),
   //   attr 3 = top    (y=1),  attr 4 = left  (x=0)
   Element::Type etype = Mesh_ctx.hex ? Element::QUADRILATERAL : Element::TRIANGLE;
   Mesh mesh = Mesh::MakeCartesian2D(Mesh_ctx.n, Mesh_ctx.n, etype, true);

   for (int l = 0; l < Mesh_ctx.serial_ref_levels; l++)
      mesh.UniformRefinement();

   ParMesh solid_mesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   for (int l = 0; l < Mesh_ctx.parallel_ref_levels; l++)
      solid_mesh.UniformRefinement();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Create solver
   ///////////////////////////////////////////////////////////////////////////////////////////////
   elasticity::ElasticitySolver elasticity(
      Elasticity_ctx.solver_type, &solid_mesh,
      Elasticity_ctx.order,
      Elasticity_ctx.density,
      Elasticity_ctx.lambda,
      Elasticity_ctx.mu);

   auto *fes = elasticity.GetFESpace();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Set boundary and volume conditions
   ///////////////////////////////////////////////////////////////////////////////////////////////

   // --- Domain (all elements) ---
   Array<int> domain_marker(solid_mesh.attributes.Max());
   domain_marker = 1;

   // --- Boundary attribute markers ---
   const int max_bdr = solid_mesh.bdr_attributes.Max(); // should be 4

   // attr 4: left face -- Dirichlet (u = 0 for all time)
   Array<int> left_bdr(max_bdr);   left_bdr = 0;   left_bdr[3] = 1; // attr 4
   // attr 2: right face -- Robin
   Array<int> robin_bdr(max_bdr);  robin_bdr = 0;  robin_bdr[1] = 1; // attr 2
   // attr 1: bottom face -- Neumann (traction)
   Array<int> bot_bdr(max_bdr);    bot_bdr = 0;    bot_bdr[0] = 1;   // attr 1
   // attr 3: top face -- Neumann (traction)
   Array<int> top_bdr(max_bdr);    top_bdr = 0;    top_bdr[2] = 1;   // attr 3

   // Dirichlet on left
   elasticity.AddPrescribedDisplacement(disp_left_mms, vel_left_mms, acc_left_mms, left_bdr);

   // Robin on right:  alpha * du/dt + sigma*n = g
   VectorFunctionCoefficient robin_g(2, robin_g_mms);
   elasticity.AddRobinBC(Elasticity_ctx.alpha, &robin_g, robin_bdr, false /*not owned*/);

   // Neumann on top and bottom (manufactured traction)
   elasticity.AddBoundaryLoad(traction_bot_mms, bot_bdr);
   elasticity.AddBoundaryLoad(traction_top_mms, top_bdr);

   // Body force
   elasticity.AddBodyForce(forcing_mms, domain_marker);

   elasticity.Setup();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Grid functions and initial conditions
   ///////////////////////////////////////////////////////////////////////////////////////////////
   ParGridFunction u_gf(fes), v_gf(fes);
   u_gf = 0.0;
   v_gf = 0.0;

   // Project non-trivial initial velocity: du/dt(x,y,0) = [x*omega, 0]
   VectorFunctionCoefficient v_exact_coef(2, vel_mms);
   v_exact_coef.SetTime(0.0);
   v_gf.ProjectCoefficient(v_exact_coef);

   Vector u_tdof, v_tdof;
   u_gf.GetTrueDofs(u_tdof);
   v_gf.GetTrueDofs(v_tdof);

   // Exact solution GFs for output
   ParGridFunction u_exact_gf(fes), v_exact_gf(fes);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7. Output setup
   ///////////////////////////////////////////////////////////////////////////////////////////////
   int step = 0;
   ParaViewDataCollection pvdc("Solid-Robin-MMS", &solid_mesh);
   pvdc.SetPrefixPath(outfolder);
   pvdc.SetDataFormat(VTKFormat::BINARY);
   pvdc.SetHighOrderOutput(true);
   pvdc.SetLevelsOfDetail(Elasticity_ctx.order);
   pvdc.RegisterField("displacement", &u_gf);
   pvdc.RegisterField("velocity",     &v_gf);
   pvdc.RegisterField("displacement_exact", &u_exact_gf);
   pvdc.RegisterField("velocity_exact",     &v_exact_gf);
   pvdc.SetCycle(0);
   pvdc.SetTime(0.0);
   pvdc.Save();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 8. Time loop
   ///////////////////////////////////////////////////////////////////////////////////////////////
   if (Mpi::Root())
   {
      mfem::out << "\n"
                << std::left
                << std::setw(8)  << "Step"
                << std::setw(16) << "Time"
                << std::setw(16) << "dt"
                << std::setw(20) << "L2 err (u)"
                << std::setw(20) << "L2 err (v)"
                << "\n"
                << std::string(80, '-') << "\n";
   }

   VectorFunctionCoefficient u_exact_coef(2, disp_mms);
   bool last_step = false;

   for (step = 0; !last_step; ++step)
   {
      if (current_time + Problem_ctx.dt >= Problem_ctx.t_final - Problem_ctx.dt / 2)
         last_step = true;

      u_gf.GetTrueDofs(u_tdof);
      v_gf.GetTrueDofs(v_tdof);

      elasticity.Step(u_tdof, v_tdof, current_time, Problem_ctx.dt);

      u_gf.SetFromTrueDofs(u_tdof);
      v_gf.SetFromTrueDofs(v_tdof);

      // Error w.r.t. exact solution at current_time (after step)
      u_exact_coef.SetTime(current_time);
      v_exact_coef.SetTime(current_time);
      real_t l2_err_u = u_gf.ComputeL2Error(u_exact_coef);
      real_t l2_err_v = v_gf.ComputeL2Error(v_exact_coef);

      if (Mpi::Root())
      {
         mfem::out << std::left
                   << std::setw(8)  << step
                   << std::setw(16) << std::scientific << std::setprecision(6) << current_time
                   << std::setw(16) << std::scientific << std::setprecision(6) << Problem_ctx.dt
                   << std::setw(20) << std::scientific << std::setprecision(6) << l2_err_u
                   << std::setw(20) << std::scientific << std::setprecision(6) << l2_err_v
                   << "\n";
      }

      if (step % save_freq == 0 || last_step)
      {
         u_exact_gf.ProjectCoefficient(u_exact_coef);
         v_exact_gf.ProjectCoefficient(v_exact_coef);
         pvdc.SetCycle(step + 1);
         pvdc.SetTime(current_time);
         pvdc.Save();
      }
   }

   return 0;
}

// ---------------------------------------------------------------------------
// Exact solution:  u = [x * sin(omega*t),  0]
// ---------------------------------------------------------------------------
void disp_mms(const Vector &x, real_t t, Vector &v)
{
   v    = 0.0;
   v[0] = x[0] * std::sin(Elasticity_ctx.omega * t);
}

void vel_mms(const Vector &x, real_t t, Vector &v)
{
   v    = 0.0;
   v[0] = x[0] * Elasticity_ctx.omega * std::cos(Elasticity_ctx.omega * t);
}

void acc_mms(const Vector &x, real_t t, Vector &v)
{
   v    = 0.0;
   v[0] = -x[0] * Elasticity_ctx.omega * Elasticity_ctx.omega *
          std::sin(Elasticity_ctx.omega * t);
}

// ---------------------------------------------------------------------------
// Left face (x=0): Dirichlet u = 0 for all time
// ---------------------------------------------------------------------------
void disp_left_mms(const Vector & /*x*/, real_t /*t*/, Vector &v) { v = 0.0; }
void vel_left_mms (const Vector & /*x*/, real_t /*t*/, Vector &v) { v = 0.0; }
void acc_left_mms (const Vector & /*x*/, real_t /*t*/, Vector &v) { v = 0.0; }

// ---------------------------------------------------------------------------
// Body force:  f = rho * d2u/dt2 - div(sigma)
//   div(sigma) = 0  (no spatial variation in stress)
//   => f = [-rho * omega^2 * x * sin(omega*t),  0]
// ---------------------------------------------------------------------------
void forcing_mms(const Vector &x, real_t t, Vector &v)
{
   v    = 0.0;
   v[0] = -Elasticity_ctx.density * Elasticity_ctx.omega * Elasticity_ctx.omega *
          x[0] * std::sin(Elasticity_ctx.omega * t);
}

// ---------------------------------------------------------------------------
// Robin RHS on right face (x=1, n=[1,0]):
//   g = alpha * du/dt + sigma*n
//     = alpha * omega * cos(omega*t) * [1,0]    (at x=1, du/dt|x=1 = omega*cos*[1,0])
//       + [(lambda+2*mu)*sin(omega*t), 0]
// Note: this function will be called for all pts on the right boundary,
// but since the manufactured solution is linear in x, the Robin g only
// depends on t (not on x or y).
// ---------------------------------------------------------------------------
void robin_g_mms(const Vector & /*x*/, real_t t, Vector &v)
{
   v    = 0.0;
   real_t st = std::sin(Elasticity_ctx.omega * t);
   real_t ct = std::cos(Elasticity_ctx.omega * t);
   // traction component from the exact stress at x=1 (same for all x on this face)
   v[0] = Elasticity_ctx.alpha * Elasticity_ctx.omega * ct        // alpha * du/dt component
        + (Elasticity_ctx.lambda + 2.0 * Elasticity_ctx.mu) * st; // sigma*n component
}

// ---------------------------------------------------------------------------
// Neumann traction on bottom face (y=0, n=[0,-1]):
//   sigma*n = [sigma_12, sigma_22] * [0,-1]^T applied via boundary integral
//   sigma_12 = 0,  sigma_22 = lambda * sin(omega*t)
//   => traction vector = [0, -lambda*sin(omega*t)]
// ---------------------------------------------------------------------------
void traction_bot_mms(const Vector & /*x*/, real_t t, Vector &v)
{
   v    = 0.0;
   v[1] = -Elasticity_ctx.lambda * std::sin(Elasticity_ctx.omega * t);
}

// ---------------------------------------------------------------------------
// Neumann traction on top face (y=1, n=[0,1]):
//   traction vector = [0, lambda*sin(omega*t)]
// ---------------------------------------------------------------------------
void traction_top_mms(const Vector & /*x*/, real_t t, Vector &v)
{
   v    = 0.0;
   v[1] = Elasticity_ctx.lambda * std::sin(Elasticity_ctx.omega * t);
}
