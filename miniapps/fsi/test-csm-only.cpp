// =====================================================================================
//          Test Solid
// -------------------------------------------------------------------------------------
// Sample runs:
//
//     mpirun -np 4 ./test-csm-only -p 0 -dt 1e-3 -tp 1.0 -tr 2.0 -tf 3.0 -t 2.5e5 -eo 3 -of ./Output/TestCSM
//
// The flag p controls the problem to solve:
//     0 - Transverse test
//     1 - Compression test
//     2 - Transverse test (fixed y displacement on the top)
//     3 - Compression test (fixed x displacement on the lateral sides)
//     4 - Transverse test (periodic)
//     5 - Compression test (periodic)
//     6 - Transverse test (prescribed displacement on the top)
//    7 - Compression test (prescribed displacement on the top)
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

// Solid mechanics module
#include "structure/elasticity_solver.hpp"

// Mesh utilities
#include "fsi_utils.hpp"

// Output
#include <fstream>

using namespace mfem;
using namespace common_fsi_utils;

struct s_ElasticityContext // elasticity params
{
   int order = 2;        // polynomial order of the elasticity space
   real_t density = 1e3; // density of the solid
   real_t nu = 0.4;      // Poisson's ratio
   real_t mu = 2.0e6;    // Shear modulus/rigidity
   real_t lambda;
} Elasticity_ctx;

struct s_MeshContext // mesh
{
   real_t L = 1.0; // length of the mesh
   real_t H = 1.0; // height of the mesh
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
} Mesh_ctx;

struct s_ProblemContext // problem context
{
   real_t t_final = 5.0;
   real_t dt = 1e-3;          // time step
   real_t release_time = 2.0; // time at which the solid is released
   real_t preloadT = 1.0;
   real_t traction = 1e3; // traction force applied to the solid
   real_t Dmax = 0.1;     // maximum displacement
} Problem_ctx;

// Forward declarations
void traction_fun(const Vector &x, double time, Vector &v);
void compression_fun(const Vector &x, double time, Vector &v);

void prescribed_displacement_transverse_fun(const Vector &x, double time, Vector &v);
void velocity_transverse_fun(const Vector &x, double time, Vector &v);
void acceleration_transverse_fun(const Vector &x, double time, Vector &v);

void prescribed_displacement_compression_fun(const Vector &x, double time, Vector &v);
void velocity_compression_fun(const Vector &x, double time, Vector &v); 
void acceleration_compression_fun(const Vector &x, double time, Vector &v);

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

   // Problem:
   // 0 - Transverse test
   // 1 - Compression test
   // 2 - Transverse test (fixed y displacement on the top)
   // 3 - Compression test (fixed x displacement on the lateral sides)
   // 4 - Transverse test (periodic)
   // 5 - Compression test (periodic)
   // 6 - Transverse test (prescribed displacement on the top)
   // 7 - Compression test (prescribed displacement on the top)
   int problem = 0; 

   // Mesh file
   const char *mesh_file = "../../data/multidomain2d-quad.e";

   // Timestepping parameters
   real_t time = 0.0;

   // Output parameters
   const char *outfolder = "./Output/FSI";
   int save_freq = 10;
   bool compute_output = true;

   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&Elasticity_ctx.order, "-eo", "--elasticity-order",
                  "polynomial_order for solid elasticity space");
   args.AddOption(&Elasticity_ctx.mu, "-sh", "--shear_modulus",
                  "Shear modulus/rigidity");

   args.AddOption(&problem, "-p", "--problem",
                  "Problem to solve (0 - Transverse test, 1 - Compression test).");

   args.AddOption(&compute_output, "-co", "--compute-output", "-no-co", "--no-compute-output",
                  "Compute output files (default: true).");

   args.AddOption(&Problem_ctx.dt, "-dt", "--dt",
                  "dt.");
   args.AddOption(&Problem_ctx.t_final, "-tf", "--t-final",
                  "t_final.");
   args.AddOption(&Problem_ctx.release_time, "-tr", "--t-release",
                  "release_time.");
   args.AddOption(&Problem_ctx.preloadT, "-tp", "--preloadT",
                  "preloadT.");

   args.AddOption(&Problem_ctx.traction, "-t", "--traction",
                  "traction force applied to the solid.");

   args.AddOption(&Mesh_ctx.serial_ref_levels, "-rs", "--rs", "serial_ref_levels");
   args.AddOption(&Mesh_ctx.parallel_ref_levels, "-rp", "--rp", "parallel_ref_levels");

   args.AddOption(&outfolder, "-of", "--output-folder",
                  "Output folder.");
   args.AddOption(&save_freq, "-sf", "--save-freq",
                  "Save frequency for output files.");
   args.ParseCheck();

   MFEM_VERIFY(problem >= 0 && problem <= 7,
               "Problem must be between 0 and 7.");

   // Make sure that  preloadT < release_time < t_final
   MFEM_VERIFY(Problem_ctx.preloadT < Problem_ctx.release_time &&
                   Problem_ctx.release_time < Problem_ctx.t_final,
               "preloadT must be less than release_time, which must be less than t_final.");

   Elasticity_ctx.lambda = 2 * Elasticity_ctx.mu * Elasticity_ctx.nu / (1 - 2 * Elasticity_ctx.nu);

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 3. Create serial Mesh, parallel and fluid/solid sub-meshes
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\nLoading mesh and refining in serial...";

   Mesh *serial_mesh = new Mesh(mesh_file);
   const int dim = serial_mesh->Dimension();

   for (int i = 0; i < Mesh_ctx.serial_ref_levels; i++)
   {
      serial_mesh->UniformRefinement();
   }

   // serial_mesh->SetCurvature(Mesh_ctx.curvature_order, false, dim, Ordering::byNODES); // CHECK

   if (Mpi::Root())
   {
      mfem::out << "done." << std::endl;
      mfem::out << "Creating sub-meshes...";
   }
   // Create the sub-domains for the solid domains
   Array<int> solid_domain_attributes(1);
   solid_domain_attributes[0] = 1;
   auto parent_mesh = SubMesh::CreateFromDomain(*serial_mesh, solid_domain_attributes);
   parent_mesh.SetAttributes();
   parent_mesh.EnsureNodes();

   Mesh *submesh;
   if (problem >= 4 && problem <= 6) // Periodic problems
   {
      // Make the mesh periodic in x direction
      std::vector<Vector> translations = {Vector({Mesh_ctx.L, 0.0})};

      submesh = new Mesh(Mesh::MakePeriodic(
          parent_mesh,
          parent_mesh.CreatePeriodicVertexMapping(translations)));
   }
   else
   {
      submesh = &parent_mesh;
   }

   // Generate mesh partitioning and create a parallel mesh.
   if (Mpi::Root())
      mfem::out << "Generating partitioning and creating parallel mesh... ";

   // Partition type:
   // 0) METIS_PartGraphRecursive (sorted neighbor lists)
   // 1) METIS_PartGraphKway      (sorted neighbor lists) (default)
   // 2) METIS_PartGraphVKway     (sorted neighbor lists)
   // 3) METIS_PartGraphRecursive
   // 4) METIS_PartGraphKway
   // 5) METIS_PartGraphVKway
   int partition_type = 1;
   int np = num_procs;
   int *partitioning = serial_mesh->GeneratePartitioning(np, partition_type);

   // Create parallel mesh
   ParMesh solid_mesh = ParMesh(MPI_COMM_WORLD, *submesh, partitioning, partition_type);
   ExportMeshwithPartitioning(outfolder, *submesh, partitioning);
   delete[] partitioning;

   for (int l = 0; l < Mesh_ctx.parallel_ref_levels; l++)
   {
      solid_mesh.UniformRefinement();
   }

   solid_mesh.SetAttributes();
   solid_mesh.EnsureNodes();


   if (Mpi::Root())
      mfem::out << "done." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 4. Create Solid Solver
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\nCreating Elasticity solver..." << std::endl;

   // Create the elasticity solver
   int solver_type  = 15; // HHTAlphaSolver
   elasticity::ElasticitySolver elasticity(solver_type, &solid_mesh, Elasticity_ctx.order, Elasticity_ctx.density, Elasticity_ctx.lambda, Elasticity_ctx.mu);

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

   if (Mpi::Root())
      mfem::out << "done." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 5. Set BCs
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
      mfem::out << "\nSetting up boundary conditions..." << std::endl;

   // Markers
   int solid_bottom = 1;
   int solid_lateral = 2;
   Array<int> solid_top(solid_mesh.bdr_attributes.Max());
   solid_top = 0;
   solid_top[2] = 1;
   solid_top[3] = 1;

   // Define traction function
   ParGridFunction solid_traction_gf(elasticity_fes);

   VectorCoefficient *traction_coeff = nullptr;

   switch (problem)
   {
   case 0: // Transverse test
   case 4: // Transverse test (periodic)
   {
      elasticity.AddFixedConstraint(solid_bottom);
      elasticity.AddBoundaryLoad(traction_fun, solid_top);
      traction_coeff = new VectorFunctionCoefficient(dim, traction_fun);
   }
   break;
   case 1: // Compression test
   case 5: // Compression test (periodic)
   {
      elasticity.AddFixedConstraint(solid_bottom);
      elasticity.AddBoundaryLoad(compression_fun, solid_top);
      traction_coeff = new VectorFunctionCoefficient(dim, compression_fun);
   }
   break;
   case 2: // Transverse test (fixed y displacement on the top)
   {
      elasticity.AddFixedConstraint(solid_bottom);
      elasticity.AddFixedConstraint(solid_top, 1);
      elasticity.AddBoundaryLoad(traction_fun, solid_top);
      traction_coeff = new VectorFunctionCoefficient(dim, traction_fun);
   }
   break;
   case 3: // Compression test (fixed x displacement on the lateral sides)
   {
      elasticity.AddFixedConstraint(solid_bottom);
      elasticity.AddFixedConstraint(solid_lateral, 0); // Fix displacement in x-direction
      elasticity.AddBoundaryLoad(compression_fun, solid_top);
      traction_coeff = new VectorFunctionCoefficient(dim, compression_fun);
   }
   break;
   case 6:
   {
      elasticity.AddFixedConstraint(solid_bottom);
      elasticity.AddPrescribedDisplacement(prescribed_displacement_transverse_fun, velocity_transverse_fun, acceleration_transverse_fun, solid_top);
      Problem_ctx.preloadT = Problem_ctx.t_final; // No preload, as we cannot turn off BCs for now
      Vector zero(dim); zero = 0.0;
      traction_coeff = new VectorConstantCoefficient(zero);
   }
   break;
   case 7:
   {
      elasticity.AddFixedConstraint(solid_bottom);
      elasticity.AddPrescribedDisplacement(prescribed_displacement_compression_fun, velocity_compression_fun, acceleration_compression_fun, solid_top);
      Problem_ctx.preloadT = Problem_ctx.t_final; // No preload, as we cannot turn off BCs for now
      Vector zero(dim); zero = 0.0;
      traction_coeff = new VectorConstantCoefficient(zero);
   }
   break;
   default:
      MFEM_ABORT("Unknown problem type.");
   }

   // After bcs have been set we can set up the elasticity operator
   elasticity.Setup();

   if (Mpi::Root())
      mfem::out << "done." << std::endl;

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 6. Set up output
   ///////////////////////////////////////////////////////////////////////////////////////////////

   int step = 0;

   ParaViewDataCollection pvdc_solid("Solid", &solid_mesh);
   pvdc_solid.SetPrefixPath(outfolder);
   pvdc_solid.SetDataFormat(VTKFormat::BINARY32);
   pvdc_solid.SetHighOrderOutput(true);
   pvdc_solid.SetLevelsOfDetail(Elasticity_ctx.order);
   pvdc_solid.SetCycle(step);
   pvdc_solid.SetTime(0.0);
   pvdc_solid.RegisterField("velocity", &solid_vel_gf);
   pvdc_solid.RegisterField("displacement", &solid_displacement_gf);
   pvdc_solid.Save();

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 7.  Solve problem
   ///////////////////////////////////////////////////////////////////////////////////////////////

   if (Mpi::Root())
   {
      out << "-------------------------------------------------------------------------------------------------------------" << std::endl;
      out << std::left
          << std::setw(8) << "Step"
          << std::setw(16) << "Time"
          << std::setw(16) << "dt"
          << std::setw(16) << "disp_x"
          << std::setw(16) << "disp_y"
          << std::setw(16) << "vel_x"
          << std::setw(16) << "vel_y"
          << std::setw(16) << "traction_x"
          << std::setw(16) << "traction_y"
          << std::endl;
      out << "-------------------------------------------------------------------------------------------------------------" << std::endl;
   }

   bool last_step = false;
   for (int step = 0; !last_step; ++step) // TIME-INTEGRATION LOOP
   {
      if (time + Problem_ctx.dt >= Problem_ctx.t_final - Problem_ctx.dt / 2)
      {
         last_step = true;
      }

      // --------------------------------------------------------------------------------- //
      //// ----- Solve Elasticity -----
      // --------------------------------------------------------------------------------- //

      solid_displacement_gf.GetTrueDofs(solid_displacement);
      solid_vel_gf.GetTrueDofs(solid_vel);

      elasticity.Step(solid_displacement, solid_vel, time, Problem_ctx.dt);

      solid_displacement_gf.SetFromTrueDofs(solid_displacement);
      solid_vel_gf.SetFromTrueDofs(solid_vel);

      // --------------------------------------------------------------------------------- //
      //// ----- Output
      // --------------------------------------------------------------------------------- //

      if (Mpi::Root())
      {
         out << std::left
             << std::setw(8) << step
             << std::setw(16) << std::scientific << std::setprecision(8) << time
             << std::setw(16) << std::scientific << std::setprecision(8) << Problem_ctx.dt;
      }

      // Compute output values
      if (compute_output)
      {
         Array<IntegrationPoint> ips;
         Array<int> elem_ids;

         Vector point(2);
         point(0) = 0.0;
         point(1) = 0.0;
         DenseMatrix points(dim, 1);
         points.SetCol(0, point);

         Vector val;
         solid_mesh.FindPoints(points, elem_ids, ips);

         // if elem_ids[i] = -2 then the point was found on another rank
         // if elem_ids[i] = -1 then the point was not found on any rank

         real_t disp_x = 0.0, disp_y = 0.0;
         real_t vel_x = 0.0, vel_y = 0.0;
         real_t traction_x = 0.0, traction_y = 0.0;

         solid_traction_gf.ProjectCoefficient(*traction_coeff);

         // Only evaluate grid functions if the point was found on this rank
         if (elem_ids[0] >= 0)
         {
            solid_displacement_gf.GetVectorValue(elem_ids[0], ips[0], val);
            disp_x = val(0);
            disp_y = val(1);
            solid_vel_gf.GetVectorValue(elem_ids[0], ips[0], val);
            vel_x = val(0);
            vel_y = val(1);
            solid_traction_gf.GetVectorValue(elem_ids[0], ips[0], val);
            traction_x = val(0);
            traction_y = val(1);
         }

         // Reduce values across all ranks to get the result from whichever rank found the point
         real_t global_vals[6] = {disp_x, disp_y, vel_x, vel_y, traction_x, traction_y};
         real_t reduced_vals[6];
         MPI_Allreduce(global_vals, reduced_vals, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

         disp_x = reduced_vals[0];
         disp_y = reduced_vals[1];
         vel_x = reduced_vals[2];
         vel_y = reduced_vals[3];
         traction_x = reduced_vals[4];
         traction_y = reduced_vals[5];

         if (Mpi::Root())
         {
            out << std::setw(16) << std::scientific << std::setprecision(8) << disp_x
                << std::setw(16) << std::scientific << std::setprecision(8) << disp_y
                << std::setw(16) << std::scientific << std::setprecision(8) << vel_x
                << std::setw(16) << std::scientific << std::setprecision(8) << vel_y
                << std::setw(16) << std::scientific << std::setprecision(8) << traction_x
                << std::setw(16) << std::scientific << std::setprecision(8) << traction_y
                << std::endl;
         }
      }

      // Export to paraview
      if (step % save_freq == 0 || last_step)
      {
         pvdc_solid.SetCycle(step + 1);
         pvdc_solid.SetTime(time);
         pvdc_solid.Save();
      }
   }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   /// 9. Cleanup
   ///////////////////////////////////////////////////////////////////////////////////////////////

   delete traction_coeff;

   return 0;
}

// Constant traction until release time, then zero traction
// ramp the traction from 0 to Problem_ctx.t_preload
void traction_fun(const Vector &x, double time, Vector &v)
{
   const real_t preloadT = Problem_ctx.preloadT;
   const real_t preload = (time < preloadT) ? 0.5 * (1.0 - cos(M_PI * time / preloadT)) : 1.0;

   v = 0.0;
   v[0] = time < Problem_ctx.release_time ? Problem_ctx.traction * preload : 0.0; // traction in x-direction
   v[1] = 0.0;                                                                    // traction in y-direction
}

void compression_fun(const Vector &x, double time, Vector &v)
{
   const real_t preloadT = Problem_ctx.preloadT;
   const real_t preload = (time < preloadT) ? 0.5 * (1.0 - cos(M_PI * time / preloadT)) : 1.0;

   v = 0.0;
   v[0] = 0.0;                                                                    // no traction in x-direction
   v[1] = time < Problem_ctx.release_time ? -Problem_ctx.traction * preload : 0.0; // constant traction in y-direction
}

// Prescribed displacement test (compression)
// Parabolic (in time) displacement in x-direction until release time
// v = [ Dmax * ( t / tr )^2, 0.0 ]
void prescribed_displacement_transverse_fun(const Vector &x, double time, Vector &v)
{
   const real_t tr = Problem_ctx.release_time;

   v = 0.0;
   v[0] = Problem_ctx.Dmax * pow(time / tr, 2);
   v[0] = v[0] > Problem_ctx.Dmax ? Problem_ctx.Dmax : v[0]; // limit displacement to Dmax
   v[1] = 0.0; // no displacement in y-direction
}

void velocity_transverse_fun(const Vector &x, double time, Vector &v)
{
   const real_t tr = Problem_ctx.release_time;

   v = 0.0;
   v[0] = time < tr ? 2.0 * Problem_ctx.Dmax * time / (tr * tr) : 0.0; // velocity in x-direction
   v[1] = 0.0;                                       // no velocity in y-direction
}

void acceleration_transverse_fun(const Vector &x, double time, Vector &v)
{
   const real_t tr = Problem_ctx.release_time;

   v = 0.0;
   v[0] = time < tr ? 2.0 * Problem_ctx.Dmax / (tr * tr) : 0.0; // acceleration in x-direction
   v[1] = 0.0;                                // no acceleration in y-direction
}



void prescribed_displacement_compression_fun(const Vector &x, double time, Vector &v)
{
   const real_t tr = Problem_ctx.release_time;

   v = 0.0;
   v[0] = 0.0; // no displacement in x-direction
   v[1] = Problem_ctx.Dmax * pow(time / tr, 2);
   v[1] = v[1] > Problem_ctx.Dmax ? -Problem_ctx.Dmax : -v[1]; // limit displacement to Dmax
}

void velocity_compression_fun(const Vector &x, double time, Vector &v)
{
   const real_t tr = Problem_ctx.release_time;

   v = 0.0;
   v[0] = 0.0; // no velocity in x-direction
   v[1] = time < tr ? -2.0 * Problem_ctx.Dmax * time / (tr * tr) : 0.0; // velocity in y-direction
}

void acceleration_compression_fun(const Vector &x, double time, Vector &v)
{
   const real_t tr = Problem_ctx.release_time;

   v = 0.0;
   v[0] = 0.0; // no acceleration in x-direction
   v[1] = time < tr ? -2.0 * Problem_ctx.Dmax / (tr * tr) : 0.0; // acceleration in y-direction
}
