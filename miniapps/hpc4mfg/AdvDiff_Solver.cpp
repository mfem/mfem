#include "mfem.hpp"
#include "advection_diffusion_solver.hpp"

int main(int argc, char *argv[])
{
   // 1. Initialize MPI
   int num_procs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // Define Caliper ConfigManager
#ifdef MFEM_USE_CALIPER
   cali::ConfigManager mgr;
#endif

   // Caliper instrumentation
   MFEM_PERF_FUNCTION;

   // 2. Parse command-line options
   const char *mesh_file = "../../../data/one_ele.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 2;
   bool visualization = true;
   double newton_rel_tol = 1e-4;
   double newton_abs_tol = 1e-6;
   int newton_iter = 1;
   int print_level = 0;


   const char* cali_config = "runtime-report";

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&newton_rel_tol,
                  "-rel",
                  "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&newton_abs_tol,
                  "-abs",
                  "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&newton_iter,
                  "-it",
                  "--newton-iterations",
                  "Maximum iterations for the Newton solve.");
   args.AddOption((&print_level), "-prt", "--print-level", "Print level.");
   args.AddOption(&cali_config, "-p", "--caliper",
                  "Caliper configuration string.");
   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }

   // Caliper configuration
#ifdef MFEM_USE_CALIPER
   mgr.add(cali_config);
   mgr.start();
#endif

   //    Read the (serial) mesh from the given mesh file on all processors. We
   //    can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.
   mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh *pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // Allocate the nonlinear diffusion solver
   mfem::Advection_Diffusion_Solver* solver=new mfem::Advection_Diffusion_Solver(pmesh,order);

   solver->SetLinearSolver(1e-12, 1e-15, 10000);

   //add boundary conditions
   solver->AddDirichletBC(1,3.0);

   //add material
   solver->AddMaterial(new mfem::IdentityMatrixCoefficient(dim));


   mfem::Vector ConstVelVector(pmesh->Dimension());   ConstVelVector = 1.0;   //ConstVelVector(0) = 1.0;
   mfem::VectorConstantCoefficient velCoeff(ConstVelVector);

   solver->SetVelocity( &velCoeff );

   mfem::Vector ConstVector(pmesh->Dimension());   ConstVector = 0.0;   //ConstVector(0) = 1.0;
   mfem::VectorConstantCoefficient avgTemp(ConstVector);

   solver->SetGradTempMean( &avgTemp );

   //solve
   solver->FSolve();

   solver->Postprocess();

   delete solver;
   delete pmesh;
   // Flush output before MPI_finalize
#ifdef MFEM_USE_CALIPER
   mgr.flush();
#endif
   MPI_Finalize();

   return 0;
}
