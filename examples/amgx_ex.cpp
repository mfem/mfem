//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/star-mixed.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/toroid-wedge.mesh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/star-mixed-p2.mesh -o 2
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/fichera-mixed-p2.mesh -o 2
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               ex1 -pa -d cuda
//               ex1 -pa -d raja-cuda
//               ex1 -pa -d occa-cuda
//               ex1 -pa -d raja-omp
//               ex1 -pa -d occa-omp
//               ex1 -pa -d ceed-cpu
//               ex1 -pa -d ceed-cuda
//               ex1 -m ../data/beam-hex.mesh -pa -d cuda
//               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cpu
//               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cuda:/gpu/cuda/ref
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include "amgx_c.h"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/**
   The dream is that this becomes almost a replacement for
   HypreAMG, like with an NvidiaAMGX(SparseMatrix) constructor.
*/
class NvidiaAMGX : public Solver
{
public:
   NvidiaAMGX();

   void ConfigureAsSolver(bool verbose=false);
   void ConfigureAsPreconditioner(bool verbose=false);

   void Configure(const char * amgx_json_file,
                  const char * amgx_parameter);

   ~NvidiaAMGX();

   void SetOperator(const Operator& op);

   void Mult(const Vector& x, Vector& y) const;

private:
   bool configured;

   // library handles
   AMGX_Mode mode;
   AMGX_config_handle cfg;
   AMGX_resources_handle rsrc;
   AMGX_matrix_handle amgx_A;
   AMGX_vector_handle amgx_b, amgx_x;
   AMGX_solver_handle solver;

   // status handling
   // AMGX_SOLVE_STATUS status;

   SparseMatrix * spop;
};

NvidiaAMGX::NvidiaAMGX()
   : configured(false)
{
}

void NvidiaAMGX::ConfigureAsSolver(bool verbose)
{
   std::string configs = "{\n"
                         "    \"config_version\": 2, \n"
                         "    \"solver\": {\n"
                         "        \"max_uncolored_percentage\": 0.15, \n"
                         "        \"algorithm\": \"AGGREGATION\", \n"
                         "        \"solver\": \"AMG\", \n"
                         "        \"smoother\": \"MULTICOLOR_GS\", \n"
                         "        \"presweeps\": 1, \n"
                         "        \"symmetric_GS\": 1, \n"
                         "        \"selector\": \"SIZE_2\", \n"
                         "        \"coarsest_sweeps\": 10, \n"
                         "        \"max_iters\": 1000, \n"
                         "        \"postsweeps\": 1, \n"
                         "        \"scope\": \"main\", \n"
                         "        \"max_levels\": 1000, \n"
                         "        \"matrix_coloring_scheme\": \"MIN_MAX\", \n"
                         "        \"tolerance\": 0.0000001, \n"
                         "        \"norm\": \"L2\", \n"
                         "        \"cycle\": \"V\"";

   if (verbose)
   {
      configs = configs + ",\n"
                "        \"obtain_timings\": 1, \n"
                "        \"monitor_residual\": 1, \n"
                "        \"print_grid_stats\": 1, \n"
                "        \"print_solve_stats\": 1 \n";
   }
   else
   {
      configs = configs + "\n";
   }
   configs = configs + "    }\n" + "}\n";

   AMGX_SAFE_CALL(AMGX_config_create(&cfg, configs.c_str()));
   configured = true;
}

void NvidiaAMGX::ConfigureAsPreconditioner(bool verbose)
{
   std::string configs = "{\n"
                         "    \"config_version\": 2, \n"
                         "    \"solver\": {\n"
                         "        \"max_uncolored_percentage\": 0.15, \n"
                         "        \"algorithm\": \"AGGREGATION\", \n"
                         "        \"solver\": \"AMG\", \n"
                         "        \"smoother\": \"MULTICOLOR_GS\", \n"
                         "        \"presweeps\": 1, \n"
                         "        \"symmetric_GS\": 1, \n"
                         "        \"selector\": \"SIZE_2\", \n"
                         "        \"coarsest_sweeps\": 10, \n"
                         "        \"max_iters\": 2, \n"
                         "        \"postsweeps\": 1, \n"
                         "        \"scope\": \"main\", \n"
                         "        \"max_levels\": 1000, \n"
                         "        \"matrix_coloring_scheme\": \"MIN_MAX\", \n"
                         "        \"tolerance\": 0.0, \n"
                         "        \"norm\": \"L2\", \n"
                         "        \"cycle\": \"V\"";

   if (verbose)
   {
      configs = configs + ",\n"
                "        \"obtain_timings\": 1, \n"
                "        \"monitor_residual\": 1, \n"
                "        \"print_grid_stats\": 1, \n"
                "        \"print_solve_stats\": 1 \n";
   }
   else
   {
      configs = configs + "\n";
   }
   configs = configs + "    }\n" + "}\n";

   AMGX_SAFE_CALL(AMGX_config_create(&cfg, configs.c_str()));
   configured = true;
}

void NvidiaAMGX::Configure(const char * amgx_json_file,
                           const char * amgx_parameter)
{
   if (strcmp(amgx_json_file, "") != 0 && strcmp(amgx_parameter, "") != 0)
   {
      std::cout << "Using file " << amgx_json_file << " AND config "
                << amgx_parameter << std::endl;
      AMGX_SAFE_CALL(AMGX_config_create_from_file_and_string(&cfg, amgx_json_file,
                                                             amgx_parameter));
      configured = true;
   }
   else if (strcmp(amgx_json_file, "") != 0)
   {
      std::cout << "Using file " << amgx_json_file << std::endl;
      AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, amgx_json_file));
      configured = true;
   }
   else if (strcmp(amgx_parameter, "") != 0)
   {
      std::cout << "Using config " << amgx_parameter << std::endl;
      AMGX_SAFE_CALL(AMGX_config_create(&cfg, amgx_parameter));
      configured = true;
   }
   else
   {
      configured = false;  // no-op is fine but this is the intent
   }
}

NvidiaAMGX::~NvidiaAMGX()
{
   AMGX_solver_destroy(solver);
   AMGX_matrix_destroy(amgx_A);
   AMGX_vector_destroy(amgx_x);
   AMGX_vector_destroy(amgx_b);
   AMGX_resources_destroy(rsrc);
}

void NvidiaAMGX::SetOperator(const Operator& op)
{
   if (!configured)
   {
      ConfigureAsSolver();
   }

   // data is assumed to be one device
   // (d)evice (D)ouble mat (D)ouble vec (I)nt32
   mode = AMGX_mode_dDDI;

   AMGX_resources_create_simple(&rsrc, cfg);
   AMGX_solver_create(&solver, rsrc, mode, cfg);
   AMGX_matrix_create(&amgx_A, rsrc, mode);
   AMGX_vector_create(&amgx_x, rsrc, mode);
   AMGX_vector_create(&amgx_b, rsrc, mode);

   spop = const_cast<SparseMatrix*>(dynamic_cast<const SparseMatrix*>(&op));
   MFEM_VERIFY(spop, "Operator is not of correct type!");

   AMGX_matrix_upload_all(amgx_A, spop->Height(),
                          spop->NumNonZeroElems(),
                          1, 1,
                          spop->ReadWriteI(),
                          spop->ReadWriteJ(),
                          spop->ReadWriteData(), NULL);

   AMGX_solver_setup(solver, amgx_A);
}

void NvidiaAMGX::Mult(const Vector& b, Vector& x) const
{
   AMGX_vector_upload(amgx_b, b.Size(), 1, b.Read());

   AMGX_vector_set_zero(amgx_x, x.Size(), 1);
   // AMGX_vector_upload(amgx_x, x.Size(), 1, x.Read()); // leads to nans...

   AMGX_solver_solve(solver, amgx_b, amgx_x);

   AMGX_vector_download(amgx_x, x.Write());

   //x.HostReadWrite();
   //x.Print();
}

int main(int argc, char *argv[])
{
   AMGX_initialize();
   AMGX_initialize_plugins();

   // 1. Parse command-line options.
   const char *mesh_file = "../data/beam-hex.mesh";
   //const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cuda";
   bool visualization = true;
   bool amgx_solver = false;
   bool amgx_verbose = false;
   const char* amgx_json_file = ""; // jason file for amgx
   const char* amgx_parameter = ""; // command line config for amgx

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&amgx_json_file, "--amgx-file", "--amgx-file",
                  "AMGX solver config file (overrides --amgx-solver, --amgx-verbose)");
   args.AddOption(&amgx_parameter, "--amgx-config", "--amgx-config",
                  "AMGX solver config as string (overrides --amgx-solver, --amgx-verbose)");
   args.AddOption(&amgx_solver, "--amgx-solver", "--amgx-solver",
                  "--amgx-preconditioner",
                  "--amgx-preconditioner",
                  "Configure AMGX as solver or preconditioner.");
   args.AddOption(&amgx_verbose, "--amgx-verbose", "--amgx-verbose",
                  "--amgx-no-verbose", "--amgx-no-verbose",
                  "Print verbose information from AMGX.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm *a = new BilinearForm(fespace);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new MassIntegrator(one));
   a->Assemble();
   a->Finalize();

   // 10. Solve the linear system A X = B.
   if (!pa)
   {
      NvidiaAMGX amgx;
      if (strcmp(amgx_json_file, "") != 0 || strcmp(amgx_parameter, "") != 0)
      {
         amgx.Configure(amgx_json_file, amgx_parameter);
      }
      else if (amgx_solver)
      {
         amgx.ConfigureAsSolver(amgx_verbose);
         amgx.SetOperator(a->SpMat());
         amgx.Mult(*b, x);
      }
      else
      {
         amgx.ConfigureAsPreconditioner(amgx_verbose);
         SparseMatrix A;
         Vector B, X;
         Array<int> ess_tdof_list(0);
         a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
         amgx.SetOperator(A);
         X = 0.0;
         PCG(A, amgx, B, X, 1, 40, 1e-12, 0.0);
      }
   }
   else // Jacobi preconditioning in partial assembly mode
   {
      OperatorPtr A;
      Vector B, X;
      Array<int> ess_tdof_list(0);
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

      if (UsesTensorBasis(*fespace))
      {
         OperatorJacobiSmoother M(*a, ess_tdof_list);
         PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
      }
      else
      {
         CG(*A, B, X, 1, 400, 1e-12, 0.0);
      }
      // Recover the solution as a finite element grid function.
      a->RecoverFEMSolution(X, *b, x);
      //x.HostReadWrite();
      //x.Print();
   }

   // 11. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 12. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 13. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   // NvidiaAMGX destructor has to be called before these
   // (could use a context or something)
   AMGX_finalize_plugins();
   AMGX_finalize();

   return 0;
}
