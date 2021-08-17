//                          MFEM Example 1, modified
//
// This code has been modified from `ex1.cpp` provided in the examples of
// MFEM.  The sections not marked as related to Ginkgo are largely unchanged
// from the version provided by MFEM.
//
// This code also contains portions from `miniapps/performance/ex1.cpp`
// pertaining to the LOR preconditioner.  The preconditioner types
// used in this example are applied to the SparseMatrix created from
// the LOR mesh, so they are all LOR preconditioners with different
// subtypes.
//
// The default mesh option is "star.mesh", provided by MFEM.
// Important non-default options:
//  -m [file] : Mesh file.
//  -d "cuda" : Use the MFEM cuda backend and Ginkgo CudaExecutor.
//  -pc-type "gko:ic" : Use the Ginkgo IC preconditioner (default is Block
//                       Jacobi)
//  -pc-type "none" : No LOR preconditioner
//
//  Options only for the Block Jacobi preconditioner (default:)
//  -pc-so "none" : Don't let Ginkgo automatically pick options for precision
//                  reduction in the storage of the Block Jacobi preconditioner
//  -pc-acc [value] : Accuracy parameter.
//
// MFEM's provided information about `ex1.cpp`:
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
#include "../../general/forall.hpp"
#include "multigridpc.hpp"
#include <cuda_profiler_api.h>

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double coeff_func(const Vector & x);

// helper functions for cg/pcg solve with timer and iter count return
int cg_solve(const Operator &A, const Vector &b, Vector &x,
             int print_iter, int max_num_iter,
             double RTOLERANCE, double ATOLERANCE, double &it_time)
{

   CGSolver cg;
   cg.SetPrintLevel(print_iter);
   cg.SetMaxIter(max_num_iter);
   cg.SetRelTol(sqrt(RTOLERANCE));
   cg.SetAbsTol(sqrt(ATOLERANCE));
   cg.SetOperator(A);

   tic_toc.Clear();
   tic_toc.Start();

   cg.Mult(b, x);

   tic_toc.Stop();
   it_time = tic_toc.RealTime();

   return cg.GetNumIterations();
}

int pcg_solve(const Operator &A, Solver &B, const Vector &b, Vector &x,
              int print_iter, int max_num_iter,
              double RTOLERANCE, double ATOLERANCE, double &it_time)
{

   CGSolver pcg;
   pcg.SetPrintLevel(print_iter);
   pcg.SetMaxIter(max_num_iter);
   pcg.SetRelTol(sqrt(RTOLERANCE));
   pcg.SetAbsTol(sqrt(ATOLERANCE));
   pcg.SetOperator(A);
   pcg.SetPreconditioner(B);

   tic_toc.Clear();
   tic_toc.Start();

   pcg.Mult(b, x);

   tic_toc.Stop();
   it_time = tic_toc.RealTime();

   return pcg.GetNumIterations();
}

int gko_cg_solve(Ginkgo::GinkgoExecutor &exec, const Operator &A, Ginkgo::GinkgoPreconditioner &B, const Vector &b, Vector &x,
                  int print_iter, int max_num_iter,
                  double RTOLERANCE, double ATOLERANCE, double &it_time)
{

   Ginkgo::CGSolver pcg(exec, B);

   pcg.SetPrintLevel(print_iter);
   pcg.SetMaxIter(max_num_iter);
   pcg.SetRelTol(RTOLERANCE);
   pcg.SetAbsTol(ATOLERANCE);
   pcg.SetOperator(A);

   tic_toc.Clear();
   tic_toc.Start();

   pcg.Mult(b, x);

   tic_toc.Stop();
   it_time = tic_toc.RealTime();

   return pcg.GetNumIterations();

}

int gko_fcg_solve(Ginkgo::GinkgoExecutor &exec, const Operator &A, Ginkgo::GinkgoPreconditioner &B, const Vector &b, Vector &x,
                  int print_iter, int max_num_iter,
                  double RTOLERANCE, double ATOLERANCE, double &it_time)
{

   Ginkgo::FCGSolver pfcg(exec, B);

   pfcg.SetPrintLevel(print_iter);
   pfcg.SetMaxIter(max_num_iter);
   pfcg.SetRelTol(RTOLERANCE);
   pfcg.SetAbsTol(ATOLERANCE);
   pfcg.SetOperator(A);

   tic_toc.Clear();
   tic_toc.Start();

   pfcg.Mult(b, x);

   tic_toc.Stop();
   it_time = tic_toc.RealTime();

   return pfcg.GetNumIterations();

}

int gmres_solve(const Operator &A, const Vector &b, Vector &x,
                int print_iter, int max_num_iter,
                double RTOLERANCE, double ATOLERANCE, double &it_time)
{

   GMRESSolver gmres;
   gmres.SetPrintLevel(print_iter);
   gmres.SetMaxIter(max_num_iter);
   gmres.SetRelTol(sqrt(RTOLERANCE));
   gmres.SetAbsTol(sqrt(ATOLERANCE));
   gmres.SetOperator(A);

   tic_toc.Clear();
   tic_toc.Start();

   gmres.Mult(b, x);

   tic_toc.Stop();
   it_time = tic_toc.RealTime();

   return gmres.GetNumIterations();
}
int pgmres_solve(const Operator &A, Solver &B, const Vector &b, Vector &x,
                 int print_iter, int max_num_iter,
                 double RTOLERANCE, double ATOLERANCE, double &it_time)
{

   GMRESSolver pgmres;
   pgmres.SetPrintLevel(print_iter);
   pgmres.SetMaxIter(max_num_iter);
   pgmres.SetRelTol(sqrt(RTOLERANCE));
   pgmres.SetAbsTol(sqrt(ATOLERANCE));
   pgmres.SetOperator(A);
   pgmres.SetPreconditioner(B);

   tic_toc.Clear();
   tic_toc.Start();

   pgmres.Mult(b, x);

   tic_toc.Stop();
   it_time = tic_toc.RealTime();

   return pgmres.GetNumIterations();
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int ref_levels = 3;
   int coarse_ref_levels = 3;
   const char *coeff_name = "var";
   int order = 2;
   const char *basis_type = "G"; // Gauss-Lobatto
   bool static_cond = false;
   bool pa = true;
   const char *device_config = "cpu";
   bool visualization = false;
   const char *pc_type = "gko:bj";
   const char *pc_storage_opt = "auto";
   double pc_acc = 1.e-1;
   int pc_max_bs = 32;
   bool permute = false;
   bool output_sol = false;
   bool output_pc = false;
   int isai_sparsity_power = 1;
   int par_ic_its = 0;
   double sigma_val = 1.0;
   const char *mg_spec = "1";
   const char *mg_coarse_solve = "mfem:jacobi";
   const char *mg_smoother = "mfem:cheb";
   bool mg_pa = false;
   const char *amgx_file = "amgx.json";
   bool use_mixed_amg = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ref_levels, "-l", "--refinement-levels",
                  "Number of uniform refinement levels for mesh.");
   args.AddOption(&coarse_ref_levels, "-cl", "--coarse-refinement-levels",
                  "Number of uniform refinement levels for coarse mesh (GMG only).");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&coeff_name, "-c", "--coeff",
                  "Type of coefficient for Laplace operator.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&pc_type, "-pc-type", "--preconditioner-type",
                  "Type of preconditioner used on LOR matrix.");
   args.AddOption(&pc_storage_opt, "-pc-so",
                  "--preconditioner-storage-optimization",
                  "Type of precision storage optimization to use for Ginkgo BlockJacobi.");
   args.AddOption(&pc_acc, "-pc-acc", "--preconditioner-accuracy",
                  "Accuracy parameter for Ginkgo BlockJacobi.");
   args.AddOption(&pc_max_bs, "-pc-mbs", "--preconditioner-max-block-size",
                  "Maximum block size for Ginkgo BlockJacobi.");
   args.AddOption(&permute, "-per", "--permutation", "-no-per",
                  "--no-permutation", "Enable preconditioner permutation.");
   args.AddOption(&output_sol, "-out", "--output-solution-and-mesh", "-no-out",
                  "--no-solution-and-mesh-output",
                  "Output mesh and solution for inspection.");
   args.AddOption(&output_pc, "-out-pc", "--output-lor-matrix-and-mesh",
                  "-no-out-pc",
                  "--no-lor-matrix-and-mesh-output",
                  "Output LOR mesh and sparse matrix for inspection.");
   args.AddOption(&isai_sparsity_power, "-isai-sp", "--isai-sparsity-power",
                  "Power to use for sparsity pattern of ISAI in Ginkgo.");
   args.AddOption(&par_ic_its, "-pic-its", "--par-ic-iterations",
                  "Number of iterations for the Ginkgo ParIC algorithm.");
   args.AddOption(&sigma_val, "-sv", "--sigma-value",
                  "Non-unity value for piecewise discontinuous coefficient.");
   args.AddOption(&mg_spec, "-mg", "--multigrid-spec",
                  "Multigrid specification. See README for description.");
   args.AddOption(&mg_coarse_solve, "-mg-coarse-solve",
                  "--multigrid-coarse-solve-type",
                  "Type of solver for coarse grid in GMG.");
   args.AddOption(&mg_smoother, "-mg-smoother", "--multigrid-smoother-type",
                  "Type of solver for smoother in GMG.");
   args.AddOption(&mg_pa, "-mg-pa", "--multigrid-partial-assembly", "-no-mg-pa",
                  "--no-multigrid-partial-assembly",
                  "Enable Partial Assembly for non-coarse grids in GMG.");
   args.AddOption(&amgx_file, "-amgx-config", "--amgx-configuration-file",
                  "Configuration file for AMGX.");
   args.AddOption(&use_mixed_amg, "-mixed-amg", "--mixed-precision-amg",
                  "-dbl-amg",
                  "--double-precision-amg",
                  "Enable mixed precision AMG from Ginkgo.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   enum CoeffType { CONSTANT, VARIABLE, PW_CONSTANT };
   CoeffType coeff_type;
   if (!strcmp(coeff_name, "const")) { coeff_type = CONSTANT; }
   else if (!strcmp(coeff_name, "var")) { coeff_type = VARIABLE; }
   else if (!strcmp(coeff_name, "pw")) { coeff_type = PW_CONSTANT; }
   else
   {
      mfem_error("Invalid coefficient type specified");
   }

   enum PCType { NONE,
                 GKO_BLOCK_JACOBI,
                 GKO_IC,
                 GKO_IC_ISAI,
                 GKO_CUIC,
                 GKO_CUIC_ISAI,
                 GKO_AMG,
                 MFEM_GS,
                 MFEM_GMG,
                 MFEM_AMGX,
                 MFEM_UMFPACK
               };
   PCType pc_choice;
   bool pc = true;
   if (!strcmp(pc_type, "gko:bj")) { pc_choice = GKO_BLOCK_JACOBI; }
   else if (!strcmp(pc_type, "gko:ic")) { pc_choice = GKO_IC; }
   else if (!strcmp(pc_type, "gko:ic-isai"))
   {
      pc_choice = GKO_IC_ISAI;
   }
   else if (!strcmp(pc_type, "gko:cuic")) { pc_choice = GKO_CUIC; }
   else if (!strcmp(pc_type, "gko:cuic-isai"))
   {
      pc_choice = GKO_CUIC_ISAI;
   }
   else if (!strcmp(pc_type, "gko:amg")) { pc_choice = GKO_AMG; }
   else if (!strcmp(pc_type, "mfem:gs")) { pc_choice = MFEM_GS; }
   else if (!strcmp(pc_type, "mfem:gmg")) { pc_choice = MFEM_GMG; }
   else if (!strcmp(pc_type, "mfem:amgx")) { pc_choice = MFEM_AMGX; }
   else if (!strcmp(pc_type, "mfem:umf"))
   {
#ifdef MFEM_USE_SUITESPARSE
      pc_choice = MFEM_UMFPACK;
#else
      mfem_error("Preconditioner requires SuiteSparse");
#endif
   }
   else if (!strcmp(pc_type, "none"))
   {
      pc_choice = NONE;
      pc = false;
   }
   else
   {
      mfem_error("Invalid Preconditioner specified");
      return 3;
   }

   SolverConfig::SolverType coarse_solver_type;
   if (!strcmp(mg_coarse_solve, "mfem:jacobi")) { coarse_solver_type = SolverConfig::JACOBI; }
   else if (!strcmp(mg_coarse_solve, "mfem:amgx")) { coarse_solver_type = SolverConfig::AMGX; }
   else if (!strcmp(mg_coarse_solve, "gko:cuic")) { coarse_solver_type = SolverConfig::GINKGO_CUIC; }
   else if (!strcmp(mg_coarse_solve, "gko:cuic-isai")) { coarse_solver_type = SolverConfig::GINKGO_CUIC_ISAI; }
   else
   {
      mfem_error("Invalid coarse grid solver specified");
      return 3;
   }
   SolverConfig::SolverType smoother_type;
   if (!strcmp(mg_smoother, "mfem:cheb")) { smoother_type = SolverConfig::CHEBYSHEV; }
   else if (!strcmp(mg_smoother, "gko:cuic")) { smoother_type = SolverConfig::GINKGO_CUIC; }
   else if (!strcmp(mg_smoother, "gko:cuic-isai")) { smoother_type = SolverConfig::GINKGO_CUIC_ISAI; }
   else
   {
      mfem_error("Invalid smoother type specified");
      return 3;
   }
   AssemblyLevel mg_asm_lvl = AssemblyLevel::LEGACYFULL;
   if (mg_pa) { mg_asm_lvl = AssemblyLevel::PARTIAL; }

   // See class BasisType in fem/fe_coll.hpp for available basis types
   int basis = BasisType::GetType(basis_type[0]);
   cout << "Using " << BasisType::Name(basis) << " basis ..." << endl;

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // Create Ginkgo executor.
   Ginkgo::GinkgoExecutor exec(device);

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement.
   {
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }
   cout << "Total elements in refined mesh: " <<  mesh->GetNE() << endl;

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim, basis);
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim, basis);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
        << endl;

   // Create the LOR mesh and finite element space. In the settings of this
   // example, we can transfer between HO and LOR with the identity operator.
   Mesh *mesh_lor = NULL;
   FiniteElementCollection *fec_lor = NULL;
   FiniteElementSpace *fespace_lor = NULL;
   Array<int> *inv_reordering = NULL;
   if (pc)
   {
      int basis_lor = basis;
      if (basis == BasisType::Positive) { basis_lor=BasisType::ClosedUniform; }
      mesh_lor = new Mesh(mesh, order, basis_lor);
      fec_lor = new H1_FECollection(1, dim);
      fespace_lor = new FiniteElementSpace(mesh_lor, fec_lor);
   }

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

   Array<int> ess_pc_tdof_list(ess_tdof_list.Size());
   if (permute)
   {
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         ess_pc_tdof_list.operator[](i) = inv_reordering->operator[](
                                             ess_tdof_list.operator[](i));
      }
   }
   else
   {

      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_pc_tdof_list);
   }

   // 7. Set up the linear form b(.) which corresponds to the right-hand side
   // of
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
   //    corresponding to the Laplacian operator -Delta, by adding the
   //    Diffusion domain integrator.
   BilinearForm *a = new BilinearForm(fespace);
   BilinearForm *a_pc = NULL;
   if (pc) { a_pc = new BilinearForm(fespace_lor); }
   if (pa)
   {
      a->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }

   Coefficient *coeff = NULL;

   if (coeff_type == CONSTANT)
   {
      coeff = new ConstantCoefficient(1.0);
   }
   else if (coeff_type == VARIABLE)
   {
      coeff = new FunctionCoefficient(&coeff_func);
   }
   else if (coeff_type == PW_CONSTANT)
   {
      int num_subregions = mesh->attributes.Max();
      cout << "Number of subregions in mesh: " << num_subregions << "\n";
      Vector sigma(num_subregions);
      sigma = 1;
      if (num_subregions < 2)
      {
         cout << "Warning: PW Constant Coefficient not used, mesh only has one element attribute!\n";
      }
      else
      {
         sigma(num_subregions-1) = sigma_val;
      }
      coeff = new PWConstCoefficient(sigma);
   }
   a->AddDomainIntegrator(new DiffusionIntegrator(*coeff));

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond)
   {
      a->EnableStaticCondensation();
   }
   a->Assemble();

   OperatorPtr A;
   Vector B, X;

   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   // 11. Solve the linear system A X = B.
   double it_time = 0.;
   int total_its = 0;

   OperatorHandle A_pc;
   SparseMatrix *A_pc_mat = NULL;
   if (pc)
   {
      tic_toc.Clear();
      tic_toc.Start();

      a_pc->AddDomainIntegrator(new DiffusionIntegrator(*coeff));
      a_pc->SetAssemblyLevel(AssemblyLevel::FULL);
      a_pc->Assemble();
      //A_pc = &(a_pc->SpMat());

      tic_toc.Stop();
      cout << "Real time assembling A_pc SparseMatrix: " <<
           tic_toc.RealTime() << "\n";

      tic_toc.Clear();
      tic_toc.Start();
      //      a_pc->FormSystemMatrix(ess_pc_tdof_list, A_pc);
      //      A_pc_mat = dynamic_cast<SparseMatrix*>(A_pc.Ptr());
      A_pc_mat = &(a_pc->SpMat());

      // Manually set essential BC rows/columns, so we can use device if applicable
      // Get device or host pointers
      bool on_dev = false;
      if (!strcmp(device_config, "cuda"))
      {
         on_dev = true;
      }
      auto d_I = A_pc_mat->ReadWriteI(on_dev);
      auto d_J = A_pc_mat->ReadWriteJ(on_dev);
      auto d_A = A_pc_mat->ReadWriteData(on_dev);
      auto d_ess_list = ess_pc_tdof_list.Read(on_dev);
      MFEM_FORALL_SWITCH(on_dev, i, ess_pc_tdof_list.Size(),
      {
         int rc = d_ess_list[i];
         if (rc < 0 ) { rc = -1-rc; }
         for (int j = d_I[rc]; j < d_I[rc+1]; j++)
         {
            const int col = d_J[j];
            if (col != rc)
            {
               d_A[j] = 0.0;
               for (int k = d_I[col]; 1; k++)
               {
                  if (k == d_I[col+1])
                  {
                     break;
                  }
                  else if (d_J[k] == rc)
                  {
                     d_A[k] = 0.0;
                     break;
                  }
               }
            }
         }
      });


      tic_toc.Stop();
      cout << "Real time adjusting A_pc for essential BC: " <<
           tic_toc.RealTime() << "\n";

      int max_iter = 15000;
//      int max_iter = 1;
      if (pc_choice == GKO_BLOCK_JACOBI)
      {

         // Create Ginkgo Jacobi preconditioner
         tic_toc.Clear();
         tic_toc.Start();
         Ginkgo::JacobiPreconditioner M(exec, pc_storage_opt, pc_acc, pc_max_bs);
         M.SetOperator(*A_pc_mat);
         tic_toc.Stop();
         cout << "Real time creating Ginkgo BlockJacobi preconditioner: " <<
              tic_toc.RealTime() << "\n";

         // Use preconditioned CG
         total_its = pcg_solve(*A, M, B, X, 0, X.Size(), 1e-12, 0.0, it_time);

         cout << "Real time in PCG: " << it_time << "\n";
      }
      else if (pc_choice == GKO_IC)
      {

         // Create Ginkgo IC preconditioner
         tic_toc.Clear();
         tic_toc.Start();

         Ginkgo::IcPreconditioner M(exec, "paric", par_ic_its);
         M.SetOperator(*A_pc_mat);

         tic_toc.Stop();
         cout << "Real time creating Ginkgo Ic preconditioner: " <<
              tic_toc.RealTime() << "\n";

         // Use preconditioned CG
         total_its = pcg_solve(*A, M, B, X, 0, X.Size(), 1e-12, 0.0, it_time);

         cout << "Real time in PCG: " << it_time << "\n";

      }
      else if (pc_choice == GKO_IC_ISAI)
      {

         // Create Ginkgo IC preconditioner
         tic_toc.Clear();
         tic_toc.Start();

         Ginkgo::IcIsaiPreconditioner M(exec, "paric", par_ic_its);
         M.SetOperator(*A_pc_mat);

         tic_toc.Stop();
         cout << "Real time creating Ginkgo Ic preconditioner: " <<
              tic_toc.RealTime() << "\n";

         // Use preconditioned CG
         total_its = pcg_solve(*A, M, B, X, 0, X.Size(), 1e-12, 0.0, it_time);

         cout << "Real time in PCG: " << it_time << "\n";

      }
      else if (pc_choice == GKO_CUIC)
      {

         // Create Ginkgo CuIC preconditioner (uses cuSPARSE for factorization)
         tic_toc.Clear();
         tic_toc.Start();

         Ginkgo::IcPreconditioner M(exec, "exact");
         M.SetOperator(*A_pc_mat);

         tic_toc.Stop();
         cout << "Real time creating Ginkgo CuIc preconditioner: " <<
              tic_toc.RealTime() << "\n";

         // Use preconditioned CG
         total_its = pcg_solve(*A, M, B, X, 0, X.Size(), 1e-12, 0.0, it_time);

         cout << "Real time in PCG: " << it_time << "\n";

      }
      else if (pc_choice == GKO_CUIC_ISAI)
      {

         // Create Ginkgo CuIC preconditioner (uses cuSPARSE for factorization)
         tic_toc.Clear();
         tic_toc.Start();

         Ginkgo::IcPreconditioner M(exec, "exact", 0, isai_sparsity_power);
         M.SetOperator(*A_pc_mat);

         tic_toc.Stop();
         cout << "Real time creating Ginkgo CuIcIsai preconditioner: " <<
              tic_toc.RealTime() << "\n";

         // Use preconditioned CG
         total_its = pcg_solve(*A, M, B, X, 0, X.Size(), 1e-12, 0.0, it_time);

         cout << "Real time in PCG: " << it_time << "\n";

      }
      else if (pc_choice == GKO_AMG)
      {

         // Create Ginkgo CuIC preconditioner (uses cuSPARSE for factorization)
         tic_toc.Clear();
         tic_toc.Start();

//         cudaProfilerStart(); 
         Ginkgo::AMGPreconditioner M(exec, Ginkgo::AMGPreconditioner::JACOBI,
                                     1, 1, Ginkgo::AMGPreconditioner::JACOBI, 4, use_mixed_amg);
         M.SetOperator(*A_pc_mat);

         tic_toc.Stop();
         cout << "Real time creating Ginkgo AMG preconditioner: " <<
              tic_toc.RealTime() << "\n";

         // Use preconditioned CG
         total_its = pcg_solve(*A, M, B, X, 1, max_iter, 1e-12, 0.0, it_time);
//         total_its = gko_fcg_solve(exec, *A, M, B, X, 1, max_iter, 1e-12, 0.0, it_time);
//         total_its = gko_cg_solve(exec, *A, M, B, X, 1, max_iter, 1e-12, 0.0, it_time);
//         cudaProfilerStop(); 

         cout << "Real time in PCG: " << it_time << "\n";

      }
      else if (pc_choice == MFEM_GS)
      {

         // Create MFEM preconditioner
         tic_toc.Clear();
         tic_toc.Start();

         GSSmoother M(*A_pc_mat);

         tic_toc.Stop();
         cout << "Real time creating MFEM GS preconditioner: " <<
              tic_toc.RealTime() << endl;

         // Use preconditioned CG
         total_its = pcg_solve(*A, M, B, X, 0, max_iter, 1e-12, 0.0, it_time);

         cout << "Real time in PCG: " << it_time << endl;

      }
      else if (pc_choice == MFEM_UMFPACK)
      {

#ifdef MFEM_USE_SUITESPARSE
         // Create MFEM preconditioner
         tic_toc.Clear();
         tic_toc.Start();

         UMFPackSolver M;
         M.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         M.SetOperator(*A_pc_mat);

         tic_toc.Stop();
         cout << "Real time creating MFEM UMFPACK preconditioner: " <<
              tic_toc.RealTime() << endl;

         // Use preconditioned CG
         total_its = pcg_solve(*A, M, B, X, 0, max_iter, 1e-12, 0.0, it_time);

         cout << "Real time in PCG: " << it_time << endl;
#endif
      }
      else if (pc_choice == MFEM_AMGX)
      {
#ifdef MFEM_USE_AMGX
         // Create MFEM preconditioner
         tic_toc.Clear();
         tic_toc.Start();

         AmgXSolver M;
         M.ReadParameters(amgx_file, AmgXSolver::EXTERNAL);
         M.InitSerial();
         M.SetOperator(*A_pc_mat);

         tic_toc.Stop();
         cout << "Real time creating MFEM AMGX preconditioner: " <<
              tic_toc.RealTime() << "\n";

         // Use preconditioned CG
         total_its = pcg_solve(*A, M, B, X, 0, max_iter, 1e-12, 0.0, it_time);
//         Ginkgo::MFEMPreconditioner M_gko(exec, M);
//         total_its = gko_fcg_solve(exec, *A, M_gko, B, X, 0, max_iter, 1e-12, 0.0, it_time);

         cout << "Real time in PCG: " << it_time << "\n";
#endif
      }
      else if (pc_choice == MFEM_GMG)
      {
         // Create coarse grid mesh
         Mesh *coarse_mesh = new Mesh(mesh_file, 1, 1);
         // Refine the coarse mesh to increase the resolution.
         for (int l = 0; l < coarse_ref_levels; l++)
         {
            coarse_mesh->UniformRefinement();
         }

         // Parse arguments:

         int coarse_order = 0, order = 0, h_ref = 0;
         // Parse order specification
         vector<MGRefinement> mg_refinements;
         {
            istringstream mg_stream(mg_spec);
            string ref;
            mg_stream >> coarse_order;
            int prev_order = order = coarse_order;
            cout << "\nCoarse order " << coarse_order << '\n';
            while (mg_stream >> ref)
            {
               if (ref == "r")
               {
                  cout << "h-MG uniform refinement\n";
                  mg_refinements.push_back(MGRefinement::h());
                  ++h_ref;
               }
               else
               {
                  try { order = stoi(ref); }
                  catch (...)
                  {
                     MFEM_ABORT("Multigrid refinement must either be an integer or "
                                "the character `r`");
                  }
                  cout << "p-MG order   " << order << '\n';
                  MFEM_VERIFY(order > 0, "Orders must be positive");
                  MFEM_VERIFY(order > prev_order, "Orders must be sorted");
                  mg_refinements.push_back(MGRefinement::p(order));
                  prev_order = order;
               }
            }
            cout << endl;
         }

         // Create MFEM preconditioner
         tic_toc.Clear();
         tic_toc.Start();

         std::vector<FiniteElementCollection*> fe_collections;
         fe_collections.push_back(new H1_FECollection(coarse_order, dim));
         FiniteElementSpace fes_coarse(coarse_mesh, fe_collections.back());

         FiniteElementSpaceHierarchy hierarchy(coarse_mesh, &fes_coarse, false, false);

         for (MGRefinement ref : mg_refinements)
         {
            if (ref.type == MGRefinement::H_MG)
            {
               hierarchy.AddUniformlyRefinedLevel();
            }
            else // P_MG
            {
               fe_collections.push_back(new H1_FECollection(ref.order, dim));
               hierarchy.AddOrderRefinedLevel(fe_collections.back());
            }
         }

         SolverConfig gmg_solver(coarse_solver_type, smoother_type,
                                 mg_asm_lvl, amgx_file, device);
         Array<int> ess_bdr(coarse_mesh->bdr_attributes.Max());
         ess_bdr = 1;
         DiffusionMultigrid M(hierarchy, *coeff, ess_bdr, gmg_solver);
         M.SetCycleType(Multigrid::CycleType::VCYCLE, 1, 1);

         tic_toc.Stop();
         cout << "Real time creating MFEM GMG preconditioner: " <<
              tic_toc.RealTime() << "\n";

         // Use preconditioned CG
         total_its = pcg_solve(*A, M, B, X, 0, max_iter, 1e-12, 0.0, it_time);

         cout << "Real time in PCG: " << it_time << "\n";

      }
   }
   else
   {

      int max_iter = 15000;
   //   int max_iter = 1;
      total_its = cg_solve(*A, B, X, 0, max_iter, 1e-12, 0.0, it_time);

      cout << "Real time in CG: " << it_time << endl;
   }

   cout << "Total iterations: " << total_its << endl;
   cout << "Avg time per iteration: " << it_time/double(
           total_its) << endl;

   // 12. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);

   // 13. Save the refined mesh and the solution. This output can be viewed
   // later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".

   if (output_sol)
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   if (pc && output_pc)
   {
      ofstream mesh_lor_ofs("lor-refined.mesh");
      mesh_lor_ofs.precision(8);
      mesh_lor->Print(mesh_lor_ofs);

      ofstream apc_lor_ofs("lor-mat.dat");
      apc_lor_ofs.precision(8);
      A_pc_mat->PrintCSR(apc_lor_ofs);

      ofstream apc_lor_ofs_mm("lor-mat-mm.dat");
      apc_lor_ofs_mm.precision(8);
      A_pc_mat->PrintMM(apc_lor_ofs_mm);
   }

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 15. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   delete fespace_lor;
   delete fec_lor;
   delete mesh_lor;
   delete coeff;
   if (order > 0)
   {
      delete fec;
   }
   delete mesh;
}

double coeff_func(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));
   double zi(0.0);

   if (x.Size() == 3)
   {
      zi = x(2);
   }

   return 0.2*(sin(2*xi)*sin(2*yi)*cos(2*zi) + 2.0);
}
