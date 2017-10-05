//                         MFEM Example 3  - High-Performance
//
// Compile with: make ex4
//
// Sample runs:  ex4 -m ../data/square-disc.mesh
//               ex4 -m ../data/star.mesh
//               ex4 -m ../data/beam-tet.mesh
//               ex4 -m ../data/beam-hex.mesh
//               ex4 -m ../data/escher.mesh
//               ex4 -m ../data/fichera.mesh -o 2 -hb
//               ex4 -m ../data/fichera-q2.vtk
//               ex4 -m ../data/fichera-q3.mesh -o 2 -sc
//               ex4 -m ../data/square-disc-nurbs.mesh
//               ex4 -m ../data/beam-hex-nurbs.mesh
//               ex4 -m ../data/periodic-square.mesh -no-bc
//               ex4 -m ../data/periodic-cube.mesh -no-bc
//               ex4 -m ../data/amr-quad.mesh
//               ex4 -m ../data/amr-hex.mesh
//               ex4 -m ../data/amr-hex.mesh -o 2 -hb
//               ex4 -m ../data/fichera-amr.mesh -o 2 -sc
//               ex4 -m ../data/star-surf.mesh -o 1
//
//  This example code solves a simple 2D/3D H(div) diffusion
//               problem corresponding to the second order definite equation
//               -grad(alpha div F) + beta F = f with boundary condition F dot n
//               = <given normal field>. Here, we use a given exact solution F
//               and compute the corresponding r.h.s. f.  We discretize with
//               Raviart-Thomas finite elements.
//
//               The example demonstrates the use of H(div) finite element
//               spaces with the grad-div and H(div) vector finite element mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known. Bilinear form
//               hybridization and static condensation are also illustrated.
//
//               We recommend viewing examples 1-3 before viewing this example.

#include "mfem.hpp"
#include "mfem-performance.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void F_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;
int dim;

// Define template parameters for optimized build.
const Geometry::Type geom     = Geometry::CUBE;//Geometry::SQUARE;// // mesh elements  (default: hex)
const int            mesh_p   = 3;              // mesh curvature (default: 3)
const int            sol_p    = 4;              // solution order (default: 3)
const int            rdim     = Geometry::Constants<geom>::Dimension;
const int            ir_order = 2*sol_p+rdim-1;

// Static mesh type
typedef H1_FiniteElement<geom,mesh_p>         mesh_fe_t;
typedef H1_FiniteElementSpace<mesh_fe_t>      mesh_fes_t;
typedef TMesh<mesh_fes_t>                     mesh_t;

// Static solution finite element space type
typedef RT_FiniteElement<geom,sol_p-1>          sol_fe_t;
typedef RT_FiniteElementSpace<sol_fe_t>       sol_fes_t;
// Static quadrature, coefficient and integrator types
typedef TIntegrationRule<geom,ir_order>       int_rule_t;
typedef TConstantCoefficient<>                coeff_t;
//typedef TIntegrator<coeff_t,THDivMassKernel> integ_t;
typedef TIntegrator<coeff_t,THDivDivKernel>  integ_t;

typedef RTShapeEvaluator<sol_fe_t,int_rule_t>  sol_Shape_Eval;
typedef RTFieldEvaluator<sol_fes_t,ScalarLayout,int_rule_t> sol_Field_Eval;

// Static bilinear form type, combining the above types
typedef TBilinearForm<mesh_t,sol_fes_t,int_rule_t,integ_t,sol_Shape_Eval,sol_Field_Eval> HPCBilinearForm;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/beam-hex.mesh";//"../../data/beam-quad.mesh";//
   int order = sol_p;
   bool static_cond = false;
   const char *pc = "none";
   bool perf = true;
   bool matrix_free = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&perf, "-perf", "--hpc-version", "-std", "--standard-version",
                   "Enable high-performance, tensor-based, assembly/evaluation.");
   args.AddOption(&matrix_free, "-mf", "--matrix-free", "-asm", "--assembly",
                   "Use matrix-free evaluation or efficient matrix assembly in "
                   "the high-performance version.");
   args.AddOption(&pc, "-pc", "--preconditioner",
                   "Preconditioner: lor - low-order-refined (matrix-free) GS, "
                   "ho - high-order (assembled) GS, none.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                   "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (static_cond && perf && matrix_free)
   {
      cout << "\nStatic condensation can not be used with matrix-free"
            " evaluation!\n" << endl;
      return 2;
   }
   MFEM_VERIFY(perf || !matrix_free,
                "--standard-version is not compatible with --matrix-free");
   args.PrintOptions(cout);
   kappa = freq * M_PI;

   enum PCType { NONE, LOR, HO };
   PCType pc_choice;
   if (!strcmp(pc, "ho")) { pc_choice = HO; }
   else if (!strcmp(pc, "lor")) { pc_choice = LOR; }
   else if (!strcmp(pc, "none")) { pc_choice = NONE; }
   else
   {
      mfem_error("Invalid Preconditioner specified");
      return 3;
   }

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // 3. Check if the optimized version matches the given mesh
   if (perf)
   {
      cout << "High-performance version using integration rule with "
           << int_rule_t::qpts << " points ..." << endl;
      if (!mesh_t::MatchesGeometry(*mesh))
      {
          cout << "The given mesh does not match the optimized 'geom' parameter.\n"
               << "Recompile with suitable 'geom' value." << endl;
          delete mesh;
          return 4;
      }
      else if (!mesh_t::MatchesNodes(*mesh))
      {
          cout << "Switching the mesh curvature to match the "
               << "optimized value (order " << mesh_p << ") ..." << endl;
          mesh->SetCurvature(mesh_p, false, -1, Ordering::byNODES);
      }
   }
    
   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
       int ref_levels = 0;
         //(int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }
   if (mesh->MeshGenerator() & 1) // simplex mesh
   {
      MFEM_VERIFY(pc_choice != LOR, "triangle and tet meshes do not support"
                  " the LOR preconditioner yet");
   }
   //mesh->ReorientTetMesh();

   // 5. Define a finite element space on the mesh. Here we use the Nedelec
   //    finite elements of the specified order.
   FiniteElementCollection *fec = new RT_FECollection(order-1, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 6. Check if the optimized version matches the given space
   if (perf && !sol_fes_t::Matches(*fespace))
   {
      cout << "The given order does not match the optimized parameter.\n"
           << "Recompile with suitable 'sol_p' value." << endl;
      delete fespace;
      delete fec;
      delete mesh;
      return 5;
   }
   // 7. Determine the list of true (i.e. conforming) essential boundary dofs.
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

   // 8. Set up the linear form b(.) which corresponds to the right-hand side
   //    of the FEM linear system, which in this case is (f,phi_i) where f is
   //    given by the function f_exact and phi_i are the basis functions in the
   //    finite element fespace.
   VectorFunctionCoefficient f(sdim, f_exact);
   LinearForm *b = new LinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   LinearForm *b1 = new LinearForm(fespace);
   b1->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b1->Assemble();

   // 9. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   GridFunction x(fespace);
   GridFunction x1(fespace);
   //VectorFunctionCoefficient E(sdim, E_exact);
   //x.ProjectCoefficient(E);
   x = 1.0;
   x1 = 1.0;
   // 10. Set up the bilinear form corresponding to the EM diffusion operator
   //     curl muinv curl + sigma I, by adding the curl-curl and the mass domain
   //     integrators.
   Coefficient *alpha = new ConstantCoefficient(1.0);
   Coefficient *beta = new ConstantCoefficient(1.0);
   BilinearForm *a = new BilinearForm(fespace);
   BilinearForm *a_h = new BilinearForm(fespace);
   // 11. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond)
   {
	   a->EnableStaticCondensation();
	   //MFEM_VERIFY(pc_choice != LOR,
		//		   "cannot use LOR preconditioner with static condensation");
   }
   
   cout << "Assembling the bilinear form ..." << endl;
   tic_toc.Clear();
   tic_toc.Start();

   // Pre-allocate sparsity assuming dense element matrices
   a_h->UsePrecomputedSparsity();
   
   HPCBilinearForm *a_hpc = NULL;
   Operator *a_oper = NULL;

   {
      a->UsePrecomputedSparsity();
      a->AddDomainIntegrator(new DivDivIntegrator(*alpha));
	  //a->AddDomainIntegrator(new VectorFEMassIntegrator(*beta));
      if (static_cond) { a->EnableStaticCondensation(); }
      a->Assemble();
   }

   {
      cout << "Assemble a_hpc" << endl;
      // High-performance assembly/evaluation using the templated operator type
      a_hpc = new HPCBilinearForm(integ_t(coeff_t(1.0)), *fespace);
      if (matrix_free)
      {
         a_hpc->Assemble(); // partial assembly
      }
      else
      {
         cout << "a_hpc.AssembleBilinearForm" << endl;
         a_hpc->AssembleBilinearForm(*a_h); // full matrix assembly
      }
   }
   tic_toc.Stop();
   cout << " done, " << tic_toc.RealTime() << "s." << endl;

   // 12. Solve the system A X = B with CG. In the standard case, use a simple
   //     symmetric Gauss-Seidel preconditioner.
   // Setup the operator matrix (if applicable)
   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   //A.PrintMatlab(std::cout);
   //A.PrintInfo(std::cout);
   
   SparseMatrix A1;
   Vector B1, X1;

   if (matrix_free)
   {
	   a_hpc->FormLinearSystem(ess_tdof_list, x1, *b1, a_oper, X1, B1);
	   cout << "Size of linear system: " << a_hpc->Height() << endl;
	   Vector BB(B1);
	   //BB.Print();
	   BB = 1.0;
	   BB.Add(-1.0, X);
	   //BB.Print();
	   Vector X_tmp(X);
	   A.Mult(BB, X_tmp);
	   std::cout<<"L_inf(X) = "<<X_tmp.Normlinf()<<std::endl;
	   Vector X_tmp1(X);
	   //X_tmp.Print();
	   a_oper->Mult(BB, X_tmp1);
	   //X_tmp1.Print();
	   std::cout<<"L_inf(X1) = "<<X_tmp.Normlinf()<<std::endl;
	   X_tmp1.Add(-1.0, X_tmp);
	   //X_tmp1.Print();
	   std::cout<<"L_inf(X1-X) = "<<X_tmp1.Normlinf()<<std::endl;
   }
   else
   {
	   std::cout<<"L_inf(A) = "<<A.MaxNorm()<<std::endl;
	   //A.Print(std::cout);
	   a_h->FormLinearSystem(ess_tdof_list, x1, *b1, A1, X1, B1);
	   std::cout<<"L_inf(A1) = "<<A1.MaxNorm()<<std::endl;
	   //A1.Print(std::cout);
	   A.Add(-1,A1);
	   //A.Print(std::cout);
	   std::cout<<"L_inf(A1-A) = "<<A.MaxNorm()<<std::endl;
	   //A.Print(std::cout);
	   A.PrintInfo(std::cout);
   }

   //if (perf && matrix_free)
   //{
//	   cout <<"Get here"<<endl;  
   //}
   //else
   //{
//	   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
//	   cout << "Size of linear system: " << A.Height() << endl;
//	   a_oper = &A;
   //}
//#ifndef MFEM_USE_SUITESPARSE
   // 10. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //     solve the system Ax=b with PCG.
   //GSSmoother M(A);
   //PCG(A, M, B, X, 1, 500, 1e-12, 0.0);
//#else
   // 10. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
//   UMFPackSolver umf_solver;
//   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
//   umf_solver.SetOperator(A);
//   umf_solver.Mult(B, X);
//#endif

   // 11. Recover the solution as a finite element grid function.
//   a->RecoverFEMSolution(X, *b, x);

   // 12. Compute and print the L^2 norm of the error.
//   cout << "\n|| E_h - E ||_{L^2} = " << x.ComputeL2Error(E) << '\n' << endl;

   // 13. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
  // {
  //    ofstream mesh_ofs("refined.mesh");
  //    mesh_ofs.precision(8);
  //    mesh->Print(mesh_ofs);
  //    ofstream sol_ofs("sol.gf");
  //    sol_ofs.precision(8);
  //    x.Save(sol_ofs);
  // }

   // 14. Send the solution by socket to a GLVis server.
  // if (visualization)
  // {
  //    char vishost[] = "localhost";
  //    int  visport   = 19916;
  //    socketstream sol_sock(vishost, visport);
  //    sol_sock.precision(8);
  //    sol_sock << "solution\n" << *mesh << x << flush;
  // }

   // 15. Free the used memory.
   //delete a;
   //delete sigma;
   //delete muinv;
   //delete b;
   //delete fespace;
   //delete fec;
   //delete mesh;

   return 0;
}


void F_exact(const Vector &p, Vector &F)
{
	int dim = p.Size();

	double x = p(0);
	double y = p(1);
	// double z = (dim == 3) ? p(2) : 0.0;

	F(0) = cos(kappa*x)*sin(kappa*y);
	F(1) = cos(kappa*y)*sin(kappa*x);
	if (dim == 3)
	{
		F(2) = 0.0;
	}
}

void f_exact(const Vector &p, Vector &f)
{
	int dim = p.Size();

	double x = p(0);
	double y = p(1);
	// double z = (dim == 3) ? p(2) : 0.0;

	double temp = 1 + 2*kappa*kappa;

	f(0) = temp*cos(kappa*x)*sin(kappa*y);
	f(1) = temp*cos(kappa*y)*sin(kappa*x);
	if (dim == 3)
	{
		f(2) = 0;
	}
}
