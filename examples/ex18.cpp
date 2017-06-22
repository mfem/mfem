// This file is part of CEED. For more details, see exascaleproject.org.
//
//                    MFEM Bake-off problem 1, version 1
//
// Compile with: ...
//
// Sample runs:  ...
//
// Description:  ... based on mfem/miniapps/performance/ex1p.cpp ...

// Comment/uncomment to disable/enable the use of add_mult_mass_{quad,hex}(...):
#define MFEM_EXPERIMENT_1

#ifndef PROBLEM
#define PROBLEM 0
#endif

#define MFEM_EXPERIMENT_1_PROBLEM PROBLEM

#ifdef MFEM_USE_CEED
#include "ceed.h"
#endif

#include "mfem-performance.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#ifndef GEOM
#define GEOM Geometry::SQUARE
#endif

#ifndef MESH_P
#define MESH_P 1
#endif

#ifndef SOL_P
#define SOL_P 3
#endif

#ifndef IR_ORDER
#define IR_ORDER 0
#endif

// Define template parameters for optimized build.
const Geometry::Type geom     = GEOM;      // mesh elements  (default: hex)
const int            mesh_p   = MESH_P;    // mesh curvature (default: 3)
const int            sol_p    = SOL_P;     // solution order (default: 3)
const int            rdim     = Geometry::Constants<geom>::Dimension;
const int            ir_order = IR_ORDER ? IR_ORDER : 2*(sol_p+2)-1;

// Static mesh type
typedef H1_FiniteElement<geom,mesh_p>         mesh_fe_t;
typedef H1_FiniteElementSpace<mesh_fe_t>      mesh_fes_t;
typedef TMesh<mesh_fes_t>                     mesh_t;

// Static solution finite element space type
typedef H1_FiniteElement<geom,sol_p>          sol_fe_t;
typedef H1_FiniteElementSpace<sol_fe_t>       sol_fes_t;

// Static quadrature, coefficient and integrator types
typedef TIntegrationRule<geom,ir_order>       int_rule_t;
typedef TConstantCoefficient<>                coeff_t;
#if (PROBLEM == 0)
typedef TIntegrator<coeff_t,TDiffusionKernel> integ_t;
#else
typedef TIntegrator<coeff_t,TMassKernel> integ_t;
#endif

// Static bilinear form type, combining the above types
typedef TBilinearForm<mesh_t,sol_fes_t,int_rule_t,integ_t> HPCBilinearForm;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/fichera.mesh";
   int ser_ref_levels = -1;
   int par_ref_levels = +1;
   Array<int> nxyz;
   int order = sol_p;
   const char *basis_type = "G"; // Gauss-Lobatto
   bool static_cond = false;
   bool perf = true;
   bool matrix_free = true;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&nxyz, "-c", "--cartesian-partitioning",
                  "Use Cartesian partitioning.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&basis_type, "-b", "--basis-type",
                  "Basis: G - Gauss-Lobatto, P - Positive, U - Uniform");
   args.AddOption(&perf, "-perf", "--hpc-version", "-std", "--standard-version",
                  "Enable high-performance, tensor-based, assembly/evaluation.");
   args.AddOption(&matrix_free, "-mf", "--matrix-free", "-asm", "--assembly",
                  "Use matrix-free evaluation or efficient matrix assembly in "
                  "the high-performance version.");
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

   // See class BasisType in fem/fe_coll.hpp for available basis types
   int basis = BasisType::GetType(basis_type[0]);
   cout << "Using " << BasisType::Name(basis) << " basis ..." << endl;

   // 2. Read the (serial) mesh from the given mesh file. We can handle
   //    triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

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

   // 4. Refine the serial mesh on to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      ref_levels = (ser_ref_levels != -1) ? ser_ref_levels : ref_levels;
      for (int l = 0; l < ref_levels; l++)
      {
         cout << "Serial refinement: level " << l << " -> level " << l+1
              << " ..." << flush;
         mesh->UniformRefinement();
         cout << " done." << endl;
      }
   }
   if (!perf && mesh->NURBSext)
   {
      const int new_mesh_p = std::min(sol_p, mesh_p);
      cout << "NURBS mesh: switching the mesh curvature to be "
           << "min(sol_p, mesh_p) = " << new_mesh_p << " ..." << endl;
      mesh->SetCurvature(new_mesh_p, false, -1, Ordering::byNODES);
   }

   // 5. Define a serial finite element space on the serial mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
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
   long size = fespace->GetTrueVSize();
   cout << "Number of finite element unknowns: " << size << endl;

   Mesh *mesh_lor = NULL;
   FiniteElementCollection *fec_lor = NULL;
   FiniteElementSpace *fespace_lor = NULL;

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

   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Set up the linear form b(.) which corresponds to the
   //     right-hand side of the FEM linear system, which in this case is
   //     (1,phi_i) where phi_i are the basis functions in fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 9. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 10. Set up the parallel bilinear form a(.,.) on the finite element space
   //     that will hold the matrix corresponding to the Laplacian operator.
   BilinearForm *a = new BilinearForm(fespace);
   BilinearForm *a_pc = NULL;

   // 11. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond)
   {
      a->EnableStaticCondensation();
   }

   cout << "Assembling the local matrix ..." << flush;
   tic_toc.Clear();
   tic_toc.Start();
   // Pre-allocate sparsity assuming dense element matrices; the actual memory
   // allocation happens when a->Assemble() is called.
   a->UsePrecomputedSparsity();

   HPCBilinearForm *a_hpc = NULL;
   Operator *a_oper = NULL;

   if (!perf)
   {
      // Standard assembly using a diffusion domain integrator
      a->AddDomainIntegrator(new DiffusionIntegrator(one));
      a->Assemble();
   }
   else
   {
      // High-performance assembly/evaluation using the templated operator type
      a_hpc = new HPCBilinearForm(integ_t(coeff_t(1.0)), *fespace);
      if (matrix_free)
      {
         a_hpc->Assemble(); // partial assembly
      }
      else
      {
         a_hpc->AssembleBilinearForm(*a); // full matrix assembly
      }
   }
   tic_toc.Stop();
   double my_rt;
   my_rt = tic_toc.RealTime();

   cout << " done, " << my_rt <<  " s." << endl;
   cout << "\n\"DOFs/sec\" in assembly: "
        << 1e-6*size/my_rt << "million.\n" << endl;

   // 12. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
   //     preconditioner from hypre.

   // Setup the operator matrix (if applicable)
   SparseMatrix A;
   Vector B, X;
   cout << "FormLinearSystem() ..." << endl;

   tic_toc.Clear();
   tic_toc.Start();

   if (perf && matrix_free)
   {
      a_hpc->FormLinearSystem(ess_tdof_list, x, *b, a_oper, X, B);
      cout << "Size of linear system: " << size << endl;
   }
   else
   {
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
      a_oper = &A;
   }

   tic_toc.Stop();
   my_rt = tic_toc.RealTime();

   cout << "FormLinearSystem() ... done, " << my_rt << endl;
   cout << "\n\"DOFs/sec\" in FormLinearSystem(): "
        << 1e-6*size/my_rt << " million.\n" << endl;

   // Solve with CG or PCG, depending if the matrix A_pc is available
   CGSolver *cg;
   cg = new CGSolver();
   cg->SetRelTol(1e-6);
   cg->SetMaxIter(50);
   cg->SetPrintLevel(3);
   cg->SetOperator(*a_oper);

   tic_toc.Clear();
   tic_toc.Start();

   cg->Mult(B, X);

   tic_toc.Stop();
   my_rt = tic_toc.RealTime();

   // Note: In the pcg algorithm, the number of operator Mult() calls is
   //       N_iter and the number of preconditioner Mult() calls is N_iter+1.
   cout << "Total CG time:    " << my_rt << " sec."
        << endl;
   cout << "Time per CG step: " << my_rt / cg->GetNumIterations() << " sec." 
        << endl;
   cout << "\n\"DOFs/sec\" in CG: "
        << 1e-6*size*cg->GetNumIterations()/my_rt << " million."
        << endl;
   cout << "\nNumber of iterations in CG: "
        << cg->GetNumIterations() << endl;

   // 13. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   if (perf && matrix_free)
   {
      a_hpc->RecoverFEMSolution(X, *b, x);
   }
   else
   {
      a->RecoverFEMSolution(X, *b, x);
   }

   // 14. Save the refined mesh and the solution. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   if (false)
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6);
      sol_name << "sol." << setfill('0') << setw(6);

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 16. Free the used memory.
   delete a;
   delete a_hpc;
   if (a_oper != &A) { delete a_oper; }
   delete a_pc;
   delete b;
   delete fespace;
   delete fespace_lor;
   delete fec_lor;
   delete mesh_lor;
   if (order > 0) { delete fec; }
   delete mesh;
   delete cg;

   return 0;
}
