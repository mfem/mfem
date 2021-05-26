//                       MFEM Example 11 - Serial Version
//
// Compile with: make ex11
//
// Sample runs:  ex11 -m ../data/square-disc.mesh
//               ex11 -m ../data/star.mesh
//               ex11 -m ../data/star-mixed.mesh
//               ex11 -m ../data/periodic-annulus-sector.msh
//               ex11 -m ../data/square-disc-p2.vtk -o 2
//               ex11 -m ../data/square-disc-p3.mesh -o 3
//               ex11 -m ../data/square-disc-nurbs.mesh -o -1
//               ex11 -m ../data/disc-nurbs.mesh -o -1 -n 20
//               ex11 -m ../data/star-surf.mesh
//               ex11 -m ../data/square-disc-surf.mesh
//               ex11 -m ../data/inline-segment.mesh
//               ex11 -m ../data/inline-quad.mesh
//               ex11 -m ../data/inline-tri.mesh
//               ex11 -m ../data/amr-quad.mesh
//               ex11 -m ../data/amr-hex.mesh
//               ex11 -m ../data/mobius-strip.mesh -n 8
//
// Description:  This example code demonstrates the use of MFEM to solve the
//               eigenvalue problem -Delta u = lambda u with homogeneous
//               Dirichlet boundary conditions.
//
//               We compute a number of the lowest eigenmodes by discretizing
//               the Laplacian and Mass operators using a FE space of the
//               specified order, or an isoparametric/isogeometric space if
//               order < 1 (quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of the ARPACK eigenvalue solver
//               (regular inverse mode). Reusing a single GLVis visualization
//               window for multiple eigenfunctions is also illustrated.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int ser_ref_levels = 3;
   int order = 1;
   int nev = 5;
   double dbc_eig = 1e3;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&nev, "-n", "--num-eigs",
                  "Number of desired eigenmodes.");
   args.AddOption(&dbc_eig, "-d", "--dbc-eig",
                  "Eigenvalues associated with Dirichlet BC "
                  "(should be larger than the maximum desired eigenvalue).");
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

   // 2. Read the (serial) mesh from the given mesh file on all processors. We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();
   int dim = mesh->Dimension();

   // 3. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement (2 by default, or
   //    specified on the command line with -rs).
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 4. Define a finite element space on the mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   int size = fespace->GetVSize();

   cout << "Number of unknowns: " << size << endl;

   // 5. Set up the parallel bilinear forms a(.,.) and m(.,.) on the finite
   //    element space. The first corresponds to the Laplacian operator -Delta,
   //    while the second is a simple mass matrix needed on the right hand side
   //    of the generalized eigenvalue problem below. The boundary conditions
   //    are implemented by elimination with special values on the diagonal to
   //    shift the Dirichlet eigenvalues out of the computational range. After
   //    serial and parallel assembly we extract the corresponding parallel
   //    matrices A and M.
   ConstantCoefficient one(1.0);
   Array<int> ess_bdr;
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
      ess_bdr = 1;
   }

   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   if (mesh->bdr_attributes.Size() == 0)
   {
      // Add a mass term if the mesh has no boundary, e.g. periodic mesh or
      // closed surface.
      a->AddDomainIntegrator(new MassIntegrator(one));
   }
   a->Assemble();
   if (mesh->bdr_attributes.Size() != 0)
   {
      a->EliminateEssentialBCDiag(ess_bdr, dbc_eig);
   }
   a->Finalize();

   BilinearForm *m = new BilinearForm(fespace);
   m->AddDomainIntegrator(new MassIntegrator(one));
   m->Assemble();
   if (mesh->bdr_attributes.Size() != 0)
   {
      // shift the eigenvalue corresponding to eliminated dofs to a large value
      m->EliminateEssentialBCDiag(ess_bdr, 1.0);
   }
   m->Finalize();

   // 6. Define and configure the ARPACK eigensolver
   ArPackSym * arpack = new ArPackSym();
   Solver    * solver = NULL;

#ifndef MFEM_USE_SUITESPARSE
   // 7. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system A X = B with PCG.
   cout << "Building CGSolver" << endl;
   GSSmoother M(m->SpMat());
   CGSolver * cg_solver = new CGSolver;
   cg_solver->SetPreconditioner(M);
   cg_solver->SetRelTol(1.0e-12);
   solver = cg_solver;
#else
   // 7. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   cout << "Building UMFPackSolver" << endl;
   UMFPackSolver * umf_solver = new UMFPackSolver;
   umf_solver->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   solver = umf_solver;
#endif
   solver->SetOperator(m->SpMat());

   arpack->SetNumModes(nev);
   arpack->SetMaxIter(400);
   arpack->SetTol(1e-8);
   arpack->SetMode(2);
   arpack->SetPrintLevel(2);

   arpack->SetOperator(*a);
   arpack->SetMassMatrix(*m);
   arpack->SetSolver(*solver);

   // 8. Compute the eigenmodes and extract the array of eigenvalues. Define a
   //    parallel grid function to represent each of the eigenmodes returned by
   //    the solver.
   Array<double> eigenvalues;
   arpack->Solve();
   arpack->GetEigenvalues(eigenvalues);

   cout << endl;
   std::ios::fmtflags old_fmt = cout.flags();
   cout.setf(std::ios::scientific);
   std::streamsize old_prec = cout.precision(14);
   for (int i=0; i<nev; i++)
   {
      cout << "Eigenvalue lambda   " << eigenvalues[i] << endl;
   }
   cout.precision(old_prec);
   cout.flags(old_fmt);
   cout << endl;

   GridFunction x(fespace);

   // 9. Save the refined mesh and the modes in parallel. This output can be
   //    viewed later using GLVis: "glvis -np <np> -m mesh -g mode".
   {
      ostringstream mesh_name, mode_name;
      mesh_name << "ex11.mesh";

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      for (int i=0; i<nev; i++)
      {
         // convert eigenvector from HypreParVector to ParGridFunction
         x = arpack->GetEigenvector(i);

         mode_name << "mode_" << setfill('0') << setw(2) << i;

         ofstream mode_ofs(mode_name.str().c_str());
         mode_ofs.precision(8);
         x.Save(mode_ofs);
         mode_name.str("");
      }
   }

   // 10. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mode_sock(vishost, visport);
      mode_sock.precision(8);

      for (int i=0; i<nev; i++)
      {
         cout << "Eigenmode " << i+1 << '/' << nev
              << ", Lambda = " << eigenvalues[i] << endl;

         // convert eigenvector from HypreParVector to ParGridFunction
         x = arpack->GetEigenvector(i);

         mode_sock << "solution\n" << *mesh << x << flush
                   << "window_title 'Eigenmode " << i+1 << '/' << nev
                   << ", Lambda = " << eigenvalues[i] << "'" << endl;

         char c;
         cout << "press (q)uit or (c)ontinue --> " << flush;
         cin >> c;

         if (c != 'c')
         {
            break;
         }
      }
      mode_sock.close();
   }

   // 11. Free the used memory.
   delete arpack;
   delete solver;
   delete m;
   delete a;

   delete fespace;
   if (order > 0)
   {
      delete fec;
   }
   delete mesh;

   return 0;
}