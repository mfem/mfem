//                       MFEM Example 13
//
// Compile with: make ex3p
//
// Sample runs:  ex13 -m ../data/beam-tet.mesh
//               ex13 -m ../data/beam-hex.mesh
//               ex13 -m ../data/escher.mesh
//               ex13 -m ../data/fichera.mesh
//               ex13 -m ../data/fichera-q2.vtk
//               ex13 -m ../data/fichera-q3.mesh
//               ex13 -m ../data/beam-hex-nurbs.mesh
//
// Description:  This example code solves a simple 3D electromagnetic
//               eigenmode problem corresponding to the second order
//               Maxwell equation curl curl E = lambda E with boundary
//               condition E x n = 0. We discretize with Nedelec finite
//               elements.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl and the (vector finite element) mass
//               bilinear form, as well as the use of the ARPACK eigenmode
//               solver for symmetric matrices using the shift-invert mode.
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/beam-tet.mesh";
   int order = 1;
   int nev = 5;
   int sr = 3;
   double sigma = 11.0;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&nev, "-n", "--num-eigs",
                  "Number of desired eigenmodes.");
   args.AddOption(&sr, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&sigma, "-s", "--shift",
                  "Average of the desired eigenvalue range.");
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

   // 2. Read the mesh from the given mesh file.  We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes
   //    with the same code.
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

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement.
   {
      int ref_levels = sr;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 4. Define a finite element space on the mesh. Here we use the lowest
   //    order Nedelec finite elements, but we can easily switch
   //    to higher-order spaces by changing the value of p.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   int size = fespace->GetVSize();

   cout << "Number of unknowns: " << size << endl;
   cout << "Number of boundary attributes: " << mesh->bdr_attributes.Max()
        << endl;

   // 5. Set up the parallel bilinear form corresponding to the EM diffusion
   //    operator curl muinv curl - sigma I, by adding the curl-curl and the
   //    mass domain integrators and finally imposing homogeneous Dirichlet
   //    boundary conditions. The boundary conditions are implemented by
   //    marking all the boundary attributes from the mesh as essential
   //    (Dirichlet). After serial and parallel assembly we extract the
   //    parallel matrices A and M.
   Coefficient *muinv    = new ConstantCoefficient(1.0);
   Coefficient *negSigma = new ConstantCoefficient(-sigma);

   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*negSigma));
   a->Assemble();
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   a->EliminateEssentialBC(ess_bdr);
   a->Finalize();

   BilinearForm *m = new BilinearForm(fespace);
   m->AddDomainIntegrator(new VectorFEMassIntegrator());
   m->Assemble();
   m->EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
   m->Finalize();

   // 6. Define a parallel grid function to approximate each of the
   // eigenmodes returned by the solver.  Use this as a template to
   // create a special multi-vector object needed by the eigensolver
   // which is then initialized with random values.
   GridFunction x(fespace);
   x = 0.0;

   // 7. Define and configure the ARPACK eigensolver and a GMRES
   //    solver to be used within the eigensolver.
   ArPackSym * arpack = new ArPackSym();
   Solver    * solver = NULL;
   if ( false )
   {
      GMRESSolver *  gmres = new GMRESSolver();

      gmres->SetOperator(*a);
      gmres->SetRelTol(1e-8);
      gmres->SetMaxIter(1000);
      gmres->SetPrintLevel(0);
      solver = gmres;
   }
   else
   {
#ifndef MFEM_USE_SUITESPARSE
      cout << "Building MINRESSolver" << endl;
      MINRESSolver * minres = new MINRESSolver();

      minres->SetRelTol(1e-12);
      minres->SetMaxIter(1000);
      minres->SetPrintLevel(0);
      solver = minres;
#else
      // 7. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      cout << "Building UMFPackSolver" << endl;
      UMFPackSolver * umf_solver = new UMFPackSolver;
      umf_solver->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      solver = umf_solver;
#endif
   }
   solver->SetOperator(a->SpMat());

   arpack->SetNumModes(nev);
   arpack->SetMaxIter(400);
   arpack->SetTol(1e-8);
   arpack->SetShift(sigma);
   arpack->SetMode(3);
   arpack->SetPrintLevel(2);

   arpack->SetOperator(*a);
   arpack->SetMassMatrix(*m);
   arpack->SetSolver(*solver);

   // Obtain the eigenvalues and eigenvectors
   Array<double> eigenvalues(nev);
   eigenvalues = -1.0;

   // arpack->Solve(eigenvalues, *eigenvectors);
   arpack->Solve();

   arpack->GetEigenvalues(eigenvalues);

   cout << endl;
   std::ios::fmtflags old_fmt = cout.flags();
   cout.setf(std::ios::scientific);
   std::streamsize old_prec = cout.precision(14);
   for (int i=0; i<min(nev,eigenvalues.Size()); i++)
   {
      cout << "Eigenvalue lambda   " << eigenvalues[i] << endl;
   }
   cout.precision(old_prec);
   cout.flags(old_fmt);
   cout << endl;

   VisItDataCollection visit_dc("Example13", mesh);
   GridFunction ** mode = new GridFunction*[min(nev,eigenvalues.Size())];
   for (int i=0; i<min(nev,eigenvalues.Size()); i++)
   {
      mode[i] = new GridFunction(fespace);
      *mode[i] = arpack->GetEigenvector(i);

      ostringstream modeName;
      modeName << "mode_" << setfill('0') << setw(2) << i;
      visit_dc.RegisterField(modeName.str().c_str(),mode[i]);
   }
   visit_dc.Save();

   // 8. Save the refined mesh and the modes. This output can
   //    be viewed later using GLVis: "glvis -m mesh -g mode".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      for (int i=0; i<min(nev,eigenvalues.Size()); i++)
      {
         x = arpack->GetEigenvector(i);

         ostringstream modeName;
         modeName << "mode_" << setfill('0') << setw(2) << i;

         ofstream mode_ofs(modeName.str().c_str());
         mode_ofs.precision(8);
         x.Save(mode_ofs);
         modeName.str("");
      }
   }

   // 9. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mode_sock(vishost, visport);
      mode_sock.precision(8);

      for (int i=0; i<min(nev,eigenvalues.Size()); i++)
      {
         x = arpack->GetEigenvector(i);

         mode_sock << "solution\n" << *mesh << x << flush;

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

   // 10. Free the used memory.
   delete a;
   delete m;
   delete negSigma;
   delete muinv;
   delete arpack;
   delete solver;
   // delete X;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}
