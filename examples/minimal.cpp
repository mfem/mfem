#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void ExtractComponent(const GridFunction &phi, GridFunction &phi_i, int d)
{
   const FiniteElementSpace *fes = phi_i.FESpace();
   // ASSUME phi IS ORDERED byNODES!
   int ndof = fes->GetNDofs();
   for (int i = 0; i < ndof; i++)
   {
      int j = d*ndof + i;
      phi_i[i] = phi[j];
   }
}

void AddToComponent(GridFunction &phi, const GridFunction &phi_i, int d,
                    double alpha=1.0)
{
   const FiniteElementSpace *fes = phi_i.FESpace();
   // ASSUME phi IS ORDERED byNODES!
   int ndof = fes->GetNDofs();
   for (int i = 0; i < ndof; i++)
   {
      int j = d*ndof + i;
      phi[j] += alpha*phi_i[i];
   }
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/cylinder.mesh";
   int niter = 4;
   int order = 4;
   int ref_levels = 1;
   bool visualization;
   bool vis_mesh = false;
   bool vis_solution = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&ref_levels, "-r", "--ref-levels", "Refinement");
   args.AddOption(&niter, "-n", "--niter", "Number of iterations");
   args.AddOption(&vis_solution, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_mesh, "-vm", "--vis-mesh", "-no-vm",
                  "--no-vis-mesh",
                  "Enable or disable final mesh visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   visualization = vis_solution|vis_mesh;

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   const int sdim = mesh.SpaceDimension();
   mesh.SetCurvature(order, false, sdim, Ordering::byNODES);
   const int dim = mesh.Dimension();

   // 3. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }

   // 4. Define a finite element space on the mesh.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   if (mesh.bdr_attributes.Size())
   {
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   GridFunction *phi = mesh.GetNodes();
   GridFunction phi_new(phi->FESpace());
   std::vector<GridFunction> phi_i(3);
   for (int i=0; i<sdim; ++i) { phi_i[i].SetSpace(&fespace); }

   ConstantCoefficient zero(0.0);
   ConstantCoefficient one(1.0);

   GridFunction x(&fespace), b(&fespace);
   x = 0.0;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost, visport);
      sol_sock.precision(8);
   }

   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));

   BilinearForm a0(&fespace);
   a0.AddDomainIntegrator(new DiffusionIntegrator(one));

   OperatorPtr A0;
   for (int iiter=0; iiter<niter; ++iiter)
   {
      a.Assemble();
      a0.Assemble();
      a0.FormSystemMatrix(ess_tdof_list, A0);

#ifdef MFEM_USE_SUITESPARSE
      UMFPackSolver A0_inv;
      A0_inv.SetOperator(*A0);
#endif
      phi_new = *phi;
      for (int i=0; i<sdim; ++i)
      {
         ExtractComponent(*phi, phi_i[i], i);
         a.Mult(phi_i[i], b);
         b.ProjectBdrCoefficient(zero, ess_bdr);
#ifndef MFEM_USE_SUITESPARSE
         GSSmoother M((SparseMatrix&)(*A0));
         PCG(*A0, M, b, x, 1, 200, 1e-12, 0.0);
#else
         A0_inv.Mult(b, x);
#endif
         AddToComponent(phi_new, x, i, -1.0);
         phi_i[i] = x;
      }

      if (vis_solution)
      {
         sol_sock << "solution\n" << mesh << phi_i[0] << flush;
         sol_sock << "pause\n" << flush;
      }
      *phi = phi_new;
   }

   if (vis_mesh) { sol_sock << "mesh\n" << mesh << flush; }

   // 14. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);
   return 0;
}
