#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "./cylinder.mesh";
   int niter = 4;
   int order = 4;
   int ref_levels = 2;
   bool vis_mesh = true;
   bool vis_solution = true;
   const char vishost[] = "localhost";
   const int  visport   = 19916;
   socketstream sol_sock;
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
   if (vis_solution || vis_mesh)
   {
      sol_sock.open(vishost, visport);
      sol_sock.precision(8);
   }
   const int generate_edges = 1;
   const int refine = 1;
   Mesh mesh(mesh_file, generate_edges, refine);
   const int sdim = mesh.SpaceDimension();
   mesh.SetCurvature(order, false, sdim, Ordering::byNODES);
   const int mdim = mesh.Dimension();
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }
   const H1_FECollection fec(order, mdim);
   FiniteElementSpace *fes = new FiniteElementSpace(&mesh, &fec, sdim);
   cout << "Number of finite element unknowns: " << fes->GetTrueVSize() << endl;
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   GridFunction *phi = mesh.GetNodes();
   ConstantCoefficient one(1.0);
   GridFunction x(fes), b(fes);
   OperatorPtr A;
   BilinearForm a(fes);
   a.AddDomainIntegrator(new VectorDiffusionIntegrator(one));
   if (vis_solution)
   {
      sol_sock << "mesh\n" << mesh << flush;
      sol_sock << "pause\n" << flush;
   }
   for (int iiter=0; iiter<niter; ++iiter)
   {
      a.Assemble();
      Vector B, X;
      x = *phi; // should only copy the BC
      b = 0.0;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
      GSSmoother M(static_cast<SparseMatrix&>(*A));
      PCG(*A, M, B, X, 1, 2000, 1e-12, 0.0);
      a.RecoverFEMSolution(X, b, x);
      *phi = x;

      if (vis_solution)
      {
         sol_sock << "mesh\n" << mesh << flush;
         sol_sock << "pause\n" << flush;
      }
      a.Update();
   }
   if (vis_mesh) { sol_sock << "mesh\n" << mesh << flush; }
   return 0;
}
