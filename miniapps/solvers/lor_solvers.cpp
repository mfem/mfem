#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>

#include "lor_mms.hpp"

using namespace std;
using namespace mfem;

int problem = 0;
int dim = 0;

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../data/star.mesh";
   int ref_levels = 1;
   int order = 3;
   const char *fe = "h";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. Possible values are 0 or 1.");
   args.AddOption(&fe, "-fe", "--fe-type",
                  "FE type. h for H1, n for Hcurl, r for Hdiv, l for L2");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   bool H1 = false, ND = false, RT = false, L2 = false;
   if (string(fe) == "h") { H1 = true; }
   else if (string(fe) == "n") { ND = true; }
   else if (string(fe) == "r") { RT = true; }
   else if (string(fe) == "l") { L2 = true; }
   else { MFEM_ABORT("Bad FE type. Must be 'h', 'n', 'r', or 'l'."); }

   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }

   int b1 = BasisType::GaussLobatto, b2 = BasisType::Integrated;
   unique_ptr<FiniteElementCollection> fec;
   if (H1) { fec.reset(new H1_FECollection(order, dim, b1)); }
   else if (ND) { fec.reset(new ND_FECollection(order, dim, b1, b2)); }
   else if (RT) { fec.reset(new RT_FECollection(order-1, dim, b1, b2)); }
   else { fec.reset(new L2_FECollection(order, dim, b1)); }

   FiniteElementSpace fes(&mesh, fec.get());
   cout << "Number of DOFs: " << fes.GetTrueVSize() << endl;

   Array<int> ess_dofs;
   if (!L2) { fes.GetBoundaryTrueDofs(ess_dofs); }

   BilinearForm a(&fes);
   if (H1 || L2)
   {
      a.AddDomainIntegrator(new MassIntegrator);
      a.AddDomainIntegrator(new DiffusionIntegrator);
   }
   else
   {
      a.AddDomainIntegrator(new VectorFEMassIntegrator);
   }

   if (ND) { a.AddDomainIntegrator(new CurlCurlIntegrator); }
   else if (RT) { a.AddDomainIntegrator(new DivDivIntegrator); }
   else if (L2)
   {
      double kappa = (order+1)*(order+1);
      a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(-1.0, kappa));
      a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(-1.0, kappa));
   }
   // TODO: L2 diffusion not implemented with partial assemble
   if (!L2) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.Assemble();

   LinearForm b(&fes);
   FunctionCoefficient f_coeff(f);
   VectorFunctionCoefficient f_vec_coeff(dim, f_vec);
   if (H1 || L2)
   {
      b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
   }
   else
   {
      b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_vec_coeff));
   }
   b.Assemble();

   GridFunction x(&fes);
   x = 0.0;

   Vector X, B;
   OperatorHandle A;
   a.FormLinearSystem(ess_dofs, x, b, A, X, B);

#ifdef MFEM_USE_SUITESPARSE
   LORSolver<UMFPackSolver> solv_lor(a, ess_dofs);
#else
   LORSolver<GSSmoother> solv_lor(a, ess_dofs);
#endif

   CGSolver cg;
   cg.SetAbsTol(0.0);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(100);
   cg.SetPrintLevel(1);
   cg.SetOperator(*A);
   cg.SetPreconditioner(solv_lor);
   cg.Mult(B, X);

   a.RecoverFEMSolution(X, b, x);

   // Save the solution and mesh to disk. The output can be viewed using GLVis
   // as follows: "glvis -m mesh.mesh -g sol.gf"
   x.Save("sol.gf");
   mesh.Save("mesh.mesh");

   // Also save the solution for visualization using ParaView
   ParaViewDataCollection dc("LOR", &mesh);
   dc.SetPrefixPath("ParaView");
   dc.SetHighOrderOutput(true);
   dc.SetLevelsOfDetail(order);
   dc.RegisterField("u", &x);
   dc.SetCycle(0);
   dc.SetTime(0.0);
   dc.Save();

   return 0;
}
