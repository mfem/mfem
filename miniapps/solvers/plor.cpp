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
   MPI_Session mpi;

   const char *mesh_file = "../data/star.mesh";
   int ser_ref_levels = 1, par_ref_levels = 1;
   int order = 3;
   const char *fe = "h";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
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
   else { MFEM_ABORT("Bad FE type. Must be 'n' or 'r'."); }

   Mesh serial_mesh(mesh_file, 1, 1);
   dim = serial_mesh.Dimension();
   for (int l = 0; l < ser_ref_levels; l++) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   for (int l = 0; l < par_ref_levels; l++) { mesh.UniformRefinement(); }
   serial_mesh.Clear();

   int b1 = BasisType::GaussLobatto, b2 = BasisType::Integrated;
   unique_ptr<FiniteElementCollection> fec;
   if (H1) { fec.reset(new H1_FECollection(order, dim, b1)); }
   else if (ND) { fec.reset(new ND_FECollection(order, dim, b1, b2)); }
   else if (RT) { fec.reset(new RT_FECollection(order-1, dim, b1, b2)); }
   else { fec.reset(new L2_FECollection(order, dim, b1)); }

   ParFiniteElementSpace fes(&mesh, fec.get());
   HYPRE_Int ndofs = fes.GlobalTrueVSize();
   if (mpi.Root()) { cout << "Number of DOFs: " << ndofs << endl; }

   Array<int> ess_dofs;
   if (!L2) { fes.GetBoundaryTrueDofs(ess_dofs); }

   ParBilinearForm a(&fes);
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

   ParLinearForm b(&fes);
   if (H1 || L2)
   {
      FunctionCoefficient f_coeff(f_exact);
      b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
   }
   else
   {
      VectorFunctionCoefficient f_coeff(dim, f_exact_vec);
      b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_coeff));
   }
   b.Assemble();

   ParGridFunction x(&fes);
   x = 0.0;

   Vector X, B;
   OperatorHandle A;
   a.FormLinearSystem(ess_dofs, x, b, A, X, B);

   ParLOR lor(a, ess_dofs);
   ParFiniteElementSpace &fes_lor = lor.GetParFESpace();

   unique_ptr<Solver> solv_lor;
   if (H1 || L2)
   {
      solv_lor.reset(new LORSolver<HypreBoomerAMG>(lor));
   }
   else if (RT && dim == 3)
   {
      solv_lor.reset(new LORSolver<HypreADS>(lor, &fes_lor));
   }
   else
   {
      solv_lor.reset(new LORSolver<HypreAMS>(lor, &fes_lor));
   }

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetAbsTol(0.0);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(100);
   cg.SetPrintLevel(1);
   cg.SetOperator(*A);
   cg.SetPreconditioner(*solv_lor);
   cg.Mult(B, X);

   a.RecoverFEMSolution(X, b, x);

   // Save the solution and mesh to disk. The output can be viewed using GLVis
   // as follows: "glvis -np <np> -m mesh -g sol"
   x.Save("sol");
   mesh.Save("mesh");

   // Also save the solution for visualization using ParaView
   ParaViewDataCollection dc("PLOR", &mesh);
   dc.SetPrefixPath("ParaView");
   dc.SetHighOrderOutput(true);
   dc.SetLevelsOfDetail(order);
   dc.RegisterField("u", &x);
   dc.SetCycle(0);
   dc.SetTime(0.0);
   dc.Save();

   return 0;
}
