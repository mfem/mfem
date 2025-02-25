// Hybrid mixed method

#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   string mesh_file = "../data/inline-quad.mesh";
   int order = 1;
   int ref = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&ref, "-r", "--refine", "Refinement levels");
   args.ParseCheck();

   Mesh mesh(mesh_file);
   for (int i = 0; i < ref; ++i) { mesh.UniformRefinement(); }

   BrokenHdivFECollection fec_broken_rt(order - 1, mesh.Dimension());
   L2_FECollection fec_l2(order - 1, mesh.Dimension());
   DG_Interface_FECollection fec_trace(order - 1, mesh.Dimension());

   FiniteElementSpace S_h(&mesh, &fec_broken_rt);
   FiniteElementSpace V_h(&mesh, &fec_l2);
   FiniteElementSpace M_h(&mesh, &fec_trace);

   cout << "S_h size: " << S_h.GetTrueVSize() << '\n';
   cout << "V_h size: " << V_h.GetTrueVSize() << '\n';
   cout << "M_h size: " << M_h.GetTrueVSize() << '\n';

   BilinearForm mass(&S_h);
   mass.AddDomainIntegrator(new VectorFEMassIntegrator);
   mass.Assemble();
   mass.Finalize();

   MixedBilinearForm div(&S_h, &V_h);
   div.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   div.Assemble();
   div.Finalize();

   MixedBilinearForm trace(&M_h, &S_h);
   trace.AddTraceFaceIntegrator(new NormalTraceJumpIntegrator);
   trace.Assemble();
   trace.Finalize();

   Array<int> empty;
   Array<int> ess_trace_dofs;
   M_h.GetBoundaryTrueDofs(ess_trace_dofs);

   std::cout << "Boundary DOFs: " << ess_trace_dofs.Size() << '\n';

   SparseMatrix T;
   trace.FormRectangularSystemMatrix(ess_trace_dofs, empty, T);
   unique_ptr<SparseMatrix> Tt(Transpose(T));

   SparseMatrix &D(div.SpMat());
   unique_ptr<SparseMatrix> Dt(Transpose(D));
   (*Dt) *= -1.0;

   SparseMatrix BC(M_h.GetTrueVSize(), M_h.GetTrueVSize());
   for (int i : ess_trace_dofs)
   {
      BC.Set(i, i, 1.0);
   }
   BC.Finalize();

   Array<int> offsets(4);
   offsets[0] = 0;
   offsets[1] = S_h.GetTrueVSize();
   offsets[2] = offsets[1] + V_h.GetTrueVSize();
   offsets[3] = offsets[2] + M_h.GetTrueVSize();

   BlockMatrix matrix(offsets);
   // Row 0
   matrix.SetBlock(0, 0, &mass.SpMat());
   matrix.SetBlock(0, 1, Dt.get());
   matrix.SetBlock(0, 2, &T);
   // Row 1
   matrix.SetBlock(1, 0, &D);
   // Row 2
   matrix.SetBlock(2, 0, Tt.get());
   matrix.SetBlock(2, 2, &BC);

   unique_ptr<SparseMatrix> monolithic(matrix.CreateMonolithic());

   // Form right-hand side
   auto f = [](const Vector &xvec)
   {
      return 2*M_PI*M_PI*(sin(M_PI*xvec[0]) * sin(M_PI*xvec[1]));
   };
   FunctionCoefficient f_coeff(f);
   LinearForm F(&V_h);
   F.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
   F.Assemble();

   BlockVector rhs(offsets);
   rhs.GetBlock(0) = 0.0;
   rhs.GetBlock(1) = F;
   rhs.GetBlock(2) = 0.0;

   BlockVector solution(offsets);

   UMFPackSolver solver(*monolithic);
   solver.Mult(rhs, solution);

   GridFunction sigma(&S_h);
   GridFunction u(&V_h);
   GridFunction lambda(&M_h);

   sigma = solution.GetBlock(0);
   u = solution.GetBlock(1);
   lambda = solution.GetBlock(2);

   auto u_exact = [](const Vector &xvec)
   {
      return sin(M_PI*xvec[0]) * sin(M_PI*xvec[1]);
   };
   FunctionCoefficient u_coeff(u_exact);

   std::cout << "Error: " << u.ComputeL2Error(u_coeff) << '\n';

   ParaViewDataCollection pv("HMM", &mesh);
   pv.SetPrefixPath("ParaView");
   pv.SetHighOrderOutput(true);
   pv.SetLevelsOfDetail(order);
   pv.RegisterField("u", &u);
   pv.RegisterField("sigma", &sigma);
   pv.SetCycle(0);
   pv.SetTime(0.0);
   pv.Save();

   return 0;
}
