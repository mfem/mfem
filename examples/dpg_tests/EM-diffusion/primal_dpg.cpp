//                     MFEM primal_dpg example
//
// Compile with: make primal_dpg
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command line options
   const char *mesh_file = "../../../data/star.mesh";
   int order = 1;
   bool static_cond = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");   
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   // mesh.UniformRefinement();

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace H1fes(&mesh, &fec);

   RT_Trace_FECollection trace_fec(order-1, mesh.Dimension());
   FiniteElementSpace RTtrace_fes(&mesh, &trace_fec);

   int dim = mesh.Dimension();
   int test_order = order;
   if (dim == 2 && (order%2 == 0 || (mesh.MeshGenerator() & 2 && order > 1)))
   {
      test_order++;
   }

   test_order++;

   H1_FECollection test_fec(test_order,mesh.Dimension());

   Array<FiniteElementSpace * > trial_fes; 
   Array<FiniteElementCollection * > test_fecs; 

   trial_fes.Append(&H1fes);
   trial_fes.Append(&RTtrace_fes);
   test_fecs.Append(&test_fec);

   NormalEquations * a = new NormalEquations(trial_fes,test_fecs);


   ConstantCoefficient one(1.0);
   a->AddTrialIntegrator(new DiffusionIntegrator(one),0,0);
   a->AddTrialIntegrator(new TraceIntegrator,1,0);

   BilinearFormIntegrator * diffusion = new DiffusionIntegrator(one);
   BilinearFormIntegrator * mass = new MassIntegrator(one);
   a->AddTestIntegrator(diffusion,0,0);
   a->AddTestIntegrator(mass,0,0);

   a->AddDomainLFIntegrator(new DomainLFIntegrator(one),0);
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      H1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   Vector X,B;
   OperatorPtr Ah;

   int size = H1fes.GetVSize() + RTtrace_fes.GetVSize();

   Vector x(size);
   x = 0.0;

   a->FormLinearSystem(ess_tdof_list,x,Ah,X,B);

   BlockMatrix * A = (BlockMatrix *)(Ah.Ptr());
   BlockDiagonalPreconditioner * M = new BlockDiagonalPreconditioner(A->RowOffsets());
   M->owns_blocks = 1;
   for (int i=0; i<A->NumRowBlocks(); i++)
   {
      M->SetDiagonalBlock(i,new UMFPackSolver(A->GetBlock(i,i)));
   }

   CGSolver cg;
   cg.SetRelTol(1e-6);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(3);
   cg.SetPreconditioner(*M);
   cg.SetOperator(*A);
   cg.Mult(B, X);

   delete M;


   a->RecoverFEMSolution(X,x);

   GridFunction u_gf;
   double *data = x.GetData();
   u_gf.MakeRef(&H1fes,data);

   GridFunction s_gf;
   s_gf.MakeRef(&RTtrace_fes,&data[H1fes.GetVSize()]);



   RT_FECollection RTfec(order-1, mesh.Dimension());
   FiniteElementSpace RTfes(&mesh, &RTfec);

   GridFunction sigma_gf(&RTfes);
   sigma_gf = 0.0;
   for (int i = 0; i<mesh.GetNE(); i++)
   {
      Array<int> strace_dofs;
      Array<int> trace_dofs;
      Vector dofs;
      RTtrace_fes.GetElementDofs(i,trace_dofs);
      strace_dofs.SetSize(trace_dofs.Size());
      // shift dofs;
      for (int j = 0; j< trace_dofs.Size(); j++)
      {
         int offset = trace_dofs[j] < 0 ? -H1fes.GetVSize() : H1fes.GetVSize();
         strace_dofs[j] = offset + trace_dofs[j];
      }
      x.GetSubVector(strace_dofs, dofs);
      sigma_gf.SetSubVector(trace_dofs,dofs);

   }

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream solu_sock(vishost, visport);
   solu_sock.precision(8);
   solu_sock << "solution\n" << mesh << u_gf <<
             "window_title 'Numerical u' "
             << flush;

   socketstream soltrace_sock(vishost, visport);
   soltrace_sock.precision(8);
   soltrace_sock << "solution\n" << mesh << sigma_gf <<
                 "window_title 'Flux sigma_n' "
                 << flush;



}
