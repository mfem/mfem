//                                MFEM primal dpg example for grad-dic problem
//
// Compile with: make primal_dpg
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution, F, and r.h.s., f. See below for implementation.
void F_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;


int main(int argc, char *argv[])
{
   // 1. Parse command line options
   const char *mesh_file = "../../../data/star.mesh";
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   kappa = freq * M_PI;


   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   // mesh.UniformRefinement();

   RT_FECollection fec(order-1, mesh.Dimension());
   FiniteElementSpace RTfes(&mesh, &fec);

   H1_Trace_FECollection trace_fec(order, mesh.Dimension());
   FiniteElementSpace H1trace_fes(&mesh, &trace_fec);

   int dim = mesh.Dimension();
   int test_order = order;
   if (dim == 2 && (order%2 == 0 || (mesh.MeshGenerator() & 2 && order > 1)))
   {
      test_order++;
   }

   test_order++;

   RT_FECollection test_fec(test_order,mesh.Dimension());

   Array<FiniteElementSpace *> trial_fes; 
   Array<FiniteElementCollection * > test_fecs; 
   
   trial_fes.Append(&RTfes);
   trial_fes.Append(&H1trace_fes);
   test_fecs.Append(&test_fec);


   GridFunction rt_gf(&RTfes);
   VectorFunctionCoefficient F(dim, F_exact);
   rt_gf.ProjectCoefficient(F);

   Vector x(RTfes.GetVSize()+H1trace_fes.GetVSize());
   x = 0.;
   x.SetVector(rt_gf,0);


   ConstantCoefficient alpha(1.0);
   ConstantCoefficient beta(1.0);
   NormalEquations * a = new NormalEquations(trial_fes,test_fecs);
   a->AddTrialIntegrator(new DivDivIntegrator(alpha),0,0);
   a->AddTrialIntegrator(new VectorFEMassIntegrator(beta),0,0);
   a->AddTrialIntegrator(new NormalTraceIntegrator,1,0);
   a->AddTestIntegrator(new DivDivIntegrator(alpha),0,0);
   a->AddTestIntegrator(new VectorFEMassIntegrator(beta),0,0);


   VectorFunctionCoefficient f(dim, f_exact);
   a->AddDomainLFIntegrator(new VectorFEDomainLFIntegrator(f),0);
   a->Assemble();

   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      RTfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   Vector X,B;

   OperatorPtr Ah;
   a->FormLinearSystem(ess_tdof_list,x,Ah,X,B);

   BlockMatrix * A = (BlockMatrix *)(Ah.Ptr());
   BlockDiagonalPreconditioner * M = new BlockDiagonalPreconditioner(A->RowOffsets());
   M->owns_blocks = 1;
   for (int i=0; i<A->NumRowBlocks(); i++)
   {
      M->SetDiagonalBlock(i,new UMFPackSolver(A->GetBlock(i,i)));
   }

   CGSolver cg;
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(3);
   cg.SetPreconditioner(*M);
   cg.SetOperator(*A);
   cg.Mult(B, X);

   delete M;

   a->RecoverFEMSolution(X,x);

   // GridFunction u_gf;
   double *data = x.GetData();
   rt_gf.MakeRef(&RTfes,data);

   GridFunction exact_gf(&RTfes);
   exact_gf.ProjectCoefficient(F);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream solu_sock(vishost, visport);
   solu_sock.precision(8);
   solu_sock << "solution\n" << mesh << rt_gf <<
             "window_title 'Numerical u' "
             << flush;

   socketstream soltrace_sock(vishost, visport);
   soltrace_sock.precision(8);
   soltrace_sock << "solution\n" << mesh << exact_gf <<
                 "window_title 'Exact' "
                 << flush;

}


// The exact solution (for non-surface meshes)
void F_exact(const Vector &p, Vector &F)
{
   int dim = p.Size();

   double x = p(0);
   double y = p(1);
   // double z = (dim == 3) ? p(2) : 0.0; // Uncomment if F is changed to depend on z

   F(0) = cos(kappa*x)*sin(kappa*y);
   F(1) = cos(kappa*y)*sin(kappa*x);
   if (dim == 3)
   {
      F(2) = 0.0;
   }
}

// The right hand side
void f_exact(const Vector &p, Vector &f)
{
   int dim = p.Size();

   double x = p(0);
   double y = p(1);
   // double z = (dim == 3) ? p(2) : 0.0; // Uncomment if f is changed to depend on z

   double temp = 1 + 2*kappa*kappa;

   f(0) = temp*cos(kappa*x)*sin(kappa*y);
   f(1) = temp*cos(kappa*y)*sin(kappa*x);
   if (dim == 3)
   {
      f(2) = 0;
   }
}