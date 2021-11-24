
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double freq = 1.0, kappa;

// solving diffusion and grad-div problem at once, just for testing the block assembly
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


int main(int argc, char *argv[])
{
   // 1. Parse command line options
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   kappa = freq * M_PI;

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   // mesh.UniformRefinement();

   Array<FiniteElementSpace *> domain_fes;
   Array<FiniteElementSpace *> trace_fes;
   Array<FiniteElementCollection * > test_fecols;

   int dim = mesh.Dimension();
   int test_order = order;
   if (dim == 2 && (order%2 == 0 || (mesh.MeshGenerator() & 2 && order > 1)))
   {
      test_order++;
   }

   test_order++;

   cout << "test_order = " << test_order << endl;

   // diffusion problem spaces
   H1_FECollection H1fec(order, mesh.Dimension());
   FiniteElementSpace H1fes(&mesh, &H1fec);
   cout << "Number of H1 field unknowns: " << H1fes.GetTrueVSize() << endl;

   RT_Trace_FECollection trace_fec(order-1, mesh.Dimension());
   FiniteElementSpace RTtrace_fes(&mesh, &trace_fec);
   cout << "Number of RT trace unknowns: " << RTtrace_fes.GetTrueVSize() << endl;

   H1_FECollection H1test_fec(test_order,mesh.Dimension());

   domain_fes.Append(&H1fes);
   trace_fes.Append(&RTtrace_fes);
   test_fecols.Append(&H1test_fec);


   // grad-div problem spaces
   RT_FECollection RTfec(order-1, mesh.Dimension());
   FiniteElementSpace RTfes(&mesh, &RTfec);
   cout << "Number of RT field unknowns: " << RTfes.GetTrueVSize() << endl;

   H1_Trace_FECollection H1trace_fec(order, mesh.Dimension());
   FiniteElementSpace H1trace_fes(&mesh, &H1trace_fec);
   cout << "Number of H1 trace unknowns: " << H1trace_fes.GetTrueVSize() << endl;

   RT_FECollection RTtest_fec(test_order,mesh.Dimension());

   domain_fes.Append(&RTfes);
   trace_fes.Append(&H1trace_fes);
   test_fecols.Append(&RTtest_fec);


   NormalEquations * a = new NormalEquations(domain_fes,trace_fes,test_fecols);



   ConstantCoefficient one(1.0);
   // Diffusion problem
   a->AddDomainBFIntegrator(new DiffusionIntegrator(one),0,0);
   a->AddTraceElementBFIntegrator(new TraceIntegrator,0,0);
   a->AddTestIntegrator(new DiffusionIntegrator(one),0,0);
   a->AddTestIntegrator(new MassIntegrator(one),0,0);
   a->AddDomainLFIntegrator(new DomainLFIntegrator(one),0);

   // GradDiv Problem
   a->AddDomainBFIntegrator(new DivDivIntegrator(one),1,1);
   a->AddDomainBFIntegrator(new VectorFEMassIntegrator(one),1,1);
   a->AddTraceElementBFIntegrator(new NormalTraceIntegrator,1,1);
   a->AddTestIntegrator(new DivDivIntegrator(one),1,1);
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),1,1);
   VectorFunctionCoefficient f(dim, f_exact);
   a->AddDomainLFIntegrator(new VectorFEDomainLFIntegrator(f),1);

   a->Assemble();


   Array<int> ess_tdof_list0;
   Array<int> ess_tdof_list1;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      H1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list0);
      RTfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list1);
   }

   Array<int> ess_tdof_list(ess_tdof_list0.Size() + ess_tdof_list1.Size());
   for (int i = 0; i< ess_tdof_list0.Size(); i++)
   {
      ess_tdof_list[i] = ess_tdof_list0[i];
   }
   for (int i = ess_tdof_list0.Size(); i < ess_tdof_list.Size(); i++)
   {
      ess_tdof_list[i] = H1fes.GetTrueVSize() + 
                         ess_tdof_list1[i-ess_tdof_list0.Size()];
   }


   Vector X,B;
   OperatorPtr A;

   int size = H1fes.GetVSize() + RTtrace_fes.GetVSize() 
            + RTfes.GetVSize() + H1trace_fes.GetVSize();

   Vector x(size);
   x = 0.0;

   GridFunction rt_gf(&RTfes);
   VectorFunctionCoefficient F(dim, F_exact);
   rt_gf.ProjectCoefficient(F);
   x.SetVector(rt_gf,H1fes.GetVSize());

   a->FormLinearSystem(ess_tdof_list,x,A,X,B);

   GSSmoother M((SparseMatrix&)(*A));
   CGSolver cg;


   cg.SetRelTol(1e-6);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(3);
   cg.SetPreconditioner(M);
   cg.SetOperator(*A);
   cg.Mult(B, X);


   a->RecoverFEMSolution(X,x);

   GridFunction u_gf;
   double *data = x.GetData();
   u_gf.MakeRef(&H1fes,data);

   GridFunction rt_gf1;
   rt_gf1.MakeRef(&RTfes,&data[H1fes.GetVSize()]);




   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream solu_sock(vishost, visport);
   solu_sock.precision(8);
   solu_sock << "solution\n" << mesh << u_gf <<
             "window_title 'Numerical u' "
             << flush;

   socketstream soltrace_sock(vishost, visport);
   soltrace_sock.precision(8);
   soltrace_sock << "solution\n" << mesh << rt_gf1 <<
                 "window_title 'rt_gf' "
                 << flush;



}
