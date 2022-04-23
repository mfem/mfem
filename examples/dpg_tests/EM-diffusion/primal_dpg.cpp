//                     MFEM primal_dpg example
//
// Compile with: make primal_dpg
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;
int dim;
int main(int argc, char *argv[])
{
   // 1. Parse command line options
   const char *mesh_file = "../../../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   int ref = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");   
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&ref, "-ref", "--refinements",
                  "Number of refinements.");                         
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   kappa = freq * M_PI;

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);

   for (int i = 0; i<ref; i++)
   {
      mesh.UniformRefinement();
   }

   dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();
   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   ND_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace NDfes(&mesh, &fec);

   FiniteElementCollection * trace_fec = nullptr;
   if (dim == 3)
   {
      trace_fec = new ND_Trace_FECollection(order,mesh.Dimension());
   }
   else
   {
      trace_fec = new H1_Trace_FECollection(order,mesh.Dimension());
   }
   FiniteElementSpace trace_fes(&mesh, trace_fec);

   int test_order = order+1;

   ND_FECollection test_fec(test_order,mesh.Dimension());

   Array<FiniteElementSpace * > trial_fes; 
   Array<FiniteElementCollection * > test_fecs; 

   trial_fes.Append(&NDfes);
   trial_fes.Append(&trace_fes);
   test_fecs.Append(&test_fec);

   NormalEquations * a = new NormalEquations(trial_fes,test_fecs);


   ConstantCoefficient one(1.0);
   a->AddTrialIntegrator(new CurlCurlIntegrator(one),0,0);
   a->AddTrialIntegrator(new VectorFEMassIntegrator(one),0,0);
   a->AddTrialIntegrator(new TangentTraceIntegrator,1,0);
   
   a->AddTestIntegrator(new CurlCurlIntegrator(one),0,0);
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),0,0);

   VectorFunctionCoefficient f(sdim, f_exact);
   a->AddDomainLFIntegrator(new VectorFEDomainLFIntegrator(f),0);


   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      NDfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }


   Vector X,B;
   OperatorPtr Ah;

   VectorFunctionCoefficient E(sdim, E_exact);

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = NDfes.GetVSize();
   offsets[2] = trace_fes.GetVSize();
   offsets.PartialSum();

   BlockVector x(offsets);
   x = 0.;


   GridFunction E_gf(&NDfes);
   E_gf.MakeRef(&NDfes,x.GetBlock(0));
   E_gf.ProjectBdrCoefficientTangent(E,ess_bdr);
   E_gf.ProjectCoefficient(E);


   a->FormLinearSystem(ess_tdof_list,x,Ah,X,B);


   BlockMatrix * A = (BlockMatrix *)(Ah.Ptr());


   BlockDiagonalPreconditioner * M = new BlockDiagonalPreconditioner(A->RowOffsets());
   M->owns_blocks = 1;
   for (int i=0; i<A->NumRowBlocks(); i++)
   {
      M->SetDiagonalBlock(i,new UMFPackSolver(A->GetBlock(i,i)));
   }

   GMRESSolver cg;
   cg.SetRelTol(1e-8);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(3);
   cg.SetPreconditioner(*M);
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete M;

   a->RecoverFEMSolution(X,x);

   E_gf.MakeRef(&NDfes,x.GetData());

   double L2Error = E_gf.ComputeL2Error(E);
   mfem::out << "L2_error = " << L2Error << endl;


   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream solu_sock(vishost, visport);
   solu_sock.precision(8);
   solu_sock << "solution\n" << mesh << E_gf <<
             "window_title 'Numerical u' "
             << flush;
   delete trace_fec;
   return 0;
}


void E_exact(const Vector &x, Vector &E)
{
   if (dim == 3)
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(2));
      E(2) = sin(kappa * x(0));
   }
   else
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(0));
      if (x.Size() == 3) { E(2) = 0.0; }
   }
}

void f_exact(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
      f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}
