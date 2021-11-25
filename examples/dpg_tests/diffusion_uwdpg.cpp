//                                MFEM Example 1
//
// Compile with: make poisson_fosls
//
//     - Δ u = f, in Ω
//         u = 0, on ∂Ω

// First Order System

//   ∇ u - σ = 0, in Ω
// - ∇⋅σ     = f, in Ω
//        u  = 0, in ∂Ω

// UW-DPG:
// 
// u ∈ L^2(Ω), σ ∈ (L^2(Ω))^dim 
// û ∈ H^1/2, σ̂ ∈ H^-1/2  
// -(u , ∇⋅v) + < û, v⋅n> - (σ , v) = 0,      ∀ v ∈ H(div,Ω)      
//  (σ , ∇ τ) - < σ̂, τ  >           = (f,τ)   ∀ τ ∈ H^1(Ω)
//            û = 0        on ∂Ω 



// ----------------------------------------------------------------------
// |   |     u     |     σ      |     û      |    σ̂    |  RHS    |
// ----------------------------------------------------------------------
// | v | -(u,∇⋅v)  |  -(σ,v)    |  < û, v⋅n> |         |    0    |
// |   |           |                         |         |
// | τ |           |  (σ,∇ τ)   |            | -<σ̂,τ>  |  (f,τ)  |  

// where (v,τ) ∈  H(div,Ω) × H^1(Ω) 

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 1;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
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

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();


   // Define spaces
   // L2 space for u
   FiniteElementCollection *u_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *u_fes = new FiniteElementSpace(&mesh,u_fec);

   // Vector L2 space for σ 
   FiniteElementCollection *sigma_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *sigma_fes = new FiniteElementSpace(&mesh,sigma_fec,dim); 

   // H^1/2 space for û 
   FiniteElementCollection * hatu_fec = new H1_Trace_FECollection(order,dim);
   FiniteElementSpace *hatu_fes = new FiniteElementSpace(&mesh,hatu_fec);

   // H^-1/2 space for σ̂ 
   FiniteElementCollection * hatsigma_fec = new RT_Trace_FECollection(order-1,dim);   
   FiniteElementSpace *hatsigma_fes = new FiniteElementSpace(&mesh,hatsigma_fec);

   // testspace fe collections
   int test_order = order + 2;
   FiniteElementCollection * v_fec = new RT_FECollection(test_order-1, dim);
   FiniteElementCollection * tau_fec = new H1_FECollection(test_order, dim);


   // Coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);
   Vector negone_v(dim); negone_v = -1.0;
   Vector one_v(dim); one_v = 1.0;

   VectorConstantCoefficient one_vec(one_v);
   VectorConstantCoefficient negone_vec(negone_v);


   // Normal equation weak formulation
   Array<FiniteElementSpace * > domain_fes; 
   Array<FiniteElementSpace * > trace_fes; 
   Array<FiniteElementCollection * > test_fec; 

   domain_fes.Append(u_fes);
   domain_fes.Append(sigma_fes);

   trace_fes.Append(hatu_fes);
   trace_fes.Append(hatsigma_fes);

   test_fec.Append(v_fec);
   test_fec.Append(tau_fec);

   NormalEquations * a = new NormalEquations(domain_fes,trace_fes,test_fec);

   //  -(u,∇⋅v)
   a->AddDomainBFIntegrator(new MixedScalarWeakGradientIntegrator(one),0,0);

   // -(σ,v) 
   TransposeIntegrator * mass = new TransposeIntegrator(new VectorFEMassIntegrator(negone));
   a->AddDomainBFIntegrator(mass, 1, 0);

   // (σ,∇ τ)
   a->AddDomainBFIntegrator(new MixedScalarWeakDivergenceIntegrator(negone_vec),1,1);

   //  < û, v⋅n>
   a->AddTraceElementBFIntegrator(new NormalTraceIntegrator,0,0);

   // -<σ̂,τ> (sign is included in the variable)
   a->AddTraceElementBFIntegrator(new TraceIntegrator,1,1);


   // test integrators (try mathematician's norm now Hdiv × H1)
   a->AddTestIntegrator(new DivDivIntegrator(one),0,0);
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),0,0);
   a->AddTestIntegrator(new DiffusionIntegrator(one),1,1);
   a->AddTestIntegrator(new MassIntegrator(one),1,1);


   // RHS
   a->AddDomainLFIntegrator(new DomainLFIntegrator(one),1);

   a->Assemble();


   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      hatu_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // shift the ess_tdofs
   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      ess_tdof_list[i] += u_fes->GetTrueVSize() + sigma_fes->GetTrueVSize();
   }

   Vector X,B;
   OperatorPtr A;

   int size = u_fes->GetTrueVSize() + sigma_fes->GetVSize() 
            + hatu_fes->GetVSize() + hatsigma_fes->GetVSize();

   Vector x(size);
   x = 0.0;

   a->FormLinearSystem(ess_tdof_list,x,A,X,B);


   GSSmoother M((SparseMatrix&)(*A));
   CGSolver cg;




   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(3);
   cg.SetPreconditioner(M);
   cg.SetOperator(*A);
   cg.Mult(B, X);


   a->RecoverFEMSolution(X,x);


   GridFunction u_gf;
   double *data = x.GetData();
   u_gf.MakeRef(u_fes,data);


   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream solu_sock(vishost, visport);
   solu_sock.precision(8);
   solu_sock << "solution\n" << mesh << u_gf <<
             "window_title 'Numerical u' "
             << flush;

   // delete a;
   delete tau_fec;
   delete v_fec;
   delete hatsigma_fes;
   delete hatsigma_fec;
   delete hatu_fes;
   delete hatu_fec;
   delete sigma_fec;
   delete sigma_fes;
   delete u_fec;
   delete u_fes;


   return 0;
}
