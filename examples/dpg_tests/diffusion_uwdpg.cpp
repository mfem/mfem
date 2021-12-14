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

// -------------------------------------------------------------
// |   |     u     |     σ     |    û      |    σ̂    |  RHS    |
// -------------------------------------------------------------
// | v | -(u,∇⋅v)  |  -(σ,v)   | < û, v⋅n> |         |    0    |
// |   |           |           |           |         |         |
// | τ |           |  (σ,∇ τ)  |           | -<σ̂,τ>  |  (f,τ)  |  

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
   int delta_order = 1;
   int ref = 1;
   bool adjoint_graph_norm = false;
   bool visualization = true;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&ref, "-ref", "--num_refinements",
                  "Number of uniform refinements");               
   args.AddOption(&adjoint_graph_norm, "-graph-norm", "--adjoint-graph-norm",
                  "-no-graph-norm", "--no-adjoint-graph-norm",
                  "Enable or disable Adjoint Graph Norm on the test space");               
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

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // for (int i = 0; i<ref; i++)
   // {
   //    mesh.UniformRefinement();
   // }

   // Define spaces
   // L2 space for u
   FiniteElementCollection *u_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *u_fes = new FiniteElementSpace(&mesh,u_fec);

   // Vector L2 space for σ 
   FiniteElementCollection *sigma_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *sigma_fes = new FiniteElementSpace(&mesh,sigma_fec, dim); 

   // H^1/2 space for û 
   FiniteElementCollection * hatu_fec = new H1_Trace_FECollection(order,dim);
   FiniteElementSpace *hatu_fes = new FiniteElementSpace(&mesh,hatu_fec);

   // H^-1/2 space for σ̂ 
   FiniteElementCollection * hatsigma_fec = new RT_Trace_FECollection(order-1,dim);   
   FiniteElementSpace *hatsigma_fes = new FiniteElementSpace(&mesh,hatsigma_fec);

   // testspace fe collections
   int test_order = order+delta_order;
   FiniteElementCollection * v_fec = new RT_FECollection(test_order-1, dim);
   FiniteElementCollection * tau_fec = new H1_FECollection(test_order, dim);


   // Coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);

   // Normal equation weak formulation
   Array<FiniteElementSpace * > fes; 
   Array<FiniteElementCollection * > test_fec; 

   fes.Append(u_fes);
   fes.Append(sigma_fes);
   fes.Append(hatu_fes);
   fes.Append(hatsigma_fes);

   test_fec.Append(v_fec);
   test_fec.Append(tau_fec);

   NormalEquations * a = new NormalEquations(fes,test_fec);

   //  -(u,∇⋅v)
   a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(one),0,0);

   // -(σ,v) 
   TransposeIntegrator * mass = new TransposeIntegrator(new VectorFEMassIntegrator(negone));
   a->AddTrialIntegrator(mass,1,0);

   // (σ,∇ τ)
   TransposeIntegrator * grad = new TransposeIntegrator(new GradientIntegrator(one));
   a->AddTrialIntegrator(grad,1,1);

   //  <û,v⋅n>
   a->AddTrialIntegrator(new NormalTraceIntegrator,2,0);

   // -<σ̂,τ> (sign is included in σ̂)
   a->AddTrialIntegrator(new TraceIntegrator,3,1);

   // test integrators (space-induced norm for H(div) × H1)
   // (∇⋅v,∇⋅δv)
   a->AddTestIntegrator(new DivDivIntegrator(one),0,0);
   // (v,δv)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),0,0);
   // (∇τ,∇δτ)
   a->AddTestIntegrator(new DiffusionIntegrator(one),1,1);
   // (τ,δτ)
   a->AddTestIntegrator(new MassIntegrator(one),1,1);

   // additional terms for adjoint graph norm
   if (adjoint_graph_norm)
   {
      // -(∇τ,δv) 
      a->AddTestIntegrator(new MixedVectorGradientIntegrator(negone),1,0);
      // -(v,∇δv) 
      a->AddTestIntegrator(new MixedVectorWeakDivergenceIntegrator(one),0,1);
      // (v,δv)
      a->AddTestIntegrator(new VectorFEMassIntegrator(one),0,0);
   }

   // RHS
   a->AddDomainLFIntegrator(new DomainLFIntegrator(one),1);


   for (int i = 0; i<ref; i++)
   {
      mesh.UniformRefinement();
      for (int i =0; i<fes.Size(); i++)
      {
         fes[i]->Update(false);
      }
      a->StoreMatrices(true);
      a->Update();
      a->Assemble();



      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (mesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh.bdr_attributes.Max());
         ess_bdr = 1;
         hatu_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

   // shift the ess_tdofs
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         ess_tdof_list[i] += u_fes->GetTrueVSize() + sigma_fes->GetTrueVSize();
      }

      Vector X,B;
      Array<int> offsets(5);
      offsets[0] = 0;
      offsets[1] = u_fes->GetVSize();
      offsets[2] = sigma_fes->GetVSize();
      offsets[3] = hatu_fes->GetVSize();
      offsets[4] = hatsigma_fes->GetVSize();
      offsets.PartialSum();
      BlockVector x(offsets);
      x = 0.0;

      OperatorPtr Ah;
      a->FormLinearSystem(ess_tdof_list,x,Ah,X,B);

      BlockMatrix * A = Ah.As<BlockMatrix>();

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

      a->ComputeResidual(x);

      GridFunction u_gf;
      u_gf.MakeRef(u_fes,x.GetBlock(0));

      GridFunction sigma_gf;
      sigma_gf.MakeRef(sigma_fes,x.GetBlock(1));

      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream solu_sock(vishost, visport);
         solu_sock.precision(8);
         solu_sock << "solution\n" << mesh << u_gf <<
                  "window_title 'Numerical u' "
                  << flush;

         socketstream sols_sock(vishost, visport);
         sols_sock.precision(8);
         sols_sock << "solution\n" << mesh << sigma_gf <<
               "window_title 'Numerical flux' "
               << flush;
      }
   }

   delete a;
   delete tau_fec;
   delete v_fec;
   delete hatsigma_fes;
   delete hatsigma_fec;
   delete hatu_fes;
   delete hatu_fec;
   delete sigma_fec;
   delete u_fec;
   delete u_fes;

   return 0;
}
