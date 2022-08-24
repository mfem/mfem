//                       MFEM Ultraweak DPG example
//
// Compile with: make uw_dpg
//
// sample runs 
//   ./diffusion -m ../../../data/l-shape.mesh -o 2 -ref 10 -graph-norm -do 1 -prob 0 -sc
//   ./diffusion -m ../../../data/l-shape.mesh -o 2 -ref 10 -no-graph-norm -do 1 -prob 0
//   ./diffusion -m ../../../data/inline-quad.mesh -o 2 -ref 0 -graph-norm -do 1 -prob 1
//   ./diffusion -m ../../../data/inline-tri.mesh -o 2 -ref 2 -graph-norm -do 1 -prob 1 -theta 0.0

//     - Δ u = f,   in Ω
//         u = u_0, on ∂Ω

// First Order System

//   ∇ u - σ = 0, in Ω
// - ∇⋅σ     = f, in Ω
//        u  = 0, in ∂Ω

// UW-DPG:
// 
// u ∈ L^2(Ω), σ ∈ (L^2(Ω))^dim 
// û ∈ H^1/2, σ̂ ∈ H^-1/2  
// -(u , ∇⋅τ) - (σ , τ) + < û, τ⋅n> = 0,      ∀ τ ∈ H(div,Ω)      
//  (σ , ∇ v) + < σ̂, v  >           = (f,v)   ∀ v ∈ H^1(Ω)
//                                û = 0       on ∂Ω 

// Note: 
// û := u
// σ̂ := -σ

// -------------------------------------------------------------
// |   |     u     |     σ     |    û      |    σ̂    |  RHS    |
// -------------------------------------------------------------
// | τ | -(u,∇⋅τ)  |  -(σ,τ)   | < û, τ⋅n> |         |    0    |
// |   |           |           |           |         |         |
// | v |           |  (σ,∇ v)  |           |  <σ̂,v>  |  (f,v)  |  

// where (τ,v) ∈  H(div,Ω) × H^1(Ω) 

#include "mfem.hpp"
#include "util/weakform.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

enum prob_type
{
   lshape,   
   general  
};

prob_type prob;

void solution(const Vector & X, double & u, Vector & du, double & d2u);

double exact_u(const Vector & X)
{
   double u, d2u;
   Vector du;
   solution(X,u,du,d2u);
   return u;
}

void exact_sigma(const Vector & X, Vector & sigma)
{
   double u, d2u;
   Vector du;
   solution(X,u,du,d2u);
   // σ = ∇ u
   sigma = du;
}

double exact_hatu(const Vector & X)
{
   return exact_u(X);
}

void exact_hatsigma(const Vector & X, Vector & hatsigma)
{
   exact_sigma(X,hatsigma);
   hatsigma *= -1.;
}

double f_exact(const Vector & X)
{
   double u, d2u;
   Vector du;
   solution(X,u,du,d2u);
   return -d2u;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   int ref = 0;
   double theta = 0.7;
   bool adjoint_graph_norm = true;
   bool visualization = true;
   int iprob = 0;
   bool static_cond = false;

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
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: lshape, 1: General");
   args.AddOption(&theta, "-theta", "--theta",
                  "AMR fraction.");                  
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");                  
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

   if (iprob > 1) { iprob = 1; }
   prob = (prob_type)iprob;

   if (prob == prob_type::lshape)
   {
      mesh_file = "../../../data/l-shape.mesh";
      // mesh_file = "lshape2.mesh";
   }

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   if (prob == prob_type::lshape)
   {
      mesh.EnsureNodes();
      GridFunction * nodes = mesh.GetNodes();
      *nodes -= 0.5;
      for (int i = 0; i<nodes->Size()/2; i++)
      {
         double temp = (*nodes)[2*i];
         (*nodes)[2*i] = (*nodes)[2*i+1];
         (*nodes)[2*i+1] = -temp;
      }
      *nodes *= 2.0;
   }

   mesh.UniformRefinement();

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
   FiniteElementCollection * tau_fec = new RT_FECollection(test_order-1, dim);
   FiniteElementCollection * v_fec = new H1_FECollection(test_order, dim);


   // Coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);

   // Normal equation weak formulation
   Array<FiniteElementSpace * > trial_fes; 
   Array<FiniteElementCollection * > test_fec; 

   trial_fes.Append(u_fes);
   trial_fes.Append(sigma_fes);
   trial_fes.Append(hatu_fes);
   trial_fes.Append(hatsigma_fes);

   test_fec.Append(tau_fec);
   test_fec.Append(v_fec);

   DPGWeakForm * a = new DPGWeakForm(trial_fes,test_fec);
   a->StoreMatrices(true);

   //  -(u,∇⋅τ)
   a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(one),0,0);

   // -(σ,τ) 
   a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(negone)),1,0);

   // (σ,∇ v)
   a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(one)),1,1);

   //  <û,τ⋅n>
   a->AddTrialIntegrator(new NormalTraceIntegrator,2,0);

   //  <σ̂,v> 
   a->AddTrialIntegrator(new TraceIntegrator,3,1);

   // test integrators (space-induced norm for H(div) × H1)
   // (∇⋅τ,∇⋅δτ)
   a->AddTestIntegrator(new DivDivIntegrator(one),0,0);
   // (τ,δτ)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),0,0);
   // (∇v,∇δv)
   a->AddTestIntegrator(new DiffusionIntegrator(one),1,1);
   // (v,δv)
   a->AddTestIntegrator(new MassIntegrator(one),1,1);

   // additional terms for adjoint graph norm
   if (adjoint_graph_norm)
   {
      // -(∇v,δτ) 
      a->AddTestIntegrator(new MixedVectorGradientIntegrator(negone),1,0);
      // -(τ,∇δv) 
      a->AddTestIntegrator(new MixedVectorWeakDivergenceIntegrator(one),0,1);
      // (τ,δτ)
      a->AddTestIntegrator(new VectorFEMassIntegrator(one),0,0);
   }

   // RHS
   FunctionCoefficient f(f_exact);
   if (prob == prob_type::general)
   {
      a->AddDomainLFIntegrator(new DomainLFIntegrator(f),1);
   }

   FunctionCoefficient uex(exact_u);

   FunctionCoefficient hatuex(exact_hatu);
   Array<int> elements_to_refine;
   GridFunction hatu_gf;

   socketstream u_out;
   socketstream sigma_out;
   socketstream mesh_out;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      u_out.open(vishost, visport);
      sigma_out.open(vishost, visport);
      mesh_out.open(vishost, visport);
   }

   ConvergenceStudy rates_u;
   ConvergenceStudy rates_sigma;

   for (int iref = 0; iref<=ref; iref++)
   {
      if (static_cond) { a->EnableStaticCondensation(); }
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

      Array<int> offsets(5);
      offsets[0] = 0;
      offsets[1] = u_fes->GetVSize();
      offsets[2] = sigma_fes->GetVSize();
      offsets[3] = hatu_fes->GetVSize();
      offsets[4] = hatsigma_fes->GetVSize();
      offsets.PartialSum();
      BlockVector x(offsets);
      x = 0.0;
      hatu_gf.MakeRef(hatu_fes,x.GetBlock(2));
      hatu_gf.ProjectBdrCoefficient(hatuex,ess_bdr);

      OperatorPtr Ah;
      Vector X,B;
      a->FormLinearSystem(ess_tdof_list,x,Ah,X,B);

      BlockMatrix * A = Ah.As<BlockMatrix>();

      BlockDiagonalPreconditioner * M = new BlockDiagonalPreconditioner(A->RowOffsets());
      M->owns_blocks = 1;
      for (int i=0; i<A->NumRowBlocks(); i++)
      {
         M->SetDiagonalBlock(i,new GSSmoother(A->GetBlock(i,i)));
      }

      CGSolver cg;
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(200000);
      cg.SetPrintLevel(3);
      cg.SetPreconditioner(*M);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      delete M;

      a->RecoverFEMSolution(X,x);

      Vector & residuals = a->ComputeResidual(x);
      mfem::out << "residual = " << residuals.Norml2() << endl;
      elements_to_refine.SetSize(0);
      double max_resid = residuals.Max();
      for (int iel = 0; iel<mesh.GetNE(); iel++)
      {
         if (residuals[iel] > theta * max_resid)
         {
            elements_to_refine.Append(iel);
         }
      }

      GridFunction u_gf;
      u_gf.MakeRef(u_fes,x.GetBlock(0));

      rates_u.AddL2GridFunction(&u_gf,&uex);

      GridFunction sigma_gf;
      sigma_gf.MakeRef(sigma_fes,x.GetBlock(1));

      if (visualization)
      {
         u_out.precision(8);
         string keys = (iref == 0) ? "keys em\n" : "keys";
         u_out << "solution\n" << mesh << u_gf 
               << "window_title 'Numerical u' "
                  << flush;

         sigma_out.precision(8);
         sigma_out << "solution\n" << mesh << sigma_gf <<
               "window_title 'Numerical flux' "
               << flush;

         mesh_out.precision(8);
         mesh_out << "mesh\n" << mesh 
                  << keys
                  << "window_title 'Mesh' "
                  << flush;

      }

      if (iref == ref) break;

      mesh.GeneralRefinement(elements_to_refine);
      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();
   }

   rates_u.Print();

   delete a;
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


void solution(const Vector & X, double & u, Vector & du, double & d2u)
{
   double x = X[0];
   double y = X[1];
   double z = 0.;
   if (X.Size() == 3) z = X[2];
   du.SetSize(X.Size());
   du = 0.;
   d2u = 0.;

   switch(prob)
   {
      case lshape:
      {
         // // shift, rotate and scale to match the benchmark problem
         // x = 2.0*(X[1] - 0.5);
         // y = -2.0*(X[0] - 0.5);   

         double r = sqrt(x*x + y*y);
         double alpha = 2./3.;
         double phi = atan2(y,x);
         if (phi < 0) phi += 2*M_PI;
         u = pow(r,alpha) * sin(alpha * phi);
      }
      break;
      default:
      {
         double alpha = M_PI * (x + y + z);
         u = sin(alpha);
         du.SetSize(X.Size());
         for (int i = 0; i<du.Size(); i++)
         {
            du[i] = M_PI * cos(alpha);
         }
         d2u = - M_PI*M_PI * u * du.Size();
      }
      break;
   }
}