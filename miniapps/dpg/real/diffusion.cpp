//                       MFEM Ultraweak DPG example for diffusion
//
// Compile with: make diffusion
//
// sample runs 
//   ./diffusion -m ../../../data/star.mesh -o 3 -ref 1 -do 1 -prob 1 -sc
//   ./diffusion -m ../../../data/inline-tri.mesh -o 2 -ref 3 -do 1 -prob 0
//   ./diffusion -m ../../../data/inline-quad.mesh -o 4 -ref 2 -do 2 -prob 0 -sc
//   ./diffusion -m ../../../data/inline-tet.mesh -o 3 -ref 0 -do 1 -prob 1 -sc

// Description:  
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the Poisson problem

//       - Δ u = f,   in Ω
//         u = u_0, on ∂Ω

// It solves two kinds of problems 
// a) f = 1 and u_0 = 0 (like ex1)
// b) A manufactured solution problem where u_exact = sin(π * (x + y + z)). 
//    This example computes and prints out convergence rates for the L2 error.

// The DPG UW deals with the First Order System
//   ∇ u - σ = 0, in Ω
// - ∇⋅σ     = f, in Ω
//        u  = u_0, in ∂Ω

// Ultraweak-DPG is obtained by integration by parts of both equations and the 
// introduction of trace unknowns on the mesh skeleton
// 
// u ∈ L^2(Ω), σ ∈ (L^2(Ω))^dim 
// û ∈ H^1/2(Γ_h), σ̂ ∈ H^-1/2(Γ_h)  
// -(u , ∇⋅τ) - (σ , τ) + < û, τ⋅n> = 0,      ∀ τ ∈ H(div,Ω)      
//  (σ , ∇ v) + < σ̂, v  >           = (f,v)   ∀ v ∈ H^1(Ω)
//                                û = u_0       on ∂Ω 

// Note: 
// û := u and σ̂ := -σ
//
// -------------------------------------------------------------
// |   |     u     |     σ     |    û      |    σ̂    |  RHS    |
// -------------------------------------------------------------
// | τ | -(u,∇⋅τ)  |  -(σ,τ)   | < û, τ⋅n> |         |    0    |
// |   |           |           |           |         |         |
// | v |           |  (σ,∇ v)  |           |  <σ̂,v>  |  (f,v)  |  

// where (τ,v) ∈  H(div,Ω) × H^1(Ω) 

// Here we use the "space-induced" test norm i.e.,
//
// ||(t,v)||^2_H(div)×H^1 := ||t||^2 + ||∇⋅t||^2 + ||v||^2 + ||∇v||^2 

#include "mfem.hpp"
#include "util/weakform.hpp"
#include "../../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

enum prob_type
{
   manufactured,   
   general  
};

prob_type prob;

void solution(const Vector & X, double & u, Vector & du, double & d2u);
double exact_u(const Vector & X);
void exact_sigma(const Vector & X, Vector & sigma);
double exact_hatu(const Vector & X);
void exact_hatsigma(const Vector & X, Vector & hatsigma);
double f_exact(const Vector & X);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   int ref = 0;
   bool visualization = true;
   int iprob = 1;
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
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: manufactured, 1: general");
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

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

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

   // test space fe collections
   int test_order = order+delta_order;
   FiniteElementCollection * tau_fec = new RT_FECollection(test_order-1, dim);
   FiniteElementCollection * v_fec = new H1_FECollection(test_order, dim);

   Array<FiniteElementSpace * > trial_fes; 
   Array<FiniteElementCollection * > test_fec; 

   trial_fes.Append(u_fes);
   trial_fes.Append(sigma_fes);
   trial_fes.Append(hatu_fes);
   trial_fes.Append(hatsigma_fes);
   test_fec.Append(tau_fec);
   test_fec.Append(v_fec);

   // Required coefficients for the weak formulation
   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);
   FunctionCoefficient f(f_exact); // rhs for the manufactured solution problem

   // Required coefficients for the exact solution case
   FunctionCoefficient uex(exact_u);
   VectorFunctionCoefficient sigmaex(dim,exact_sigma);
   FunctionCoefficient hatuex(exact_hatu);

   // Define the DPG weak formulation
   DPGWeakForm * a = new DPGWeakForm(trial_fes,test_fec);

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

   // RHS
   if (prob == prob_type::manufactured)
   {
      a->AddDomainLFIntegrator(new DomainLFIntegrator(f),1);
   }
   else
   {
      a->AddDomainLFIntegrator(new DomainLFIntegrator(one),1);
   }

   // GridFunction for Dirichlet bdr data
   GridFunction hatu_gf;

   // Visualization streams
   socketstream u_out;
   socketstream sigma_out;

   if (prob == prob_type::manufactured)
   {
      mfem::out << "\n Refinement |" 
                << "    Dofs    |" 
                << "  L2 Error  |" 
                << "  Rate  |" 
                << " CG it  |" << endl;
      mfem::out << " --------------------"      
                <<  "-------------------"    
                <<  "-------------------" << endl;   
   }

   double err0 = 0.;
   int dof0=0.;
   if (static_cond) { a->EnableStaticCondensation(); }
   for (int iref = 0; iref<=ref; iref++)
   {
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
      if (prob == prob_type::manufactured)
      {
         hatu_gf.MakeRef(hatu_fes,x.GetBlock(2));
         hatu_gf.ProjectBdrCoefficient(hatuex,ess_bdr);
      }

      OperatorPtr Ah;
      Vector X,B;
      a->FormLinearSystem(ess_tdof_list,x,Ah,X,B);

      BlockMatrix * A = Ah.As<BlockMatrix>();

      BlockDiagonalPreconditioner M(A->RowOffsets());
      M.owns_blocks = 1;
      for (int i=0; i<A->NumRowBlocks(); i++)
      {
         M.SetDiagonalBlock(i,new GSSmoother(A->GetBlock(i,i)));
      }

      CGSolver cg;
      cg.SetRelTol(1e-10);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(prob== prob_type::general ? 3 : 0);
      cg.SetPreconditioner(M);
      cg.SetOperator(*A);
      cg.Mult(B, X);

      a->RecoverFEMSolution(X,x);

      GridFunction u_gf, sigma_gf;
      u_gf.MakeRef(u_fes,x.GetBlock(0));
      sigma_gf.MakeRef(sigma_fes,x.GetBlock(1));

      if (prob == prob_type::manufactured)
      {
         int l2dofs = u_fes->GetVSize() + sigma_fes->GetVSize();
         double u_err = u_gf.ComputeL2Error(uex);
         double sigma_err = sigma_gf.ComputeL2Error(sigmaex);
         double L2Error = sqrt(u_err*u_err + sigma_err*sigma_err);
         double rate_err = (iref) ? dim*log(err0/L2Error)/log((double)dof0/l2dofs) : 0.0;
         err0 = L2Error;
         dof0 = l2dofs;

         std::ios oldState(nullptr);
         oldState.copyfmt(std::cout);
         std::cout << std::right << std::setw(11) << iref << " | " 
                   << std::setw(10) <<  dof0 << " | " 
                   << std::setprecision(3) 
                   << std::setw(10) << std::scientific <<  err0 << " | " 
                   << std::setprecision(2) 
                   << std::setw(6) << std::fixed << rate_err << " | " 
                   << std::setw(6) << std::fixed << cg.GetNumIterations() << " | " 
                   << std::setprecision(3) 
                   << std::resetiosflags(std::ios::showbase)
                   << std::endl;
         std::cout.copyfmt(oldState);
      }

      if (visualization)
      {
         const char * keys = (iref == 0 && dim == 2) ? "jRcm\n" : "";
         char vishost[] = "localhost";
         int  visport   = 19916;
         common::VisualizeField(u_out,vishost, visport, u_gf, 
                                "Numerical u", 0,0, 500, 500, keys);
         common::VisualizeField(sigma_out,vishost, visport, sigma_gf,
                                "Numerical flux", 500,0,500, 500, keys);
      }

      if (iref == ref) break;

      mesh.UniformRefinement();
      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();
   }

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
   du.SetSize(X.Size());
   double alpha = M_PI * (X.Sum());
   u = sin(alpha);
   du.SetSize(X.Size());
   for (int i = 0; i<du.Size(); i++)
   {
      du[i] = M_PI * cos(alpha);
   }
   d2u = - M_PI*M_PI * u * du.Size();
}

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