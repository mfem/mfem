//                                MFEM Ultraweak DPG example
//
// Compile with: make ejproblem
//
// sample runs 
// ./ejproblem -m ../../data/inline-quad.mesh -o 2 -ref 20 -graph-norm -do 1

//     - εΔu + ∇⋅(βu) = f,   in Ω
//                  u = u_0, on ∂Ω

// First Order System

//     - ∇⋅σ + ∇⋅(βu) = f,   in Ω
//        1/ε σ - ∇u  = 0,   in Ω
//                  u = u_0, on ∂Ω

// UW-DPG:
// 
// u ∈ L^2(Ω), σ ∈ (L^2(Ω))^dim 
// û ∈ H^1/2, σ̂ ∈ H^-1/2  
// -(βu , ∇v)  + (σ , ∇v)     + < f̂ ,  v  > = (f,v),   ∀ v ∈ H^1(Ω)      
//   (u , ∇⋅τ) + 1/ε (σ , τ)  + < û , τ⋅n > = 0,       ∀ τ ∈ H(div,Ω)
//                                        û = u_0  on ∂Ω 

// Note: 
// f̂ := βu - σ
// û := -u

// -------------------------------------------------------------
// |   |     u     |     σ     |   û       |     f̂    |  RHS    |
// -------------------------------------------------------------
// | v |-(βu , ∇v) | (σ , ∇v)  |           | < f̂ ,v > |  (f,v)  |
// |   |           |           |           |          |         |
// | τ | (u ,∇⋅τ)  | 1/ε(σ , τ)|  <û,τ⋅n>  |          |    0    |  

// where (v,τ) ∈  H^1(Ω_h) × H(div,Ω_h)  

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

enum prob_type
{
   EJ,   
   general  
};

prob_type prob;
Vector beta;
double epsilon = 0.5;
// Function returns the solution u, and gradient du and the Laplacian d2u
void solution(const Vector & x, double & u, Vector & du, double & d2u);


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
   // σ = ε ∇ u
   sigma = du;
   sigma *= epsilon;
}

double exact_hatu(const Vector & X)
{
   return -exact_u(X);
}

void exact_hatf(const Vector & X, Vector & hatf)
{
   Vector sigma;
   exact_sigma(X,sigma);
   double u = exact_u(X);
   hatf.SetSize(X.Size());
   for (int i = 0; i<hatf.Size(); i++)
   {
      hatf[i] = beta[i] * u - sigma[i];
   }
}

double f_exact(const Vector & X)
{
   // f = - εΔu - ∇⋅(βu)
   double u, d2u;
   Vector du;
   solution(X,u,du,d2u);

   double s = 0;
   for (int i = 0; i<du.Size(); i++)
   {
      s += beta[i] * du[i];
   }
   return -epsilon * d2u + s;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   int ref = 1;
   bool adjoint_graph_norm = false;
   bool visualization = true;
   int iprob = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&epsilon, "-eps", "--epsilon",
                  "Epsilon coefficient");               
   args.AddOption(&ref, "-ref", "--num_refinements",
                  "Number of uniform refinements");        
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: lshape, 1: General");                         
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

   if (iprob > 1) { iprob = 1; }
   prob = (prob_type)iprob;

   if (prob == prob_type::EJ)
   {
      mesh_file = "../../data/inline-quad.mesh";
   }

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   beta.SetSize(dim);
   beta[0] = 1.;
   beta[1] = 0.;

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
   FiniteElementCollection * hatf_fec = new RT_Trace_FECollection(order-1,dim);   
   FiniteElementSpace *hatf_fes = new FiniteElementSpace(&mesh,hatf_fec);

   // testspace fe collections
   int test_order = order+delta_order;
   FiniteElementCollection * v_fec = new H1_FECollection(test_order, dim);
   FiniteElementCollection * tau_fec = new RT_FECollection(test_order-1, dim);

   // Coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);
   ConstantCoefficient eps(epsilon);
   ConstantCoefficient eps1(1./epsilon);
   ConstantCoefficient negeps1(-1./epsilon);
   ConstantCoefficient eps2(1/(epsilon*epsilon));

   ConstantCoefficient negeps(-epsilon);
   VectorConstantCoefficient betacoeff(beta);
   Vector negbeta = beta;
   negbeta.Neg();

   DenseMatrix bbt(beta.Size());
   MultVVt(beta, bbt); 
   MatrixConstantCoefficient bbtcoeff(bbt);


   VectorConstantCoefficient negbetacoeff(negbeta);
   // Normal equation weak formulation
   Array<FiniteElementSpace * > trial_fes; 
   Array<FiniteElementCollection * > test_fec; 

   trial_fes.Append(u_fes);
   trial_fes.Append(sigma_fes);
   trial_fes.Append(hatu_fes);
   trial_fes.Append(hatf_fes);

   test_fec.Append(v_fec);
   test_fec.Append(tau_fec);

   NormalEquations * a = new NormalEquations(trial_fes,test_fec);
   a->StoreMatrices(true);

   //-(βu , ∇v)
   a->AddTrialIntegrator(new MixedScalarWeakDivergenceIntegrator(betacoeff),0,0);
   
   // (σ,∇ v)
   a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(one)),1,0);

   // (u ,∇⋅τ)
   a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(negone),0,1);

   // 1/ε (σ,τ)
   a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(eps1)),1,1);

   //  <û,τ⋅n>
   a->AddTrialIntegrator(new NormalTraceIntegrator,2,1);

   // <f̂ ,v> 
   a->AddTrialIntegrator(new TraceIntegrator,3,0);


   //test integrators (space-induced norm for H1 × H(div))
   // (∇v,∇δv)
   a->AddTestIntegrator(new DiffusionIntegrator(one),0,0);
   // (v,δv)
   a->AddTestIntegrator(new MassIntegrator(one),0,0);
   // (∇⋅τ,∇⋅δτ)
   a->AddTestIntegrator(new DivDivIntegrator(one),1,1);
   // (τ,δτ)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),1,1);

   // additional terms for adjoint graph norm
   if (adjoint_graph_norm)
   {
      //test integrators (Adoint graph norm)
      // 1/ε^2 (τ,δτ)
      a->AddTestIntegrator(new VectorFEMassIntegrator(eps2),1,1);

      // 1/ε (∇v, δτ)
      a->AddTestIntegrator(new MixedVectorGradientIntegrator(eps1),0,1);

      // 1/ε (τ,∇δv) 
      a->AddTestIntegrator(new MixedVectorWeakDivergenceIntegrator(negeps1),1,0);

      // (β⋅∇v, β⋅∇δv)   
      a->AddTestIntegrator(new DiffusionIntegrator(bbtcoeff), 0,0);
      // - (β ⋅ ∇v,∇⋅δτ)  
      a->AddTestIntegrator(new MixedGradDivIntegrator(betacoeff),0,1);

      // -(β ∇⋅τ ,∇⋅δv)  
      a->AddTestIntegrator(new MixedDivGradIntegrator(betacoeff),1,0);

   }

   FunctionCoefficient f(f_exact);
   if (prob == prob_type::general)
   {
      a->AddDomainLFIntegrator(new DomainLFIntegrator(f),0);
   }

   FunctionCoefficient hatuex(exact_hatu);
   Array<int> elements_to_refine;
   FunctionCoefficient uex(exact_u);
   GridFunction hatu_gf;

   socketstream uex_out;
   socketstream u_out;
   socketstream sigma_out;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      u_out.open(vishost, visport);
      uex_out.open(vishost, visport);
      sigma_out.open(vishost, visport);
   }


   for (int i = 0; i<ref; i++)
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
      offsets[4] = hatf_fes->GetVSize();
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
      Vector & residuals = a->ComputeResidual(x);

      double residual = residuals.Norml2();
      cout << "Residual = " << residual << endl;

      elements_to_refine.SetSize(0);
      double max_resid = residuals.Max();
      double theta = 0.7;
      for (int iel = 0; iel<mesh.GetNE(); iel++)
      {
         if (residuals[iel] > theta * max_resid)
         {
            elements_to_refine.Append(iel);
         }
      }   

      GridFunction u_gf;
      u_gf.MakeRef(u_fes,x.GetBlock(0));

      GridFunction uex_gf(u_fes);
      uex_gf.ProjectCoefficient(uex);


      GridFunction sigma_gf;
      sigma_gf.MakeRef(sigma_fes,x.GetBlock(1));

      if (visualization)
      {
         uex_out.precision(8);
         uex_out << "solution\n" << mesh << uex_gf <<
                  "window_title 'Exact u' "
                  << flush;
         u_out.precision(8);
         u_out << "solution\n" << mesh << u_gf <<
                  "window_title 'Numerical u' "
                  << flush;

         sigma_out.precision(8);
         sigma_out << "solution\n" << mesh << sigma_gf <<
               "window_title 'Numerical flux' "
               << flush;
      }

      mesh.GeneralRefinement(elements_to_refine,1,1);
      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();

   }

   delete a;
   delete tau_fec;
   delete v_fec;
   delete hatf_fes;
   delete hatf_fec;
   delete hatu_fes;
   delete hatu_fec;
   delete sigma_fec;
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
      case EJ:
      {
         double alpha = sqrt(1. + 4. * epsilon * epsilon * M_PI * M_PI);
         double r1 = (1. + alpha) / (2.*epsilon);
         double r2 = (1. - alpha) / (2.*epsilon);
         double denom = exp(-r2) - exp(-r1);

         double x = X[0];
         double y = X[1];

         u = (exp(r2 * (x-1.)) - exp(r1 * (x-1.))) * cos(M_PI * y);
         u/=denom;
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