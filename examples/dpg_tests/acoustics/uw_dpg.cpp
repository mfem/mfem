//                   MFEM Ultraweak DPG acoustics example
//
// Compile with: make uw_dpg
//

//     - Δ p ± ω^2 p = f̃ ,   in Ω
//                 p = p_0, on ∂Ω

// First Order System

//   ∇ p - ω u = 0, in Ω
// - ∇⋅u ± ω p = f, in Ω
//           p = p_0, in ∂Ω
// where f:=f̃/ω 

// UW-DPG:
// 
// p ∈ L^2(Ω), u ∈ (L^2(Ω))^dim 
// p̂ ∈ H^1/2(Ω), û ∈ H^-1/2(Ω)  
// -(p,  ∇⋅v) - ω (u , v) + < p̂, v⋅n> = 0,      ∀ v ∈ H(div,Ω)      
//  (u , ∇ q) ± ω (p , q) + < û, q >  = (f,q)   ∀ q ∈ H^1(Ω)
//                                  û = u_0     on ∂Ω 

// Note: 
// p̂ := p on Γ_h (skeleton)
// û := -u on Γ_h  

// -------------------------------------------------------------
// |   |     p     |     u     |    p̂      |    û    |  RHS    |
// -------------------------------------------------------------
// | v | -(p, ∇⋅v) | - ω (u,v) | < p̂, v⋅n> |         |         |
// |   |           |           |           |         |         |
// | q | ± ω (p,q) | (u , ∇ q) |           | < û,q > |  (f,q)  |  

// where (q,v) ∈  H^1(Ω) × H(div,Ω) 

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#define DEFINITE

double p_exact(const Vector &x);
void u_exact(const Vector &x, Vector & u);
double rhs_func(const Vector &x);
void gradp_exact(const Vector &x, Vector &gradu);
double divu_exact(const Vector &x);
double d2_exact(const Vector &x);
double hatp_exact(const Vector & X);
void hatu_exact(const Vector & X, Vector & hatu);

int dim;
double omega;

int main(int argc, char *argv[])
{
  const char *mesh_file = "../../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   bool visualization = true;
   double rnum=1.0;
   int ref = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");      
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");                  
   args.AddOption(&ref, "-ref", "--serial_ref",
                  "Number of serial refinements.");                               
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   omega = 2.0 * M_PI * rnum;


   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();


   for (int i = 0; i < ref; i++ )
   {
      mesh.UniformRefinement();
   }

   // Define spaces
   // L2 space for p
   FiniteElementCollection *p_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *p_fes = new FiniteElementSpace(&mesh,p_fec);

   // Vector L2 space for u 
   FiniteElementCollection *u_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *u_fes = new FiniteElementSpace(&mesh,u_fec, dim); 

   // H^1/2 space for p̂  
   FiniteElementCollection * hatp_fec = new H1_Trace_FECollection(order,dim);
   FiniteElementSpace *hatp_fes = new FiniteElementSpace(&mesh,hatp_fec);

   // H^-1/2 space for û  
   FiniteElementCollection * hatu_fec = new RT_Trace_FECollection(order-1,dim);   
   FiniteElementSpace *hatu_fes = new FiniteElementSpace(&mesh,hatu_fec);

   // testspace fe collections
   int test_order = order+delta_order;
   FiniteElementCollection * q_fec = new H1_FECollection(test_order, dim);
   FiniteElementCollection * v_fec = new RT_FECollection(test_order-1, dim);


   // Coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);
   ConstantCoefficient omeg(omega);
   ConstantCoefficient negomeg(-omega);

   // Normal equation weak formulation
   Array<FiniteElementSpace * > trial_fes; 
   Array<FiniteElementCollection * > test_fec; 

   trial_fes.Append(p_fes);
   trial_fes.Append(u_fes);
   trial_fes.Append(hatp_fes);
   trial_fes.Append(hatu_fes);

   test_fec.Append(q_fec);
   test_fec.Append(v_fec);

   NormalEquations * a = new NormalEquations(trial_fes,test_fec);
   a->StoreMatrices(true);



   // ± ω (p,q)
#ifdef DEFINITE
   a->AddTrialIntegrator(new MixedScalarMassIntegrator(omeg),0,0);
#else
   a->AddTrialIntegrator(new MixedScalarMassIntegrator(negomeg),0,0);
#endif   

// (u , ∇ q)
   a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(one)),1,0);

// -(p, ∇⋅v)
   a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(one),0,1);

// - ω (u,v)
   a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(negomeg)),1,1);

// < p̂, v⋅n>
   a->AddTrialIntegrator(new NormalTraceIntegrator,2,1);

// < û,q >
   a->AddTrialIntegrator(new TraceIntegrator,3,0);

   // test integrators (space-induced norm for H(div) × H1)
   // (∇q,∇δq)
   a->AddTestIntegrator(new DiffusionIntegrator(one),0,0);
   // (q,δq)
   a->AddTestIntegrator(new MassIntegrator(one),0,0);

   // (∇⋅v,∇⋅δv)
   a->AddTestIntegrator(new DivDivIntegrator(one),1,1);
   // (v,δv)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),1,1);


   // RHS
   FunctionCoefficient f_rhs(rhs_func);
   a->AddDomainLFIntegrator(new DomainLFIntegrator(f_rhs),0);


   FunctionCoefficient hatpex(hatp_exact);
   Array<int> elements_to_refine;
   GridFunction hatp_gf;


   socketstream p_out;
   socketstream u_out;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      p_out.open(vishost, visport);
      u_out.open(vishost, visport);
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
         hatp_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

   // shift the ess_tdofs
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         ess_tdof_list[i] += p_fes->GetTrueVSize() + u_fes->GetTrueVSize();
      }

      Array<int> offsets(5);
      offsets[0] = 0;
      offsets[1] = p_fes->GetVSize();
      offsets[2] = u_fes->GetVSize();
      offsets[3] = hatp_fes->GetVSize();
      offsets[4] = hatu_fes->GetVSize();
      offsets.PartialSum();
      BlockVector x(offsets);
      x = 0.0;
      hatp_gf.MakeRef(hatp_fes,x.GetBlock(2));
      hatp_gf.ProjectBdrCoefficient(hatpex,ess_bdr);

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
      double theta = 0.0;
      for (int iel = 0; iel<mesh.GetNE(); iel++)
      {
         if (residuals[iel] > theta * max_resid)
         {
            elements_to_refine.Append(iel);
         }
      }
      
      GridFunction p_gf;
      p_gf.MakeRef(p_fes,x.GetBlock(0));

      GridFunction u_gf;
      u_gf.MakeRef(u_fes,x.GetBlock(1));

      if (visualization)
      {
         p_out.precision(8);
         p_out << "solution\n" << mesh << p_gf <<
                  "window_title 'Numerical presure' "
                  << flush;

         u_out.precision(8);
         u_out << "solution\n" << mesh << u_gf <<
               "window_title 'Numerical velocity' "
               << flush;
      }

      mesh.GeneralRefinement(elements_to_refine);
      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();
   }

   delete a;
   delete q_fec;
   delete v_fec;
   delete hatp_fes;
   delete hatp_fec;
   delete hatu_fes;
   delete hatu_fec;
   delete u_fec;
   delete p_fec;
   delete u_fes;
   delete p_fes;

   return 0;
}

double rhs_func(const Vector &x)
{
   double p = p_exact(x);
   double divu = divu_exact(x);
   // f = - ∇⋅u ± ω p, 
#ifdef DEFINITE   
   return -divu + omega * p;
#else
   return -divu - omega * p;
#endif   
}

double p_exact(const Vector &x)
{
   return sin(omega*x.Sum());
}

void gradp_exact(const Vector &x, Vector &grad)
{
   grad.SetSize(x.Size());
   grad = omega * cos(omega * x.Sum());
}

void u_exact(const Vector &x, Vector & u)
{
   gradp_exact(x,u);
   u *= 1./omega;
}

double divu_exact(const Vector &x)
{
   return d2_exact(x)/omega;
}

double d2_exact(const Vector &x)
{
   return -dim * omega * omega * sin(omega*x.Sum());
}

double hatp_exact(const Vector & X)
{
   return p_exact(X);
}

void hatu_exact(const Vector & X, Vector & hatu)
{
   u_exact(X,hatu);
   hatu *= -1.;
}