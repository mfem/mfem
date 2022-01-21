//                 MFEM DPG_strong acoustics Example
//
// Compile with: make strong_dpg
//
// Definite/Indefinite Helmholtz

//     - Δ p ± ω^2 p = f̃ ,   in Ω
//                 p = p_0, on ∂Ω

// First Order System

//   ∇ p - ω u = 0, in Ω
// - ∇⋅u ± ω p = f, in Ω
//           p = p_0, in ∂Ω
// where f:=f̃/ω 

// Strong DPG formulation
// (p,u) ∈ H^1(Ω) × H(div,Ω)
// 
//   (∇ p, v) - ω (u,v) = 0,     in Ω, ∀ v ∈ (L^2)^dim
//  -(∇⋅u, q) ± ω (p,q) = (f,q), in Ω, ∀ q ∈  L^2 
//                    p = p_0, in ∂Ω
//
//   ((1,0)∇ p, v_0) - ω ((1,0)u,v_0) = 0,     in Ω, ∀ v_0 ∈ L^2
//   ((0,1)∇ p, v_1) - ω ((0,1)u,v_1) = 0,     in Ω, ∀ v_1 ∈ L^2
// in 3D
//   ((1,0,0)∇ p, v_0) - ω ((1,0,0)u,v_0) = 0,     in Ω, ∀ v_0 ∈ L^2
//   ((0,1,0)∇ p, v_1) - ω ((0,1,0)u,v_1) = 0,     in Ω, ∀ v_1 ∈ L^2
//   ((0,0,1)∇ p, v_2) - ω ((0,0,1)u,v_1) = 0,     in Ω, ∀ v_2 ∈ L^2

//  -(∇⋅u, q) ± ω (p,q) = (f,q), in Ω, ∀ q ∈  L^2 
//                    p = p_0, in ∂Ω

// ----------------------------------------------------
// |     |         p       |        u        |   RHS  | 
// ----------------------------------------------------
// | q   |   ± ω (p, q)    | - (∇⋅u, q)      |  (f,q) |
// |     |                 |                 |        |
// | v_0 | ((1,0)∇ p, v_0) | -ω ((1,0)u,v_0) |        |
// |     |                 |                 |        |
// | v_1 | ((0,1)∇ p, v_1) | -ω ((0,1)u,v_1) |        |

// where (q,v) ∈ L^2  × (L^2)^dim

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// #define DEFINITE

double p_exact(const Vector &x);
void u_exact(const Vector &x, Vector & u);
double rhs_func(const Vector &x);
void gradp_exact(const Vector &x, Vector &gradu);
double divu_exact(const Vector &x);
double d2_exact(const Vector &x);

int dim;
double omega;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
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
   // H1 space for p
   FiniteElementCollection *p_fec = new H1_FECollection(order, dim);
   FiniteElementSpace * p_fes = new FiniteElementSpace(&mesh, p_fec);

   // H(div) for u
   FiniteElementCollection *u_fec = new RT_FECollection(order-1, dim);
   FiniteElementSpace * u_fes = new FiniteElementSpace(&mesh, u_fec);

   // testspace fe collections
   int test_order = order+delta_order;
   FiniteElementCollection * q_fec = new L2_FECollection(test_order-1, dim);
   Array<FiniteElementCollection * > v_fec(dim);
   for (int i = 0; i<dim; i++)
   {
      v_fec[i] = new L2_FECollection(test_order-1, dim);
   }

   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);
   ConstantCoefficient omeg(omega);
   ConstantCoefficient negomeg(-omega);

   // Normal equation weak formulation
   Array<FiniteElementSpace * > trial_fes; 
   Array<FiniteElementCollection * > test_fec; 

   trial_fes.Append(p_fes);
   trial_fes.Append(u_fes);

   test_fec.Append(q_fec);
   test_fec.Append(v_fec);


   NormalEquations * a = new NormalEquations(trial_fes,test_fec);
   a->StoreMatrices(true);

//  ± ω (p, q)
#ifdef DEFINITE
//  ω (p, q)
   a->AddTrialIntegrator(new MassIntegrator(omeg),0,0);
#else
// -ω (p, q)
   a->AddTrialIntegrator(new MassIntegrator(negomeg),0,0);
#endif

// -(∇⋅u, q) 
   a->AddTrialIntegrator(new MixedScalarDivergenceIntegrator(negone),1,0);

// 
   Array<Vector *> vone(dim);
   Array<VectorConstantCoefficient *> vonecf(dim);
   Array<ScalarVectorProductCoefficient *> vnegomegcf(dim);
   for (int i = 0; i<dim; i++)
   {
      vone[i] = new Vector(dim);
      *vone[i] = 0.;
      (*vone[i])[i] = 1.;
      vonecf[i] = new VectorConstantCoefficient(*vone[i]);
      vnegomegcf[i] = new ScalarVectorProductCoefficient(negomeg,*vonecf[i]);
   }
   Vector vone_0(dim); vone_0[0] = 1; vone_0[1] = 0; 
   Vector vone_1(dim); vone_1[0] = 0; vone_1[1] = 1; 

   VectorConstantCoefficient vone0(vone_0);
   VectorConstantCoefficient vone1(vone_1);
   ScalarVectorProductCoefficient negomega0(negomeg,vone0);
   ScalarVectorProductCoefficient negomega1(negomeg,vone1);


   for (int i = 0; i<dim; i++)
   {
      // -ω (u,v)
      a->AddTrialIntegrator(new MixedDotProductIntegrator(*vnegomegcf[i]),1,i+1);
      // (∇ p, v)
      a->AddTrialIntegrator(new MixedDirectionalDerivativeIntegrator(*vonecf[i]),0,i+1);
      // (v,δv)
      a->AddTestIntegrator(new MassIntegrator(one),i+1,i+1);
   }    

   // (q,δq)
   a->AddTestIntegrator(new MassIntegrator(one),0,0);

   FunctionCoefficient f_rhs(rhs_func);
   a->AddDomainLFIntegrator(new DomainLFIntegrator(f_rhs),0);

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      p_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   FunctionCoefficient p_ex(p_exact);
   VectorFunctionCoefficient gradp_ex(dim,gradp_exact);
   VectorFunctionCoefficient u_ex(dim,u_exact);
   FunctionCoefficient divu_ex(divu_exact);
   GridFunction p_gf, u_gf;   
   GridFunction pex_gf(p_fes);   

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = p_fes->GetVSize();
   offsets[2] = u_fes->GetVSize();
   offsets.PartialSum();
   BlockVector x(offsets);
   x = 0.0;

   p_gf.MakeRef(p_fes,x.GetBlock(0));
   p_gf.ProjectBdrCoefficient(p_ex,ess_bdr);

   u_gf.MakeRef(u_fes,x.GetBlock(1));

   a->Assemble();

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

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream p_out;
      socketstream u_out;
      p_out.open(vishost, visport);
      u_out.open(vishost, visport);
      p_out.precision(8);
      p_out << "solution\n" << mesh << p_gf <<
               "window_title 'Numerical p' "
               << flush;

      u_out.precision(8);
      u_out << "solution\n" << mesh << u_gf <<
            "window_title 'Numerical flux' "
            << flush;
   }

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