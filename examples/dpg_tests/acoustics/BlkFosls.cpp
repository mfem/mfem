//                 MFEM FOSLS acoustics Example
//
// Compile with: make fosls
//
// Definite/Indefinite Helmholtz

//     - Δ p ± ω^2 p = f̃ ,   in Ω
//                 p = p_0, on ∂Ω

// First Order System

//   ∇ p - ω u = 0, in Ω
// - ∇⋅u ± ω p = f, in Ω
//           p = p_0, in ∂Ω
// where f:=f̃/ω 

// FOSLS:
//       minimize  1/2(||∇p - ω u||^2 + ||-∇⋅u ± ω p - f||^2)

// (p,u) ∈ H^1(Ω) × H(div,Ω)
// -------------------------------------------------------------------
// |   |            p           |            u             |   RHS    | 
// -------------------------------------------------------------------
// | q |  (∇ p,∇ q) + ω^2(p,q)  | ∓ ω (∇⋅u,q) - ω (u, ∇ q) | ± ω(f,q) |
// |   |                        |                          |          |
// | v | ∓ ω (p,∇⋅v) - ω (∇ p,v)|  (∇⋅u,∇⋅v) + ω^2 (u,v)   | -(f,∇⋅v) |

// where (q,v) ∈ H^1(Ω) × H(div,Ω)

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
   bool visualization = true;
   double rnum=1.0;
   int sr = 1;

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
   args.AddOption(&sr, "-sr", "--serial_ref",
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

   for (int i = 0; i < sr; i++ )
   {
      mesh.UniformRefinement();
   }

   FiniteElementCollection *H1fec = new H1_FECollection(order, dim);
   FiniteElementCollection *RTfec = new RT_FECollection(order-1, dim);
   FiniteElementSpace * H1fes = new FiniteElementSpace(&mesh, H1fec);
   FiniteElementSpace * RTfes = new FiniteElementSpace(&mesh, RTfec);

   Array<FiniteElementSpace *> fespaces(2);
   fespaces[0] = H1fes;
   fespaces[1] = RTfes;

   Array<int> ess_bdr;
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespaces[0]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   BlockBilinearForm a(fespaces);
   a.SetDiagonalPolicy(mfem::Operator::DIAG_KEEP);
   cout << "H1 fespace = " << H1fes->GetTrueVSize() << endl;
   cout << "RT fespace = " << RTfes->GetTrueVSize() << endl;

   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);
   ConstantCoefficient omeg(omega);
   ConstantCoefficient negomeg(-omega);
   ConstantCoefficient omeg2(omega*omega);


   Array2D<BilinearFormIntegrator * > blfi(2,2);

   // blfi(0,0) = (∇ p,∇ q) + ω^2(p,q)
   SumIntegrator * integ00 = new SumIntegrator();
   integ00->AddIntegrator(new DiffusionIntegrator(one));
   integ00->AddIntegrator(new MassIntegrator(omeg2));
   blfi(0,0) = integ00;

   // blfi(0,1) = ∓ ω (∇⋅u,q) - ω (u, ∇ q)
   SumIntegrator * integ01 = new SumIntegrator();
#ifdef DEFINITE
   // -ω (∇⋅u,q)
   integ01->AddIntegrator(new MixedScalarDivergenceIntegrator(negomeg));
#else   
   // ω (∇⋅u,q)
   integ01->AddIntegrator(new MixedScalarDivergenceIntegrator(omeg));
#endif
   // - ω (u, ∇ q)
   integ01->AddIntegrator(new MixedVectorWeakDivergenceIntegrator(omeg));
   blfi(0,1) = integ01;

   // blfi(1,0) = ∓ ω (p,∇⋅v) - ω (∇ p,v)
   SumIntegrator * integ10 = new SumIntegrator();
#ifdef DEFINITE
   // - ω (p,∇⋅v)
   integ10->AddIntegrator(new MixedScalarWeakGradientIntegrator(omeg));
#else
   // ω (p,∇⋅v)
   integ10->AddIntegrator(new MixedScalarWeakGradientIntegrator(negomeg));
#endif
   // - ω (∇ p,v)
   integ10->AddIntegrator(new MixedVectorGradientIntegrator(negomeg));
   blfi(1,0) = integ10;

   // blfi(1,1) = (∇⋅u,∇⋅v) + ω^2 (u,v)
   SumIntegrator * integ11 = new SumIntegrator();
   integ11->AddIntegrator(new DivDivIntegrator(one));
   integ11->AddIntegrator(new VectorFEMassIntegrator(omeg2));
   blfi(1,1) = integ11;


   BlockLinearForm b(fespaces);
   Array<LinearFormIntegrator * > lfi(2);
   // ± ω (f,q)
   FunctionCoefficient f_rhs(rhs_func);
#ifdef DEFINITE  
   ProductCoefficient w_f(omeg,f_rhs);
#else   
   ProductCoefficient w_f(negomeg,f_rhs);
#endif   
   // lfi[0] = new DomainLFIntegrator(w_f);
   lfi[0] = new DomainLFIntegrator(w_f);


   //  -(f,∇⋅v)
   ProductCoefficient neg_f(negone,f_rhs);
   // lfi[1] = new VectorFEDomainLFDivIntegrator(f_rhs);
   lfi[1] = new VectorFEDomainLFDivIntegrator(neg_f);

   TestBlockBilinearFormIntegrator * integ = new TestBlockBilinearFormIntegrator();
   integ->SetIntegrators(blfi);
   a.AddDomainIntegrator(integ);
   a.Assemble();

   TestBlockLinearFormIntegrator * lininteg = new TestBlockLinearFormIntegrator();
   lininteg->SetIntegrators(lfi);
   b.AddDomainIntegrator(lininteg);
   b.Assemble();

   int size = 0;
   for (int i = 0; i<fespaces.Size(); i++)
   {
      size += fespaces[i]->GetVSize();
   }

   Vector x(size);
   x = 0.0;
   FunctionCoefficient p_ex(p_exact);
   VectorFunctionCoefficient gradp_ex(dim,gradp_exact);
   VectorFunctionCoefficient u_ex(dim,u_exact);
   FunctionCoefficient divu_ex(divu_exact);
   GridFunction p_gf, u_gf;   
   GridFunction pex_gf(H1fes);   

   p_gf.MakeRef(H1fes,x,0);
   // p_gf.ProjectBdrCoefficient(p_ex,ess_bdr);
   p_gf.ProjectCoefficient(p_ex);
   pex_gf.ProjectCoefficient(p_ex);

   u_gf.MakeRef(RTfes,x,H1fes->GetVSize());
   u_gf = 0.;


   OperatorPtr A;
   Vector X,B;
   a.FormLinearSystem(ess_tdof_list,x,b,A,X,B);

   GSSmoother M((SparseMatrix&)(*A));
   CGSolver cg;
   cg.SetRelTol(1e-10);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(M);
   cg.SetOperator(*A);
   cg.Mult(B, X);

   a.RecoverFEMSolution(X,b,x);

   p_gf.MakeRef(H1fes,x,0);
   u_gf.MakeRef(RTfes,x,H1fes->GetVSize());


   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream solu_sock(vishost, visport);
      solu_sock.precision(8);
      solu_sock << "solution\n" << mesh << p_gf <<
                "window_title 'Numerical u' "
                << flush;
      // socketstream sols_sock(vishost, visport);
      // sols_sock.precision(8);
      // sols_sock << "solution\n" << mesh << u_gf <<
               //  "window_title 'Numerical sigma' "
               //  << flush;

      socketstream solex_sock(vishost, visport);
      solex_sock.precision(8);
      solex_sock << "solution\n" << mesh << pex_gf <<
                    "window_title 'Exact p' "
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