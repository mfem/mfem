//                   MFEM Ultraweak DPG example for acoustics (Helmholtz)
//
// Compile with: make acoustics
//
// Sample runs
//
// acoustics -ref 5 -o 1 -rnum 1.0
// acoustics -m ../../data/inline-tri.mesh -ref 4 -o 2 -sc -rnum 3.0
// acoustics -m ../../data/amr-quad.mesh -ref 3 -o 3 -sc -rnum 4.5 -prob 1
// acoustics -m ../../data/inline-quad.mesh -ref 2 -o 4 -sc -rnum 11.5 -prob 1
// acoustics -m ../../data/inline-hex.mesh -ref 2 -o 2 -sc -rnum 1.0

// Description:
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the Helmholtz problem

//     - Δ p - ω² p = f̃ ,   in Ω
//                p = p₀, on ∂Ω

// It solves two kinds of problems
// a) f̃ = 0 and p₀ is a plane wave
// b) A manufactured solution problem where p_exact is a gaussian beam
// This example computes and prints out convergence rates for the L² error.

// The DPG UW deals with the First Order System
//  ∇ p + i ω u = 0, in Ω
//  ∇⋅u + i ω p = f, in Ω              (1)
//           p = p_0, in ∂Ω
// where f:=f̃/(i ω)

// Ultraweak-DPG is obtained by integration by parts of both equations and the
// introduction of trace unknowns on the mesh skeleton
//
// p ∈ L²(Ω), u ∈ (L²(Ω))ᵈⁱᵐ
// p̂ ∈ H^1/2(Ω), û ∈ H^-1/2(Ω)
// -(p,  ∇⋅v) + i ω (u , v) + < p̂, v⋅n> = 0,      ∀ v ∈ H(div,Ω)
// -(u , ∇ q) + i ω (p , q) + < û, q >  = (f,q)   ∀ q ∈ H¹(Ω)
//                                   p̂  = p₀      on ∂Ω

// Note:
// p̂ := p, û := u on the mesh skeleton

// For more information see https://doi.org/10.1016/j.camwa.2017.06.044

// -------------------------------------------------------------
// |   |     p     |     u     |    p̂      |    û    |  RHS    |
// -------------------------------------------------------------
// | v | -(p, ∇⋅v) | i ω (u,v) | < p̂, v⋅n> |         |         |
// |   |           |           |           |         |         |
// | q | i ω (p,q) |-(u , ∇ q) |           | < û,q > |  (f,q)  |

// where (q,v) ∈  H¹(Ω) × H(div,Ω)

// Here we use the "Adjoint Graph" norm on the test space i.e.,
// ||(q,v)||²ᵥ = ||A^*(q,v)||² + ||(q,v)||² where A is the
// acoustics operator defined by (1)

#include "mfem.hpp"
#include "util/complexweakform.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;

complex<double> acoustics_solution(const Vector & X);
void acoustics_solution_grad(const Vector & X,vector<complex<double>> &dp);
complex<double> acoustics_solution_laplacian(const Vector & X);

double p_exact_r(const Vector &x);
double p_exact_i(const Vector &x);
void u_exact_r(const Vector &x, Vector & u);
void u_exact_i(const Vector &x, Vector & u);
double rhs_func_r(const Vector &x);
double rhs_func_i(const Vector &x);
void gradp_exact_r(const Vector &x, Vector &gradu);
void gradp_exact_i(const Vector &x, Vector &gradu);
double divu_exact_r(const Vector &x);
double divu_exact_i(const Vector &x);
double d2_exact_r(const Vector &x);
double d2_exact_i(const Vector &x);
double hatp_exact_r(const Vector & X);
double hatp_exact_i(const Vector & X);
void hatu_exact_r(const Vector & X, Vector & hatu);
void hatu_exact_i(const Vector & X, Vector & hatu);

int dim;
double omega;

enum prob_type
{
   plane_wave,
   gaussian_beam
};

prob_type prob;

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   bool visualization = true;
   double rnum=1.0;
   int ref = 0;
   bool static_cond = false;
   int iprob = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelengths",
                  "Number of wavelengths");
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: plane wave, 1: Gaussian beam");
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&ref, "-ref", "--refinements",
                  "Number of serial refinements.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if (iprob > 1) { iprob = 0; }
   prob = (prob_type)iprob;

   omega = 2.*M_PI*rnum;

   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();
   MFEM_VERIFY(dim > 1, "Dimension = 1 is not supported in this example");

   // Define spaces
   enum TrialSpace
   {
      p_space     = 0,
      u_space     = 1,
      hatp_space  = 2,
      hatu_space  = 3
   };
   enum TestSpace
   {
      q_space = 0,
      v_space = 1
   };

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
   ConstantCoefficient zero(0.0);
   Vector vec0(dim); vec0 = 0.;
   VectorConstantCoefficient vzero(vec0);
   ConstantCoefficient negone(-1.0);
   ConstantCoefficient omeg(omega);
   ConstantCoefficient omeg2(omega*omega);
   ConstantCoefficient negomeg(-omega);

   Array<FiniteElementSpace * > trial_fes;
   Array<FiniteElementCollection * > test_fec;

   trial_fes.Append(p_fes);
   trial_fes.Append(u_fes);
   trial_fes.Append(hatp_fes);
   trial_fes.Append(hatu_fes);

   test_fec.Append(q_fec);
   test_fec.Append(v_fec);

   ComplexDPGWeakForm * a = new ComplexDPGWeakForm(trial_fes,test_fec);

   // i ω (p,q)
   a->AddTrialIntegrator(nullptr,new MixedScalarMassIntegrator(omeg),
                         TrialSpace::p_space,TestSpace::q_space);
   // -(u , ∇ q)
   a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(negone)),
                         nullptr,TrialSpace::u_space,TestSpace::q_space);
   // -(p, ∇⋅v)
   a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(one),nullptr,
                         TrialSpace::p_space,TestSpace::v_space);
   //  i ω (u,v)
   a->AddTrialIntegrator(nullptr,
                         new TransposeIntegrator(new VectorFEMassIntegrator(omeg)),
                         TrialSpace::u_space,TestSpace::v_space);
   // < p̂, v⋅n>
   a->AddTrialIntegrator(new NormalTraceIntegrator,nullptr,
                         TrialSpace::hatp_space,TestSpace::v_space);
   // < û,q >
   a->AddTrialIntegrator(new TraceIntegrator,nullptr,
                         TrialSpace::hatu_space,TestSpace::q_space);

   // test space integrators (Adjoint graph norm)
   // (∇q,∇δq)
   a->AddTestIntegrator(new DiffusionIntegrator(one),nullptr,
                        TestSpace::q_space, TestSpace::q_space);
   // (q,δq)
   a->AddTestIntegrator(new MassIntegrator(one),nullptr,
                        TestSpace::q_space, TestSpace::q_space);
   // (∇⋅v,∇⋅δv)
   a->AddTestIntegrator(new DivDivIntegrator(one),nullptr,
                        TestSpace::v_space, TestSpace::v_space);
   // (v,δv)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,
                        TestSpace::v_space, TestSpace::v_space);
   // -i ω (∇q,δv)
   a->AddTestIntegrator(nullptr,new MixedVectorGradientIntegrator(negomeg),
                        TestSpace::q_space, TestSpace::v_space);
   // i ω (v,∇ δq)
   a->AddTestIntegrator(nullptr,new MixedVectorWeakDivergenceIntegrator(negomeg),
                        TestSpace::v_space, TestSpace::q_space);
   // ω^2 (v,δv)
   a->AddTestIntegrator(new VectorFEMassIntegrator(omeg2),nullptr,
                        TestSpace::v_space, TestSpace::v_space);
   // - i ω (∇⋅v,δq)
   a->AddTestIntegrator(nullptr,new VectorFEDivergenceIntegrator(negomeg),
                        TestSpace::v_space, TestSpace::q_space);
   // i ω (q,∇⋅v)
   a->AddTestIntegrator(nullptr,new MixedScalarWeakGradientIntegrator(negomeg),
                        TestSpace::q_space, TestSpace::v_space);
   // ω^2 (q,δq)
   a->AddTestIntegrator(new MassIntegrator(omeg2),nullptr,
                        TestSpace::q_space, TestSpace::q_space);

   // RHS
   FunctionCoefficient f_rhs_r(rhs_func_r);
   FunctionCoefficient f_rhs_i(rhs_func_i);
   if (prob == prob_type::gaussian_beam)
   {
      a->AddDomainLFIntegrator(new DomainLFIntegrator(f_rhs_r),
                               new DomainLFIntegrator(f_rhs_i), TestSpace::q_space);
   }

   FunctionCoefficient hatpex_r(hatp_exact_r);
   FunctionCoefficient hatpex_i(hatp_exact_i);
   VectorFunctionCoefficient hatuex_r(dim,hatu_exact_r);
   VectorFunctionCoefficient hatuex_i(dim,hatu_exact_i);

   socketstream p_out_r;
   socketstream p_out_i;

   double err0 = 0.;
   int dof0;

   std::cout << "\n  Ref |"
             << "    Dofs    |"
             << "    ω    |"
             << "  L2 Error  |"
             << "  Rate  |"
             << " PCG it |" << endl;
   std::cout << std::string(60,'-')
             << endl;

   for (int it = 0; it<=ref; it++)
   {
      if (static_cond) { a->EnableStaticCondensation(); }
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
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         ess_tdof_list[j] += p_fes->GetTrueVSize() + u_fes->GetTrueVSize();
      }

      Array<int> offsets(5);
      offsets[0] = 0;
      offsets[1] = p_fes->GetVSize();
      offsets[2] = u_fes->GetVSize();
      offsets[3] = hatp_fes->GetVSize();
      offsets[4] = hatu_fes->GetVSize();
      offsets.PartialSum();

      Vector x(2*offsets.Last());
      x = 0.;

      GridFunction hatp_gf_r(hatp_fes, x, offsets[2]);
      GridFunction hatp_gf_i(hatp_fes, x, offsets.Last()+ offsets[2]);
      hatp_gf_r.ProjectBdrCoefficient(hatpex_r, ess_bdr);
      hatp_gf_i.ProjectBdrCoefficient(hatpex_i, ess_bdr);

      OperatorPtr Ah;
      Vector X,B;
      a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);

      // Setup operator and preconditioner
      ComplexOperator * Ahc = Ah.As<ComplexOperator>();

      BlockMatrix * A_r = dynamic_cast<BlockMatrix *>(&Ahc->real());
      BlockMatrix * A_i = dynamic_cast<BlockMatrix *>(&Ahc->imag());

      int num_blocks = A_r->NumRowBlocks();
      Array<int> tdof_offsets(2*num_blocks+1);
      tdof_offsets[0] = 0;
      int k = (static_cond) ? 2 : 0;
      for (int i=0; i<num_blocks; i++)
      {
         tdof_offsets[i+1] = trial_fes[i+k]->GetTrueVSize();
         tdof_offsets[num_blocks+i+1] = trial_fes[i+k]->GetTrueVSize();
      }
      tdof_offsets.PartialSum();

      BlockOperator A(tdof_offsets);
      for (int i = 0; i<num_blocks; i++)
      {
         for (int j = 0; j<num_blocks; j++)
         {
            A.SetBlock(i,j,&A_r->GetBlock(i,j));
            A.SetBlock(i,j+num_blocks,&A_i->GetBlock(i,j), -1.0);
            A.SetBlock(i+num_blocks,j+num_blocks,&A_r->GetBlock(i,j));
            A.SetBlock(i+num_blocks,j,&A_i->GetBlock(i,j));
         }
      }

      BlockDiagonalPreconditioner M(tdof_offsets);
      M.owns_blocks = 1;
      for (int i = 0; i<num_blocks; i++)
      {
         M.SetDiagonalBlock(i, new GSSmoother((SparseMatrix&)A_r->GetBlock(i,i)));
         M.SetDiagonalBlock(num_blocks+i, new GSSmoother((SparseMatrix&)A_r->GetBlock(i,
                                                                                      i)));
      }

      CGSolver cg;
      cg.SetRelTol(1e-10);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(M);
      cg.SetOperator(A);
      cg.Mult(B, X);

      a->RecoverFEMSolution(X,x);

      GridFunction p_r(p_fes, x, 0);
      GridFunction p_i(p_fes, x, offsets.Last());

      GridFunction u_r(u_fes, x, offsets[1]);
      GridFunction u_i(u_fes, x, offsets.Last() + offsets[1]);

      FunctionCoefficient p_ex_r(p_exact_r);
      FunctionCoefficient p_ex_i(p_exact_i);

      VectorFunctionCoefficient u_ex_r(dim,u_exact_r);
      VectorFunctionCoefficient u_ex_i(dim,u_exact_i);

      int dofs = 0;
      for (int i = 0; i<trial_fes.Size(); i++)
      {
         dofs += trial_fes[i]->GetTrueVSize();
      }

      double p_err_r = p_r.ComputeL2Error(p_ex_r);
      double p_err_i = p_i.ComputeL2Error(p_ex_i);
      double u_err_r = u_r.ComputeL2Error(u_ex_r);
      double u_err_i = u_i.ComputeL2Error(u_ex_i);

      double L2Error = sqrt(p_err_r*p_err_r + p_err_i*p_err_i
                            +u_err_r*u_err_r + u_err_i*u_err_i);

      double rate_err = (it) ? dim*log(err0/L2Error)/log((double)dof0/dofs) : 0.0;

      err0 = L2Error;
      dof0 = dofs;

      std::ios oldState(nullptr);
      oldState.copyfmt(std::cout);
      std::cout << std::right << std::setw(5) << it << " | "
                << std::setw(10) <<  dof0 << " | "
                << std::setprecision(1) << std::fixed
                << std::setw(4) <<  2*rnum << " π  | "
                << std::setprecision(3)
                << std::setw(10) << std::scientific << err0 << " | "
                << std::setprecision(2)
                << std::setw(6) << std::fixed << rate_err << " | "
                << std::setw(6) << std::fixed << cg.GetNumIterations() << " | "
                << std::endl;
      std::cout.copyfmt(oldState);

      if (visualization)
      {
         const char * keys = (it == 0 && dim == 2) ? "jRcml\n" : nullptr;
         char vishost[] = "localhost";
         int  visport   = 19916;
         VisualizeField(p_out_r,vishost, visport, p_r,
                        "Numerical presure (real part)", 0, 0, 500, 500, keys);
         VisualizeField(p_out_i,vishost, visport, p_i,
                        "Numerical presure (imaginary part)", 501, 0, 500, 500, keys);
      }

      if (it == ref)
      {
         break;
      }

      mesh.UniformRefinement();
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

double p_exact_r(const Vector &x)
{
   return acoustics_solution(x).real();
}

double p_exact_i(const Vector &x)
{
   return acoustics_solution(x).imag();
}

double hatp_exact_r(const Vector & X)
{
   return p_exact_r(X);
}

double hatp_exact_i(const Vector & X)
{
   return p_exact_i(X);
}

void gradp_exact_r(const Vector &x, Vector &grad_r)
{
   grad_r.SetSize(x.Size());
   vector<complex<double>> grad;
   acoustics_solution_grad(x,grad);
   for (unsigned i = 0; i < grad.size(); i++)
   {
      grad_r[i] = grad[i].real();
   }
}

void gradp_exact_i(const Vector &x, Vector &grad_i)
{
   grad_i.SetSize(x.Size());
   vector<complex<double>> grad;
   acoustics_solution_grad(x,grad);
   for (unsigned i = 0; i < grad.size(); i++)
   {
      grad_i[i] = grad[i].imag();
   }
}

double d2_exact_r(const Vector &x)
{

   return acoustics_solution_laplacian(x).real();
}

double d2_exact_i(const Vector &x)
{
   return acoustics_solution_laplacian(x).imag();
}

//  u = - ∇ p / (i ω )
//    = i (∇ p_r + i * ∇ p_i)  / ω
//    = - ∇ p_i / ω + i ∇ p_r / ω
void u_exact_r(const Vector &x, Vector & u)
{
   gradp_exact_i(x,u);
   u *= -1./omega;
}

void u_exact_i(const Vector &x, Vector & u)
{
   gradp_exact_r(x,u);
   u *= 1./omega;
}

void hatu_exact_r(const Vector & X, Vector & hatu)
{
   u_exact_r(X,hatu);
}
void hatu_exact_i(const Vector & X, Vector & hatu)
{
   u_exact_i(X,hatu);
}

//  ∇⋅u = i Δ p / ω
//      = i (Δ p_r + i * Δ p_i)  / ω
//      = - Δ p_i / ω + i Δ p_r / ω
double divu_exact_r(const Vector &x)
{
   return -d2_exact_i(x)/omega;
}

double divu_exact_i(const Vector &x)
{
   return d2_exact_r(x)/omega;
}

// f = ∇⋅u + i ω p
// f_r = ∇⋅u_r - ω p_i
double rhs_func_r(const Vector &x)
{
   double p = p_exact_i(x);
   double divu = divu_exact_r(x);
   return divu - omega * p;
}

// f_i = ∇⋅u_i + ω p_r
double rhs_func_i(const Vector &x)
{
   double p = p_exact_r(x);
   double divu = divu_exact_i(x);
   return divu + omega * p;
}

complex<double> acoustics_solution(const Vector & X)
{
   complex<double> zi = complex<double>(0., 1.);
   switch (prob)
   {
      case plane_wave:
      {
         double beta = omega/std::sqrt((double)X.Size());
         complex<double> alpha = beta * zi * X.Sum();
         return exp(alpha);
      }
      break;
      default:
      {
         double rk = omega;
         double degrees = 45;
         double alpha = (180+degrees) * M_PI/180.;
         double sina = sin(alpha);
         double cosa = cos(alpha);
         // shift the origin
         double xprim=X(0) + 0.1;
         double yprim=X(1) + 0.1;

         double  x = xprim*sina - yprim*cosa;
         double  y = xprim*cosa + yprim*sina;
         //wavelength
         double rl = 2.*M_PI/rk;
         // beam waist radius
         double w0 = 0.05;
         // function w
         double fact = rl/M_PI/(w0*w0);
         double aux = 1. + (fact*y)*(fact*y);
         double w = w0*sqrt(aux);
         double phi0 = atan(fact*y);
         double r = y + 1./y/(fact*fact);

         // pressure
         complex<double> ze = - x*x/(w*w) - zi*rk*y - zi * M_PI * x * x/rl/r +
                              zi*phi0/2.;

         double pf = pow(2.0/M_PI/(w*w),0.25);
         return pf*exp(ze);
      }
      break;
   }
}

void acoustics_solution_grad(const Vector & X, vector<complex<double>> & dp)
{
   dp.resize(X.Size());
   complex<double> zi = complex<double>(0., 1.);
   switch (prob)
   {
      case plane_wave:
      {
         double beta = omega/std::sqrt((double)X.Size());
         complex<double> alpha = beta * zi * X.Sum();
         complex<double> p = exp(alpha);
         for (int i = 0; i<X.Size(); i++)
         {
            dp[i] = zi * beta * p;
         }
      }
      break;
      default:
      {
         double rk = omega;
         double degrees = 45;
         double alpha = (180+degrees) * M_PI/180.;
         double sina = sin(alpha);
         double cosa = cos(alpha);
         // shift the origin
         double xprim=X(0) + 0.1;
         double yprim=X(1) + 0.1;

         double  x = xprim*sina - yprim*cosa;
         double  y = xprim*cosa + yprim*sina;
         double  dxdxprim = sina, dxdyprim = -cosa;
         double  dydxprim = cosa, dydyprim =  sina;
         //wavelength
         double rl = 2.*M_PI/rk;

         // beam waist radius
         double w0 = 0.05;

         // function w
         double fact = rl/M_PI/(w0*w0);
         double aux = 1. + (fact*y)*(fact*y);

         double w = w0*sqrt(aux);
         double dwdy = w0*fact*fact*y/sqrt(aux);

         double phi0 = atan(fact*y);
         double dphi0dy = cos(phi0)*cos(phi0)*fact;

         double r = y + 1./y/(fact*fact);
         double drdy = 1. - 1./(y*y)/(fact*fact);

         // pressure
         complex<double> ze = - x*x/(w*w) - zi*rk*y - zi * M_PI * x * x/rl/r +
                              zi*phi0/2.;

         complex<double> zdedx = -2.*x/(w*w) - 2.*zi*M_PI*x/rl/r;
         complex<double> zdedy = 2.*x*x/(w*w*w)*dwdy - zi*rk + zi*M_PI*x*x/rl/
                                 (r*r)*drdy + zi*dphi0dy/2.;

         double pf = pow(2.0/M_PI/(w*w),0.25);
         double dpfdy = -pow(2./M_PI/(w*w),-0.75)/M_PI/(w*w*w)*dwdy;

         complex<double> zp = pf*exp(ze);
         complex<double> zdpdx = zp*zdedx;
         complex<double> zdpdy = dpfdy*exp(ze)+zp*zdedy;

         dp[0] = (zdpdx*dxdxprim + zdpdy*dydxprim);
         dp[1] = (zdpdx*dxdyprim + zdpdy*dydyprim);
         if (dim == 3) { dp[2] = 0.0; }
      }
      break;
   }
}

complex<double> acoustics_solution_laplacian(const Vector & X)
{
   complex<double> zi = complex<double>(0., 1.);
   switch (prob)
   {
      case plane_wave:
      {
         double beta = omega/std::sqrt((double)X.Size());
         complex<double> alpha = beta * zi * X.Sum();
         complex<double> p = exp(alpha);
         return dim * beta * beta * p;
      }
      break;
      default:
      {
         double rk = omega;
         double degrees = 45;
         double alpha = (180+degrees) * M_PI/180.;
         double sina = sin(alpha);
         double cosa = cos(alpha);
         // shift the origin
         double xprim=X(0) + 0.1;
         double yprim=X(1) + 0.1;

         double  x = xprim*sina - yprim*cosa;
         double  y = xprim*cosa + yprim*sina;
         double  dxdxprim = sina, dxdyprim = -cosa;
         double  dydxprim = cosa, dydyprim =  sina;
         //wavelength
         double rl = 2.*M_PI/rk;

         // beam waist radius
         double w0 = 0.05;

         // function w
         double fact = rl/M_PI/(w0*w0);
         double aux = 1. + (fact*y)*(fact*y);

         double w = w0*sqrt(aux);
         double dwdy = w0*fact*fact*y/sqrt(aux);
         double d2wdydy = w0*fact*fact*(1. - (fact*y)*(fact*y)/aux)/sqrt(aux);

         double phi0 = atan(fact*y);
         double dphi0dy = cos(phi0)*cos(phi0)*fact;
         double d2phi0dydy = -2.*cos(phi0)*sin(phi0)*fact*dphi0dy;

         double r = y + 1./y/(fact*fact);
         double drdy = 1. - 1./(y*y)/(fact*fact);
         double d2rdydy = 2./(y*y*y)/(fact*fact);

         // pressure
         complex<double> ze = - x*x/(w*w) - zi*rk*y - zi * M_PI * x * x/rl/r +
                              zi*phi0/2.;

         complex<double> zdedx = -2.*x/(w*w) - 2.*zi*M_PI*x/rl/r;
         complex<double> zdedy = 2.*x*x/(w*w*w)*dwdy - zi*rk + zi*M_PI*x*x/rl/
                                 (r*r)*drdy + zi*dphi0dy/2.;
         complex<double> zd2edxdx = -2./(w*w) - 2.*zi*M_PI/rl/r;
         complex<double> zd2edxdy = 4.*x/(w*w*w)*dwdy + 2.*zi*M_PI*x/rl/(r*r)*drdy;
         complex<double> zd2edydx = zd2edxdy;
         complex<double> zd2edydy = -6.*x*x/(w*w*w*w)*dwdy*dwdy + 2.*x*x/
                                    (w*w*w)*d2wdydy - 2.*zi*M_PI*x*x/rl/(r*r*r)*drdy*drdy
                                    + zi*M_PI*x*x/rl/(r*r)*d2rdydy + zi/2.*d2phi0dydy;

         double pf = pow(2.0/M_PI/(w*w),0.25);
         double dpfdy = -pow(2./M_PI/(w*w),-0.75)/M_PI/(w*w*w)*dwdy;
         double d2pfdydy = -1./M_PI*pow(2./M_PI,-0.75)*(-1.5*pow(w,-2.5)
                                                        *dwdy*dwdy + pow(w,-1.5)*d2wdydy);

         complex<double> zp = pf*exp(ze);
         complex<double> zdpdx = zp*zdedx;
         complex<double> zdpdy = dpfdy*exp(ze)+zp*zdedy;
         complex<double> zd2pdxdx = zdpdx*zdedx + zp*zd2edxdx;
         complex<double> zd2pdxdy = zdpdy*zdedx + zp*zd2edxdy;
         complex<double> zd2pdydx = dpfdy*exp(ze)*zdedx + zdpdx*zdedy + zp*zd2edydx;
         complex<double> zd2pdydy = d2pfdydy*exp(ze) + dpfdy*exp(
                                       ze)*zdedy + zdpdy*zdedy + zp*zd2edydy;


         return (zd2pdxdx*dxdxprim + zd2pdydx*dydxprim)*dxdxprim
                + (zd2pdxdy*dxdxprim + zd2pdydy*dydxprim)*dydxprim
                + (zd2pdxdx*dxdyprim + zd2pdydx*dydyprim)*dxdyprim
                + (zd2pdxdy*dxdyprim + zd2pdydy*dydyprim)*dydyprim;
      }
      break;
   }
}
