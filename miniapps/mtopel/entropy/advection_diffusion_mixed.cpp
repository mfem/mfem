//                                MFEM Obstacle Problem
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


double eps = 1.0;
double gamma = 0.0;
double L2_weight = 0.0;
double Ramp_BC(const Vector &pt);
double EJ_exact_solution(const Vector &pt);

double lnit(double x)
{
   double tol = 1e-13;
   x = min(max(tol,x),1.0-tol);
   // MFEM_ASSERT(x>0.0, "Argument must be > 0");
   // MFEM_ASSERT(x<1.0, "Argument must be < 1");
   return log(x/(1.0-x));
}

double expit(double x)
{
   if (x >= 0)
   {
      return 1.0/(1.0+exp(-x));
   }
   else
   {
      return exp(x)/(1.0+exp(x));
   }
}

double dexpitdx(double x)
{
   double tmp = expit(-x);
   return tmp - pow(tmp,2);
}

class LnitGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double min_val;
   double max_val;

public:
   LnitGridFunctionCoefficient(GridFunction &u_, double min_val_=-1e10, double max_val_=1e10)
      : u(&u_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ExpitGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double min_val;
   double max_val;

public:
   ExpitGridFunctionCoefficient(GridFunction &u_, double min_val_=0.0, double max_val_=1.0)
      : u(&u_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class dExpitdxGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double min_val;
   double max_val;

public:
   dExpitdxGridFunctionCoefficient(GridFunction &u_, double min_val_=0.0, double max_val_=1.0)
      : u(&u_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../../data/inline-quad.mesh";
   int order = 1;
   bool visualization = true;
   int max_it = 5;
   int ref_levels = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&ref_levels, "-r", "--refs",
                  "Number of h-refinements.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iterations");
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

   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   H1_FECollection H1fec(order, dim);
   // H1_FECollection H1fec(order, dim, BasisType::Positive);
   FiniteElementSpace H1fes(&mesh, &H1fec);

   L2_FECollection L2fec(order-1, dim);
   FiniteElementSpace L2fes(&mesh, &L2fec);

   cout << "Number of finite element unknowns: "
        << H1fes.GetTrueVSize() 
        << " "
        << L2fes.GetTrueVSize() << endl;

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = H1fes.GetVSize();
   offsets[2] = L2fes.GetVSize();
   offsets.PartialSum();

   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   GridFunction u_gf, delta_psi_gf;
   u_gf.MakeRef(&H1fes,x.GetBlock(0));
   delta_psi_gf.MakeRef(&L2fes,x.GetBlock(1));
   delta_psi_gf = 0.0;

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr = 1;
      H1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   auto rhs_func = [](const Vector &x)
   {
      double N = 5;
      double val = 1.0 / 2.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(N*2.0*M_PI*x(i));
      }
      return (eps * x.Size() * pow(N*2.0*M_PI,2) + gamma) * val ;
   };

   auto sol_func = [](const Vector &x)
   {
      double N = 5;
      double val = 1.0 / 2.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(N*2.0*M_PI*x(i));
      }
      return val + 0.5;
   };

   auto perturbation_func = [](const Vector &x)
   {
      double scale = 5e-2;
      double val = 1.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(M_PI*x(i));
      }
      return 1.0 + scale * pow(val, 3.0);
   };
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   ConstantCoefficient L2_weight_coeff(L2_weight);
   
   Vector zero_vec(dim);
   zero_vec = 0.0;

   GridFunction u_old_gf(&H1fes);
   GridFunction psi_old_gf(&L2fes);
   GridFunction psi_gf(&L2fes);
   u_old_gf = 0.0;

   /////////// Example 1   
   u_gf = 0.5;
   Vector beta(dim);
   beta(0) = 0.0;
   beta(1) = 0.0;
   VectorConstantCoefficient beta_coeff(beta);
   // VectorConstantCoefficient beta_coeff(zero_vec);
   FunctionCoefficient f(rhs_func);
   ConstantCoefficient bdry_coef(0.5);
   u_gf.ProjectBdrCoefficient(bdry_coef, ess_bdr);
   double alpha0 = 1.0;

   // /////////// Example 2
   // u_gf = 0.5;
   // Vector beta(dim);
   // beta(0) = 1.0;
   // beta(1) = 0.5;
   // beta /= sqrt(1.25);
   // ConstantCoefficient f(0.0);
   // FunctionCoefficient bdry_coef(Ramp_BC);
   // VectorConstantCoefficient beta_coeff(beta);
   // double alpha0 = 1.0;
   // u_gf.ProjectCoefficient(bdry_coef);
   // u_gf.ProjectBdrCoefficient(bdry_coef, ess_bdr);

   // /////////// Example 3
   // // u_gf = 0.5;
   // Vector beta(dim);
   // beta(0) = 1.0;
   // beta(1) = 0.0;
   // ConstantCoefficient f(0.0);
   // FunctionCoefficient bdry_coef(EJ_exact_solution);
   // VectorConstantCoefficient beta_coeff(beta);
   // FunctionCoefficient perturbation(perturbation_func);
   // ProductCoefficient IC_coeff(bdry_coef, perturbation);
   // u_gf.ProjectCoefficient(bdry_coef);
   // // u_gf.ProjectCoefficient(IC_coeff);
   // // u_gf.ProjectBdrCoefficient(bdry_coef, ess_bdr);
   // double alpha0 = 1.0;
   // // double alpha0 = 0.5;

   ConstantCoefficient eps_coeff(eps);
   OuterProductCoefficient beta_beta(beta_coeff, beta_coeff);
   ConstantCoefficient gamma_coeff(gamma);

   // Solve linear system to get initial guess
   LinearForm b(&H1fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(f));
   b.Assemble();

   BilinearForm a(&H1fes);
   a.AddDomainIntegrator(new DiffusionIntegrator(eps_coeff));
   a.AddDomainIntegrator(new ConvectionIntegrator(beta_coeff));
   a.AddDomainIntegrator(new MassIntegrator(gamma_coeff));
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, u_gf, b, A, X, B);

   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(*A);
   umf_solver.Mult(B, X);

   a.RecoverFEMSolution(X, b, u_gf);
   u_old_gf = u_gf;

   // Construct reference "high order" solution
   GridFunction u_ref(&H1fes);
   u_ref = u_gf;
   for (int i = 0; i < u_ref.Size(); i++)
   {
      if (u_ref(i) > 1.0)
      {
         u_ref(i) = 1.0;
      }
      else if (u_ref(i) < 0.0)
      {
         u_ref(i) = 0.0;
      }
   }

   LnitGridFunctionCoefficient lnit_u(u_gf);
   psi_gf.ProjectCoefficient(lnit_u);
   psi_old_gf = psi_gf;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   ExpitGridFunctionCoefficient expit_psi(psi_gf);
   GridFunction u_alt_gf(&L2fes);
   u_alt_gf.ProjectCoefficient(expit_psi);
   sol_sock << "solution\n" << mesh << u_alt_gf << "window_title 'Discrete solution'" << flush;

   // sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'" << flush;

   // 12. Iterate
   int k;
   int total_iterations = 0;
   double increment_u = 1e-3;
   double increment_psi = 1e-4;
   for (k = 0; k < max_it; k++)
   {
      // double alpha = alpha0 / sqrt(k+1);
      // double alpha = alpha0 * sqrt(k+1);
      double alpha = alpha0;
      // alpha *= 2;

      GridFunction u_tmp(&H1fes);
      u_tmp = u_old_gf;

      int j;
      for ( j = 0; j < 15; j++)
      {
         total_iterations++;

         // A. Assembly

         LinearForm b0,b1;
         b0.Update(&H1fes,rhs.GetBlock(0),0);
         b1.Update(&L2fes,rhs.GetBlock(1),0);

         ExpitGridFunctionCoefficient expit_psi(psi_gf);
         dExpitdxGridFunctionCoefficient dexpitdx_psi(psi_gf);
         ProductCoefficient neg_dexpitdx_psi(-1.0, dexpitdx_psi);

         GridFunctionCoefficient u_old_cf(&u_old_gf);
         GradientGridFunctionCoefficient grad_u_old(&u_old_gf);
         InnerProductCoefficient beta_grad_u_old(beta_coeff, grad_u_old);
         MatrixVectorProductCoefficient beta_beta_grad_u_old(beta_beta, grad_u_old);
         ScalarVectorProductCoefficient alpha_eps_grad_u_old(eps*(1.0-alpha), grad_u_old);
         SumCoefficient LHS2(u_old_cf, beta_grad_u_old, L2_weight, -alpha);
         GridFunctionCoefficient psi_cf(&psi_gf);
         GridFunctionCoefficient psi_old_cf(&psi_old_gf);
         SumCoefficient psi_old_minus_psi(psi_old_cf, psi_cf, 1.0, -1.0);
         ProductCoefficient gamma_u_old(gamma*(1.0-alpha), u_old_cf);
         ProductCoefficient alpha_f(alpha, f);

         b0.AddDomainIntegrator(new DomainLFGradIntegrator(alpha_eps_grad_u_old));
         b0.AddDomainIntegrator(new DomainLFGradIntegrator(beta_beta_grad_u_old));
         b0.AddDomainIntegrator(new DomainLFIntegrator(LHS2));
         b0.AddDomainIntegrator(new DomainLFIntegrator(psi_old_minus_psi));
         b0.AddDomainIntegrator(new DomainLFIntegrator(gamma_u_old));
         b0.AddDomainIntegrator(new DomainLFIntegrator(alpha_f));
         b0.Assemble();

         b1.AddDomainIntegrator(new DomainLFIntegrator(expit_psi));
         b1.Assemble();

         BilinearForm a00(&H1fes);
         a00.AddDomainIntegrator(new DiffusionIntegrator(eps_coeff));
         a00.AddDomainIntegrator(new DiffusionIntegrator(beta_beta));
         a00.AddDomainIntegrator(new MassIntegrator(gamma_coeff));
         a00.AddDomainIntegrator(new MassIntegrator(L2_weight_coeff));
         a00.Assemble();
         a00.EliminateEssentialBC(ess_bdr,x.GetBlock(0),rhs.GetBlock(0),mfem::Operator::DIAG_ONE);
         a00.Finalize();
         SparseMatrix &A00 = a00.SpMat();

         MixedBilinearForm a10(&H1fes,&L2fes);
         a10.AddDomainIntegrator(new MixedScalarMassIntegrator());
         a10.Assemble();
         a10.EliminateTrialDofs(ess_bdr, x.GetBlock(0), rhs.GetBlock(1));
         a10.Finalize();
         SparseMatrix &A10 = a10.SpMat();

         SparseMatrix &A01 = *Transpose(A10);

         BilinearForm a11(&L2fes);
         // a11.AddDomainIntegrator(new MassIntegrator(zero));
         a11.AddDomainIntegrator(new MassIntegrator(neg_dexpitdx_psi));
         a11.Assemble();
         a11.Finalize();
         SparseMatrix &A11 = a11.SpMat(); 

         BlockMatrix A(offsets);
         A.SetBlock(0,0,&A00);
         A.SetBlock(1,0,&A10);
         A.SetBlock(0,1,&A01);
         A.SetBlock(1,1,&A11);

         // iterative solver
         // BlockDiagonalPreconditioner prec(offsets);
         // prec.SetDiagonalBlock(0,new GSSmoother(A00));
         // prec.SetDiagonalBlock(1,new GSSmoother(A11));

         // GMRES(A,prec,rhs,x,0,200, 50, 1e-12,0.0);

         u_gf.MakeRef(&H1fes, x.GetBlock(0), 0);
         delta_psi_gf.MakeRef(&L2fes, x.GetBlock(1), 0);
         
         // or direct solver
         SparseMatrix * A_mono = A.CreateMonolithic();
         UMFPackSolver umf(*A_mono);
         umf.Mult(rhs,x);

         u_tmp -= u_gf;
         double Newton_update_size = u_tmp.ComputeL2Error(zero);
         u_tmp = u_gf;

         // ExpitGridFunctionCoefficient expit_psi(psi_gf);
         GridFunction u_alt_gf(&L2fes);
         u_alt_gf.ProjectCoefficient(expit_psi);
         sol_sock << "solution\n" << mesh << u_alt_gf << "window_title 'Discrete solution'" << flush;

         // sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'" << flush;

         // psi_sock << "solution\n" << mesh << delta_psi_gf << "window_title 'delta psi'" << flush;

         double gamma = 1.0;
         delta_psi_gf *= gamma;
         psi_gf += delta_psi_gf;
         // psi_gf = delta_psi_gf;

         // mfem::out << "Picard_update_size = " << Newton_update_size << endl;
         mfem::out << "Newton_update_size = " << Newton_update_size << endl;

         if (Newton_update_size < increment_u/10.0)
         {
            break;
         }

      }
      mfem::out << "Number of Newton iterations = " << j+1 << endl;
      
      u_tmp = u_gf;
      u_tmp -= u_old_gf;
      increment_u = u_tmp.ComputeL2Error(zero);

      mfem::out << "increment_u = " << increment_u << endl;

      delta_psi_gf = psi_gf;
      delta_psi_gf -= psi_old_gf;
      increment_psi = delta_psi_gf.ComputeL2Error(zero);
      delta_psi_gf = 0.0;

      u_old_gf = u_gf;
      psi_old_gf = psi_gf;

      // 14. Send the solution by socket to a GLVis server.
      // sol_sock << "solution\n" << mesh << psi << "window_title 'Discrete solution'" << flush;
      
      // ExpitGridFunctionCoefficient expit_psi(psi_gf);
      // GridFunction u_alt_gf(&H1fes);
      // u_alt_gf.ProjectCoefficient(expit_psi);
      // sol_sock << "solution\n" << mesh << u_alt_gf << "window_title 'Discrete solution'" << flush;
      // sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'" << flush;
   
      if (increment_u < 1e-6)
      {
         break;
      }
   }

   mfem::out << "\n Outer iterations: " << k+1
             << "\n Total iterations: " << total_iterations
             << endl;

   // 14. Exact solution.
   if (visualization)
   {
      socketstream err_sock(vishost, visport);
      err_sock.precision(8);
      FunctionCoefficient exact_coef(sol_func);

      GridFunction error_psi(&L2fes), error_u(&H1fes), error_ref(&H1fes);
      error_psi.ProjectCoefficient(exact_coef);
      error_u.ProjectCoefficient(exact_coef);
      error_ref = error_u;

      GridFunction expit_psi_gf(&L2fes);
      ExpitGridFunctionCoefficient expit_psi(psi_gf);
      expit_psi_gf.ProjectCoefficient(expit_psi);
      error_psi -= expit_psi_gf;
      error_u   -= u_gf;
      error_ref -= u_ref;

      mfem::out << "\n Final L2-error (our u_h) = " << error_u.ComputeL2Error(zero) << endl;
      mfem::out << "\n Final L2-error (our expit(psi_h)) = " << error_psi.ComputeL2Error(zero) << endl;
      mfem::out << "\n Comparison L2-error (SOTA) = " << error_ref.ComputeL2Error(zero) << endl;

      // sol_sock << "solution\n" << mesh << expit_psi_gf << "window_title 'Discrete solution'" << flush;
      sol_sock << "solution\n" << mesh << u_ref << "window_title 'Discrete solution'" << flush;
      err_sock << "solution\n" << mesh << error_ref << "window_title 'Error'"  << flush;
      // err_sock << "solution\n" << mesh << error_psi << "window_title 'Error'"  << flush;
   }

   return 0;
}

double LnitGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, lnit(val)));
}

double ExpitGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, expit(val)));
}

double dExpitdxGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, dexpitdx(val)));
}

double Ramp_BC(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double tol = 1e-10;
   double eps = 0.05;

   if (  (abs(y) < tol && x >= 0.2)
      || (abs(x-1.0) < tol)
      || (abs(y-1.0) < tol) )
   {
      return 0.0;
   }
   else if (  (abs(x) < tol && y <= 1.0 - eps)
           || (abs(y) < tol && x <= 0.2 - eps) )
   {
      return 1.0;
   }
   else if (x >= (0.2 - eps) && abs(y) < tol)
   {
      return (0.2 - x)/eps;
   }
   else if (y >= (1.0 - eps) && abs(x) < tol)
   {
      return  (1.0 - y)/eps;
   }
   else
   {
      return 0.5;
   }
}

double EJ_exact_solution(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double lambda = M_PI*M_PI*eps;
   double r1 = (1.0 + sqrt(1.0 + 4.0 * eps * lambda))/(2*eps);
   double r2 = (1.0 - sqrt(1.0 + 4.0 * eps * lambda))/(2*eps);

   double num = exp(r2 * (x - 1.0)) - exp(r1 * (x-1.0));
   double denom = exp(-r2) - exp(-r1);
   // double denom = r1 * exp(-r2) - r2 * exp(-r1);

   double scale = 0.5;
   // double scale = (r1 * exp(-r2) - r2 * exp(-r1)) / (exp(-r2) - exp(-r1));

   return scale * num / denom * cos(M_PI * y) + 0.5;
   
}