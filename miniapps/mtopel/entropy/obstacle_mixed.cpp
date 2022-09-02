//                                MFEM Obstacle Problem
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double spherical_obstacle(const Vector &pt);
void spherical_obstacle_gradient(const Vector &pt, Vector &grad);
double exact_solution_obstacle(const Vector &pt);
double exact_solution_biactivity(const Vector &pt);
double load_biactivity(const Vector &pt);

class LogarithmGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   Coefficient *obstacle;
   double min_val;

public:
   LogarithmGridFunctionCoefficient(GridFunction &u_, Coefficient &obst_, double min_val_=-1e2)
      : u(&u_), obstacle(&obst_), min_val(min_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ExponentialGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   Coefficient *obstacle;
   double min_val;
   double max_val;

public:
   ExponentialGridFunctionCoefficient(GridFunction &u_, Coefficient &obst_, double min_val_=0.0, double max_val_=1e6)
      : u(&u_), obstacle(&obst_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "./disk.mesh";
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
   FiniteElementSpace H1fes(&mesh, &H1fec);

   L2_FECollection L2fec(order-1, dim);
   FiniteElementSpace L2fes(&mesh, &L2fec);

   cout << "Number of finite element unknowns: "
        << H1fes.GetTrueVSize() 
        << L2fes.GetTrueVSize() << endl;

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = H1fes.GetVSize();
   offsets[2] = L2fes.GetVSize();
   offsets.PartialSum();


   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr = 1;
      H1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   auto sol_func = [](const Vector &x)
   {
      double val = 1.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(M_PI*x(i));
      }
      return val;
   };

   auto rhs_func = [](const Vector &x)
   {
      double val = 1.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(M_PI*x(i));
      }
      return (x.Size()*pow(M_PI,2)+1.0) * val;
   };


   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   GridFunction u_gf, lambda_gf;
   FunctionCoefficient sol0_cf(sol_func);
   u_gf.MakeRef(&H1fes,x.GetBlock(0));
   u_gf.ProjectCoefficient(sol0_cf);
   lambda_gf.MakeRef(&L2fes,x.GetBlock(1));

   LinearForm b0,b1;
   FunctionCoefficient rhs0_cf(rhs_func);
   b0.Update(&H1fes,rhs.GetBlock(0),0);
   b1.Update(&L2fes,rhs.GetBlock(1),0);

   b0.AddDomainIntegrator(new DomainLFIntegrator(rhs0_cf));
   b0.Assemble();
   b1.Assemble();

   //-----------------------------

   BilinearForm a00(&H1fes);
   a00.AddDomainIntegrator(new DiffusionIntegrator());
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

   // MixedBilinearForm a01(&L2fes,&H1fes);
   // a01.AddDomainIntegrator(new MixedScalarMassIntegrator());
   // a01.Assemble();
   // a01.EliminateTestDofs(ess_bdr);
   // a01.Finalize();
   // SparseMatrix &A01 = a01.SpMat();
   SparseMatrix &A01 = *Transpose(A10);

   // initial guess for lambda;
   lambda_gf = 0.0;
   ConstantCoefficient zero(0.0);
   ExponentialGridFunctionCoefficient exp_l(lambda_gf,zero);
   BilinearForm a11(&L2fes);
   ProductCoefficient neg_exp_l(-1.0,exp_l);
   a11.AddDomainIntegrator(new MassIntegrator(neg_exp_l));
   // a11.AddDomainIntegrator(new MassIntegrator(exp_l));
   a11.Assemble();
   a11.Finalize();
   SparseMatrix &A11 = a11.SpMat(); 


   BlockMatrix A(offsets);
   A.SetBlock(0,0,&A00);
   A.SetBlock(1,0,&A10);
   A.SetBlock(0,1,&A01);
   A.SetBlock(1,1,&A11);

   // iterative solver
   BlockDiagonalPreconditioner prec(offsets);
   prec.SetDiagonalBlock(0,new GSSmoother(A00));
   prec.SetDiagonalBlock(1,new GSSmoother(A11));
   // PCG(A,prec,rhs,x,1,200,1e-12,0.0);
   GMRES(A,prec,rhs,x,1,200, 50, 1e-12,0.0);
   
   // // or direct solver
   // SparseMatrix * A_mono = A.CreateMonolithic();
   // UMFPackSolver umf(*A_mono);
   // umf.Mult(rhs,x);


   u_gf.MakeRef(&H1fes, x.GetBlock(0), 0);
   lambda_gf.MakeRef(&L2fes, x.GetBlock(1), 0);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      GridFunction uex_gf(&H1fes);
      uex_gf.ProjectCoefficient(sol0_cf);
      socketstream uex_sock(vishost, visport);
      uex_sock.precision(8);
      uex_sock << "solution\n" << mesh << uex_gf << "window_title 'Exact Solution u'"  << flush;

      socketstream u_sock(vishost, visport);
      u_sock.precision(8);
      u_sock << "solution\n" << mesh << u_gf << "window_title 'Solution u'"  << flush;

      socketstream lambda_sock(vishost, visport);
      lambda_sock.precision(8);
      lambda_sock << "solution\n" << mesh << lambda_gf << "window_title 'Solution λ'"  << flush;
   }


   return 0;

   // // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   // //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   // //    the basis functions in the finite element fespace.

   // auto IC_func = [](const Vector &x)
   // {
   //    double r0 = 1.0;
   //    double rr = 0.0;
   //    for (int i=0; i<x.Size(); i++)
   //    {
   //       rr += x(i)*x(i);
   //    }
   //    return r0*r0 - rr;
   // };
   // ConstantCoefficient one(1.0);
   // ConstantCoefficient zero(0.0);
   // Vector zero_vec(dim);
   // zero_vec = 0.0;

   // // 8. Define the solution vector x as a finite element grid function
   // //    corresponding to fespace. Initialize x with initial guess of zero,
   // //    which satisfies the boundary conditions.
   // GridFunction u(&fespace);
   // GridFunction u_old(&fespace);
   // GridFunction psi(&fespace);
   // GridFunction delta_psi(&fespace);
   // GridFunction psi_old(&fespace);
   // delta_psi = 0.0;
   // u_old = 0.0;

   // /////////// Example 1   
   // // u = 1.0;
   // // ConstantCoefficient f(0.0);
   // // FunctionCoefficient f(func);
   // // FunctionCoefficient IC_coef(IC_func);
   // // ConstantCoefficient bdry_coef(0.1);
   // // ConstantCoefficient obstacle(0.0);
   // // VectorConstantCoefficient obstacle_gradient(zero_vec);
   // // SumCoefficient bdry_funcoef(bdry_coef, IC_coef);
   // // u.ProjectCoefficient(bdry_funcoef);
   // // double alpha0 = 0.1;

   // /////////// Example 2
   // u = 0.5;
   // FunctionCoefficient IC_coef(IC_func);
   // ConstantCoefficient f(0.0);
   // FunctionCoefficient obstacle(spherical_obstacle);
   // VectorFunctionCoefficient obstacle_gradient(dim, spherical_obstacle_gradient);
   // u.ProjectCoefficient(IC_coef);
   // double alpha0 = 1.0;

   // /////////// Example 2
   // // u = 0.5;
   // // FunctionCoefficient f(load_biactivity);
   // // FunctionCoefficient bdry_coef(exact_solution_biactivity);
   // // ConstantCoefficient obstacle(0.0);
   // // VectorConstantCoefficient obstacle_gradient(zero_vec);
   // // u.ProjectBdrCoefficient(bdry_coef, ess_bdr);
   // // double alpha0 = 1.0;

   // LogarithmGridFunctionCoefficient ln_u(u, obstacle);
   // psi.ProjectCoefficient(ln_u);
   // psi_old = psi;


   // OperatorPtr A;
   // Vector B, X;

   // char vishost[] = "localhost";
   // int  visport   = 19916;
   // socketstream sol_sock(vishost, visport);
   // sol_sock.precision(8);

   // // 12. Iterate
   // int k;
   // double increment = 1e-4;
   // for (k = 0; k < max_it; k++)
   // {
   //    double alpha = alpha0 / sqrt(k+1);
   //    // double alpha = alpha0 * sqrt(k+1);
   //    // double alpha = alpha0;
   //    // alpha *= 2;

   //    int j;
   //    for ( j = 0; j < 15; j++)
   //    {
   //       // A. Assembly
         
   //       // MD
   //       // double c1 = 1.0;
   //       // double c2 = 1.0 - alpha;

   //       // IMD
   //       double c1 = 1.0 + alpha;
   //       double c2 = 1.0;

   //       GridFunctionCoefficient psi_cf(&psi);
   //       GridFunctionCoefficient psi_old_cf(&psi_old);
   //       ExponentialGridFunctionCoefficient exp_psi(psi, zero);
   //       ExponentialGridFunctionCoefficient exp_psi_old(psi_old, zero);
   //       GradientGridFunctionCoefficient grad_psi(&psi);
   //       GradientGridFunctionCoefficient grad_psi_old(&psi_old);
   //       ProductCoefficient c1_exp_psi(c1, exp_psi);
   //       ProductCoefficient c2_exp_psi_old(c2, exp_psi_old);
   //       ScalarVectorProductCoefficient c1_exp_psi_grad_psi(c1_exp_psi, grad_psi);
   //       ScalarVectorProductCoefficient c2_exp_psi_old_grad_psi_old(c2_exp_psi_old, grad_psi_old);

   //       BilinearForm a(&fespace);
   //       a.AddDomainIntegrator(new DiffusionIntegrator(c1_exp_psi));
   //       a.AddDomainIntegrator(new TransposeIntegrator(new ConvectionIntegrator(c1_exp_psi_grad_psi)));
   //       a.AddDomainIntegrator(new MassIntegrator(one));
   //       a.Assemble();

   //       VectorSumCoefficient gradient_term_RHS(c1_exp_psi_grad_psi, c2_exp_psi_old_grad_psi_old, -1.0, 1.0);
   //       SumCoefficient mass_term_RHS(psi_cf, psi_old_cf, -1.0, 1.0);
   //       ScalarVectorProductCoefficient minus_alpha_obstacle_gradient(-alpha, obstacle_gradient);
   //       ProductCoefficient alpha_f(alpha, f);

   //       LinearForm b(&fespace);
   //       b.AddDomainIntegrator(new DomainLFGradIntegrator(gradient_term_RHS));
   //       b.AddDomainIntegrator(new DomainLFIntegrator(mass_term_RHS));
   //       b.AddDomainIntegrator(new DomainLFGradIntegrator(minus_alpha_obstacle_gradient));
   //       b.AddDomainIntegrator(new DomainLFIntegrator(alpha_f));
   //       b.Assemble();

   //       // B. Solve state equation
   //       a.FormLinearSystem(ess_tdof_list, delta_psi, b, A, X, B);
   //       // GSSmoother S((SparseMatrix&)(*A));
   //       // GMRES(*A, S, B, X, 0, 20000, 100, 1e-8, 1e-8);
   //       umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   //       umf_solver.SetOperator(*A);
   //       umf_solver.Mult(B, X);
         

   //       // C. Recover state variable
   //       a.RecoverFEMSolution(X, b, delta_psi);

   //       double Newton_update_size = delta_psi.ComputeL2Error(zero);

   //       double gamma = 1.0;
   //       delta_psi *= gamma;
   //       psi += delta_psi;
         
   //       // double update_tol = 1e-10;
   //       if (Newton_update_size < increment/10.0)
   //       {
   //          break;
   //       }
   //    }
   //    mfem::out << "Number of Newton iterations = " << j << endl;
      
   //    psi_old = psi;

   //    // 14. Send the solution by socket to a GLVis server.
   //    u_old = u;
   //    ExponentialGridFunctionCoefficient exp_psi(psi, obstacle);
   //    u.ProjectCoefficient(exp_psi);
   //    // sol_sock << "solution\n" << mesh << psi << "window_title 'Discrete solution'" << flush;
   //    sol_sock << "solution\n" << mesh << u << "window_title 'Discrete solution'" << flush;
      
   //    GridFunction delta_u(&fespace);
   //    delta_u = u;
   //    delta_u -= u_old;
   //    increment = delta_u.ComputeL2Error(zero);
   //    if (increment < 1e-5)
   //    {
   //       break;
   //    }
   // }

   // mfem::out << "\n Outer iterations: " << k << "\n || u - u_prvs || = " << increment << endl;

   // // 14. Exact solution.
   // if (visualization)
   // {
   //    socketstream err_sock(vishost, visport);
   //    err_sock.precision(8);
   //    // FunctionCoefficient exact_coef(exact_solution_biactivity);
   //    FunctionCoefficient exact_coef(exact_solution_obstacle);

   //    GridFunction error(&fespace);
   //    error = 0.0;
   //    error.ProjectCoefficient(exact_coef);
   //    error -= u;

   //    mfem::out << "\n error = " << error.ComputeL2Error(zero) << endl;

   //    err_sock << "solution\n" << mesh << error << "window_title 'Error'"  << flush;
   // }

   // // 15. Free the used memory.
   // if (delete_fec)
   // {
   //    delete fec;
   // }

   // return 0;
}

double LogarithmGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip) - obstacle->Eval(T, ip);
   return max(min_val, log(val));
}

double ExponentialGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, exp(val) + obstacle->Eval(T, ip)));
}

double spherical_obstacle(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double r = sqrt(x*x + y*y);
   double r0 = 0.5;
   double beta = 0.9;

   double b = r0*beta;
   double tmp = sqrt(r0*r0 - b*b);
   double B = tmp + b*b/tmp;
   double C = -b/tmp;

   if (r > b)
   {
      return B + r * C;
   }
   else
   {
      return sqrt(r0*r0 - r*r);
   }
}

void spherical_obstacle_gradient(const Vector &pt, Vector &grad)
{
   double x = pt(0), y = pt(1);
   double r = sqrt(x*x + y*y);
   double r0 = 0.5;
   double beta = 0.9;

   double b = r0*beta;
   double tmp = sqrt(r0*r0-b*b);
   double C = -b/tmp;

   if (r > b)
   {
      grad(0) = C * x / r;
      grad(1) = C * y / r;
   }
   else
   {
      grad(0) = - x / sqrt( r0*r0 - r*r );
      grad(1) = - y / sqrt( r0*r0 - r*r );
   }
}

double exact_solution_obstacle(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double r = sqrt(x*x + y*y);
   double r0 = 0.5;
   double a =  0.348982574111686;
   double A = -0.340129705945858;

   if (r > a)
   {
      return A * log(r);
   }
   else
   {
      return sqrt(r0*r0-r*r);
   }
}

double exact_solution_biactivity(const Vector &pt)
{
   double x = pt(0);

   if (x > 0.0)
   {
      return x*x;
   }
   else
   {
      return 0.0;
   }
}

double load_biactivity(const Vector &pt)
{
   double x = pt(0);

   if (x > 0.0)
   {
      return -2.0;
   }
   else
   {
      return 0.0;
   }
}

// double IC_biactivity(const Vector &pt)
// {
//    double x = pt(0);
//    return x*x;
// }