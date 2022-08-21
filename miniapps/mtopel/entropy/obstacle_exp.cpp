//                                MFEM Obstacle Problem
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double spherical_obstacle(const Vector &pt);
double exact_solution(const Vector &pt);
void spherical_obstacle_gradient(const Vector &pt, Vector &grad);

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

class ReciprocalGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u;
   FunctionCoefficient *obstacle;
   double min_val;
   double max_val;

public:
   ReciprocalGridFunctionCoefficient(GridFunction &u_, FunctionCoefficient &obst_, double min_val_=1e-12, double max_val_=1e12)
      : u(&u_), obstacle(&obst_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "./disk.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool algebraic_ceed = false;
   int max_it = 5;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iterations");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic", "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
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

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels = 4;
      // int ref_levels =
      //    (int)floor(log(50000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (mesh.GetNodes())
   {
      fec = mesh.GetNodes()->OwnFEC();
      delete_fec = false;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   FiniteElementSpace fespace(&mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   // ess_tdof_list.Print();

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   auto func = [](const Vector &x)
   {
      double val = 1.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(M_PI*x(i));
      }
      // return -1.0;
      return -x.Size()*pow(M_PI,2) * val;
      // return x.Size()*pow(M_PI,2) * val;
   };
   auto IC_func = [](const Vector &x)
   {
      double r0 = 1.0;
      double rr = 0.0;
      for (int i=0; i<x.Size(); i++)
      {
         rr += x(i)*x(i);
      }
      return r0*r0 - rr;
   };
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   Vector zero_vec(dim);
   zero_vec = 0.0;

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction u(&fespace);
   GridFunction psi(&fespace);
   GridFunction delta_psi(&fespace);
   GridFunction psi_old(&fespace);
   delta_psi = 0.0;

   /////////// Example 1   
   // u = 1.0;
   // ConstantCoefficient f(0.0);
   // FunctionCoefficient f(func);
   // FunctionCoefficient IC_coef(IC_func);
   // ConstantCoefficient bdry_coef(0.1);
   // ConstantCoefficient obstacle(0.0);
   // VectorConstantCoefficient obstacle_gradient(zero_vec);
   // double alpha0 = 0.1;

   /////////// Example 2
   // u = 0.5;
   FunctionCoefficient IC_coef(IC_func);
   ConstantCoefficient f(0.0);
   ConstantCoefficient bdry_coef(0.0);
   FunctionCoefficient obstacle(spherical_obstacle);
   VectorFunctionCoefficient obstacle_gradient(dim, spherical_obstacle_gradient);
   double alpha0 = 1.0;

   SumCoefficient bdry_funcoef(bdry_coef, IC_coef);
   u.ProjectCoefficient(bdry_funcoef);
   // u.ProjectBdrCoefficient(bdry_coef, ess_bdr);
   LogarithmGridFunctionCoefficient ln_u(u, obstacle);
   psi.ProjectCoefficient(ln_u);
   psi_old = psi;


   OperatorPtr A;
   Vector B, X;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   // 12. Iterate
   for (int k = 1; k <= max_it; k++)
   {
      // double alpha = alpha0 / sqrt(k);
      // double alpha = alpha0 * sqrt(k);
      double alpha = alpha0;
      // alpha *= 2;

      for (int j = 1; j <= 3; j++)
      {
         // A. Assembly
         
         // MD
         // double c1 = 1.0;
         // double c2 = 1.0 - alpha;

         // IMD
         double c1 = 1.0 + alpha;
         double c2 = 1.0;

         GridFunctionCoefficient psi_cf(&psi);
         GridFunctionCoefficient psi_old_cf(&psi_old);
         ExponentialGridFunctionCoefficient exp_psi(psi, zero);
         ExponentialGridFunctionCoefficient exp_psi_old(psi_old, zero);
         GradientGridFunctionCoefficient grad_psi(&psi);
         GradientGridFunctionCoefficient grad_psi_old(&psi_old);
         ProductCoefficient c1_exp_psi(c1, exp_psi);
         ProductCoefficient c2_exp_psi_old(c2, exp_psi_old);
         ScalarVectorProductCoefficient c1_exp_psi_grad_psi(c1_exp_psi, grad_psi);
         ScalarVectorProductCoefficient c2_exp_psi_old_grad_psi_old(c2_exp_psi_old, grad_psi_old);

         BilinearForm a(&fespace);
         a.AddDomainIntegrator(new DiffusionIntegrator(c1_exp_psi));
         a.AddDomainIntegrator(new TransposeIntegrator(new ConvectionIntegrator(c1_exp_psi_grad_psi)));
         a.AddDomainIntegrator(new MassIntegrator(one));
         a.Assemble();

         VectorSumCoefficient gradient_term_RHS(c1_exp_psi_grad_psi, c2_exp_psi_old_grad_psi_old, -1.0, 1.0);
         SumCoefficient mass_term_RHS(psi_cf, psi_old_cf, -1.0, 1.0);
         ScalarVectorProductCoefficient minus_alpha_obstacle_gradient(-alpha, obstacle_gradient);
         ProductCoefficient alpha_f(alpha, f);

         LinearForm b(&fespace);
         b.AddDomainIntegrator(new DomainLFGradIntegrator(gradient_term_RHS));
         b.AddDomainIntegrator(new DomainLFIntegrator(mass_term_RHS));
         b.AddDomainIntegrator(new DomainLFGradIntegrator(minus_alpha_obstacle_gradient));
         b.AddDomainIntegrator(new DomainLFIntegrator(alpha_f));
         b.Assemble();

         // B. Solve state equation
         a.FormLinearSystem(ess_tdof_list, delta_psi, b, A, X, B);
         GSSmoother S((SparseMatrix&)(*A));
         GMRES(*A, S, B, X, 0, 20000, 100, 1e-8, 1e-8);
         

         // C. Recover state variable
         a.RecoverFEMSolution(X, b, delta_psi);

         double gamma = 1.0;
         delta_psi *= gamma;
         psi += delta_psi;
      }
      psi_old = psi;

      // 14. Send the solution by socket to a GLVis server.
      ExponentialGridFunctionCoefficient exp_psi(psi, obstacle);
      u.ProjectCoefficient(exp_psi);
      // sol_sock << "solution\n" << mesh << psi << "window_title 'Discrete solution'" << flush;
      sol_sock << "solution\n" << mesh << u << "window_title 'Discrete solution'" << flush;
   }

   // 14. Exact solution.
   if (visualization)
   {
      socketstream err_sock(vishost, visport);
      err_sock.precision(8);
      FunctionCoefficient exact_coef(exact_solution);

      GridFunction error(&fespace);
      error = 0.0;
      error.ProjectCoefficient(exact_coef);
      error -= u;

      err_sock << "solution\n" << mesh << error << "window_title 'Error'"  << flush;
   }

   // 15. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
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

double ReciprocalGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip) - obstacle->Eval(T, ip);
   if (val < 0)
   {
      return max_val;
   }
   else
   {
      return min(max_val, max(min_val, 1.0/val) );
   }
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

double exact_solution(const Vector &pt)
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
