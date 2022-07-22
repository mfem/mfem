//                                Solution of distributed control problem
//
// Compile with: make optimal_design
//
// Sample runs:
//    optimal_design -r 3
//    optimal_design -m ../../data/star.mesh -r 3
//    optimal_design -sl 1 -m ../../data/mobius-strip.mesh -r 4
//    optimal_design -m ../../data/star.mesh -sl 5 -r 3 -mf 0.5 -o 5 -max 0.75
//
// Description:  This examples solves the following PDE-constrained
//               optimization problem:
//
//         min J(K) = (f,u)
//
//         subject to   - div( K\nabla u ) = f    in \Omega
//                                       u = 0    on \partial\Omega
//         and            \int_\Omega K dx <= V ⋅ vol(\Omega)
//         and            a <= K(x) <= b
//
//   Joachim Peterson 1999 for proof

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>
#include <random>
#include "common/fpde.hpp"

using namespace std;
using namespace mfem;

/** The Lagrangian for this problem is
 *    
 *    L(u,K,p,λ) = (f,u) - (K \nabla u, \nabla p) + (f,p)
 *                + \gamma\epsilon/2 (\nabla K, \nabla K)
 *                + \gamma/(2\epsilon) ∫_Ω K(1-K) dx
 *                - λ (∫_Ω K dx - V ⋅ vol(\Omega))
 *                + β/2 (∫_Ω K dx - V ⋅ vol(\Omega))^2    
 *      u, p \in H^1_0(\Omega)
 *      K \in L^\infty(\Omega)
 * 
 *  Note that
 * 
 *    \partial_p L = 0        (1)
 *  
 *  delivers the state equation
 *    
 *    (\nabla u, \nabla v) = (f,v)  for all v in H^1_0(\Omega)
 * 
 *  and
 *  
 *    \partial_u L = 0        (2)
 * 
 *  delivers the adjoint equation (same as the state eqn)
 * 
 *    (\nabla p, \nabla v) = (f,v)  for all v in H^1_0(\Omega)
 *    
 *  and at the solutions u=p of (1) and (2), respectively,
 * 
 *  D_K J = D_K L = \partial_u L \partial_K u + \partial_p L \partial_K p
 *                + \partial_K L
 *                = \partial_K L
 *                = (-|\nabla u|^2 - λ + β(∫_Ω K dx - V ⋅ vol(\Omega)), \cdot)
 * 
 * We update the control K_k with projected gradient descent via
 * 
 *  1. Initialize λ 
 *  2. update until convergence 
 *     K <- P (K + α |\nabla u|^2 + λ)
 *  3. update λ 
 *     λ <- λ - β (∫_Ω K dx - V)
 * 
 * P is the projection operator enforcing a <= K(x) <= b, and α  is a specified
 * step length.
 * 
 */

class RandomFunctionCoefficient : public Coefficient
{
private:
   double a = 0.45;
   double b = 0.55;
   double x1,y1;
   std::default_random_engine generator;
   std::uniform_real_distribution<double> * distribution;
   double (*Function)(const Vector &, double, double);
public:
   RandomFunctionCoefficient(double (*F)(const Vector &, double, double)) 
   : Function(F) 
   {
      distribution = new std::uniform_real_distribution<double> (a,b);
      x1 = (*distribution)(generator);
      y1 = (*distribution)(generator);
   }
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      Vector transip(3);
      T.Transform(ip, transip);
      return ((*Function)(transip, x1, y1));
   }
   void resample()
   {
      x1 = (*distribution)(generator);
      y1 = (*distribution)(generator);
   }
};

double randomload(const Vector & X, double x1, double y1)
{
   double x = X(0) - x1;
   double y = X(1) - y1;
   double r = sqrt(x*x + y*y);
   if (r <= 0.3)
   {
      return 1.0;
   }
   else
   {
      return 0.0;
   }
   // return 1.0;
}


double load(const Vector & x)
{
   double x1 = x(0);
   double x2 = x(1);
   double r = sqrt(x1*x1 + x2*x2);
   if (r <= 0.5)
   {
      return 1.0;
   }
   else
   {
      return 0.0;
   }
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../../data/inline-quad.mesh";
   int ref_levels = 2;
   int order = 2;
   bool visualization = true;
   double alpha = 1.0;
   double beta = 1.0;
   double gamma = 1.0;
   double epsilon = 1.0;
   double theta = 0.5;
   double mass_fraction = 0.5;
   double compliance_max = 0.15;
   int max_it = 1e2;
   double tol_K = 1e-3;
   double tol_lambda = 1e-3;
   double K_max = 0.9;
   double K_min = 1e-3;
   int prob = 0;
   int batch_size_min = 5;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&alpha, "-alpha", "--alpha-step-length",
                  "Step length for gradient descent.");
   args.AddOption(&beta, "-beta", "--beta-step-length",
                  "Step length for λ"); 
   args.AddOption(&gamma, "-gamma", "--gamma-penalty",
                  "gamma penalty weight");
   args.AddOption(&epsilon, "-epsilon", "--epsilon-thickness",
                  "epsilon phase field thickness");
   args.AddOption(&theta, "-theta", "--theta-sampling-ratio",
                  "Sampling ratio theta");                  
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&tol_K, "-tk", "--tol_K",
                  "Exit tolerance for K");     
   args.AddOption(&batch_size_min, "-bs", "--batch-size",
                  "batch size for stochastic gradient descent.");                             
   args.AddOption(&tol_lambda, "-tl", "--tol_lambda",
                  "Exit tolerance for λ");                                 
   args.AddOption(&mass_fraction, "-mf", "--mass-fraction",
                  "Mass fraction for diffusion coefficient.");
   args.AddOption(&compliance_max, "-cmax", "--compliance-max",
                  "Maximum of compliance.");
   args.AddOption(&K_max, "-Kmax", "--K-max",
                  "Maximum of diffusion diffusion coefficient.");
   args.AddOption(&K_min, "-Kmin", "--K-min",
                  "Minimum of diffusion diffusion coefficient.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&prob, "-p", "--problem",
                  "Optimization problem: 0 - Compliance Minimization, 1 - Mass Minimization.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   int batch_size = batch_size_min;

   ostringstream file_name;
   file_name << "conv_order" << order << "_GD" << ".csv";
   ofstream conv(file_name.str().c_str());
   conv << "Step,    Compliance,    Mass Fraction" << endl;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 5. Define the vector finite element spaces representing the state variable u,
   //    adjoint variable p, and the control variable f.
   H1_FECollection state_fec(order, dim);
   H1_FECollection control_fec(order-1, dim, BasisType::Positive);
   FiniteElementSpace state_fes(&mesh, &state_fec);
   FiniteElementSpace control_fes(&mesh, &control_fec);

   int state_size = state_fes.GetTrueVSize();
   int control_size = control_fes.GetTrueVSize();
   cout << "Number of state unknowns: " << state_size << endl;
   cout << "Number of control unknowns: " << control_size << endl;

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_tdof_list;
   state_fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Set the initial guess for f and the boundary conditions for u.
   GridFunction u(&state_fes);
   GridFunction K(&control_fes);
   GridFunction K_old(&control_fes);
   u = 0.0;
   K = 1.0;
   K_old = 0.0;


   RandomFunctionCoefficient load_coeff(randomload);

   // 8. Set up the linear form b(.) for the state and adjoint equations.

   ConstantCoefficient eps2_cf(epsilon*epsilon);
   FPDESolver * PoissonSolver = new FPDESolver();
   PoissonSolver->SetMesh(&mesh);
   PoissonSolver->SetOrder(order-1);
   PoissonSolver->SetAlpha(1.0);
   PoissonSolver->SetBeta(1.0);
   PoissonSolver->SetDiffusionCoefficient(&eps2_cf);
   Array<int> ess_bdr_K(mesh.bdr_attributes.Max()); ess_bdr_K = 0;
   PoissonSolver->SetEssentialBoundary(ess_bdr_K);
   PoissonSolver->Init();
   PoissonSolver->SetupFEM();

   LinearForm b(&state_fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(load_coeff));
   // b.AddDomainIntegrator(new DomainLFIntegrator(f));
   OperatorPtr A;
   Vector B, C, X;

   // 9. Define the gradient function
   GridFunction grad(&control_fes);
   GridFunction tmp_grad(&control_fes);
   GridFunction avg_grad(&control_fes);

   // 10. Define some tools for later
   ConstantCoefficient zero(0.0);
   ConstantCoefficient one(1.0);
   GridFunction onegf(&control_fes);
   onegf = 1.0;
   LinearForm vol_form(&control_fes);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble(false);
   double domain_volume = vol_form(onegf);

   // 11. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_u,sout_p,sout_K;
   if (visualization)
   {
      sout_u.open(vishost, visport);
      sout_K.open(vishost, visport);
      sout_u.precision(8);
      sout_K.precision(8);
   }

   // Project initial K onto constraint set.
   for (int i = 0; i < K.Size(); i++)
   {
      if (K[i] > K_max) 
      {
          K[i] = K_max;
      }
      else if (K[i] < K_min)
      {
          K[i] = K_min;
      }
      else
      { // do nothing
      }
   }


   // 12. AL iterations
   double lambda = 0.0;
   for (int k = 1; k < max_it; k++)
   {
      // A. Form state equation

      for (int l = 1; l < max_it; l++)
      {
         cout << "Step = " << l << endl;
         cout << "batch_size = " << batch_size << endl;

         avg_grad = 0.0;
         double avg_grad_norm = 0.;
         for (int ib = 0; ib<batch_size; ib++)
         {
            BilinearForm a(&state_fes);
            GridFunctionCoefficient diffusion_coeff(&K);
            a.AddDomainIntegrator(new DiffusionIntegrator(diffusion_coeff));
            a.Assemble();

            load_coeff.resample();
            b.Assemble(false);

            a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

            // B. Solve state equation
            GSSmoother M((SparseMatrix&)(*A));
            PCG(*A, M, B, X, 0, 800, 1e-12, 0.0);

            // C. Recover state variable
            a.RecoverFEMSolution(X, b, u);

            // H. Constuct gradient function
            // i.e., \nabla J = \gamma/\epsilon (1/2 - K) - λ + β(∫_Ω K dx - V ⋅ vol(\Omega)) - R^{-1}(|\nabla u|^2)
            GradientGridFunctionCoefficient grad_u(&u);
            InnerProductCoefficient norm2_grad_u(grad_u,grad_u);

            // LinearForm d(&control_fes);
            // d.AddDomainIntegrator(new DomainLFIntegrator(norm2_grad_u));
         
            // ConstantCoefficient lambda_cf(lambda);
            // d.AddDomainIntegrator(new DomainLFIntegrator(lambda_cf));
            // d.Assemble(false);/
            // BilinearForm L2proj(&control_fes);
            // InverseIntegrator * m = new InverseIntegrator(new MassIntegrator());
            // L2proj.AddDomainIntegrator(m);
            // L2proj.Assemble();
            // Array<int> empty_list;
            // OperatorPtr invM;
            // L2proj.FormSystemMatrix(empty_list,invM);   
            // invM->Mult(d,grad);
            PoissonSolver->SetRHSCoefficient(&norm2_grad_u);
            PoissonSolver->Solve();
            
            grad = K;
            grad -= (K_max-K_min)/2.0;
            grad *= -gamma/epsilon;
            grad -= *PoissonSolver->GetFEMSolution();

            // - λ + β(∫_Ω K dx - V ⋅ vol(\Omega)))
            grad -= lambda;
            grad += beta * (vol_form(K)/domain_volume - mass_fraction)/domain_volume;

            avg_grad += grad;
            double grad_norm = grad.ComputeL2Error(zero);
            avg_grad_norm += grad_norm*grad_norm;
         } // enf of loop through batch samples
         avg_grad_norm /= (double)batch_size;  
         avg_grad /= (double)batch_size;

         double norm_avg_grad = pow(avg_grad.ComputeL2Error(zero),2);
         double variance = (avg_grad_norm - norm_avg_grad)/(batch_size - 1);  

         avg_grad *= alpha;
         K -= avg_grad;

         // K. Project onto constraint set.
         for (int i = 0; i < K.Size(); i++)
         {
            if (K[i] > K_max) 
            {
               K[i] = K_max;
            }
            else if (K[i] < K_min)
            {
               K[i] = K_min;
            }
            else
            { // do nothing
            }
         }

         GridFunctionCoefficient tmp(&K_old);
         double norm_K = K.ComputeL2Error(tmp)/alpha;
         K_old = K;
         mfem::out << "norm of reduced gradient = " << norm_K << endl;
         mfem::out << "compliance = " << b(u) << endl;
         mfem::out << "variance = " << variance << std::endl;
         if (norm_K < tol_K)
         {
            break;
         }

         double ratio = sqrt(abs(variance)) / norm_K ;
         mfem::out << "ratio = " << ratio << std::endl;
         MFEM_VERIFY(IsFinite(ratio), "ratio not finite");
         if (ratio > theta)
         {
            batch_size = (int)(pow(ratio / theta,2.) * batch_size); 
         }
         else if (ratio < 0.5*theta)
         {
            batch_size = max(batch_size/2,batch_size_min);
         }

      }
      // λ <- λ - β (∫_Ω K dx - V⋅ vol(\Omega))
      double mass = vol_form(K);

      mfem::out << "mass_fraction = " << mass / domain_volume << endl;

      double lambda_inc = mass/domain_volume - mass_fraction;

      lambda -= beta*lambda_inc;

      mfem::out << "lambda_inc = " << lambda_inc << endl;
      mfem::out << "lambda = " << lambda << endl;



      if (visualization)
      {

         sout_u << "solution\n" << mesh << u
               << "window_title 'State u'" << flush;

         sout_K << "solution\n" << mesh << K
                << "window_title 'Control K'" << flush;
      }

      if (abs(lambda_inc) < tol_lambda)
      {
         break;
      }

   }

   return 0;
}