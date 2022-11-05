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

using namespace std;
using namespace mfem;

/** The Lagrangian for this problem is
 *    
 *    L(u,K,p,λ) = (f,u) - (K \nabla u, \nabla p) + (f,p) - λ (∫_Ω K dx - V ⋅ vol(\Omega))
 *                                                + β/2 (∫_Ω K dx - V ⋅ vol(\Omega))^2    
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


double load(const Vector & x)
{
   // double x1 = x(0);
   // double x2 = x(1);
   // double r = sqrt(x1*x1 + x2*x2);
   // if (r <= 0.5)
   // {
   //    return 1.0;
   // }
   // else
   // {
   //    return 0.0;
   // }
   return 1.;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int ref_levels = 2;
   int order = 2;
   bool visualization = true;
   double alpha = 1.0;
   double beta = 1.0;
   double mass_fraction = 0.5;
   double compliance_max = 0.15;
   int max_it = 1e3;
   double tol_K = 1e-3;
   double tol_lambda = 1e-3;
   double K_max = 0.9;
   double K_min = 1e-3;
   int prob = 0;
   int batch_size = 5;

   double initial_primal_tolerance = 1e-3;
   double initial_dual_tolerance = 1e-3;
   double primal_tolerance_decay_factor = 0.5;
   double dual_tolerance_decay_factor = 0.5;
   // double primal_stepsize_decay_factor = 0.9;
   // double dual_stepsize_growth_factor = 1.0/0.9;


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
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&tol_K, "-tk", "--tol_K",
                  "Exit tolerance for K");     
   args.AddOption(&batch_size, "-bs", "--batch-size",
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

   ostringstream file_name;
   file_name << "conv_order" << order << "_GD" << ".csv";
   ofstream conv(file_name.str().c_str());
   conv << "Step,    Compliance,    Mass Fraction" << endl;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 3. Define the load f.
   FunctionCoefficient f(load);

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 5. Define the vector finite element spaces representing the state variable u,
   //    adjoint variable p, and the control variable f.
   H1_FECollection state_fec(order, dim);
   L2_FECollection control_fec(order-1, dim, BasisType::Positive);
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

   // 7. Set the initial guess for f and the boundary conditions for u and p.
   GridFunction u(&state_fes);
   GridFunction p(&state_fes);
   GridFunction K(&control_fes);
   GridFunction K_old(&control_fes);
   u = 0.0;
   p = 0.0;
   K = 1.0;
   K_old = 0.0;



   // 8. Set up the linear form b(.) for the state and adjoint equations.
   LinearForm b(&state_fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(f));
   b.Assemble();

   OperatorPtr A;
   Vector B, C, X;

   // 9. Define the gradient function
   GridFunction grad(&control_fes);

   // 10. Define some tools for later
   ConstantCoefficient zero(0.0);
   ConstantCoefficient one(1.0);
   GridFunction onegf(&control_fes);
   onegf = 1.0;
   LinearForm vol_form(&control_fes);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble();
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

   // 12. Perform projected gradient descent
   double intermediate_tol_K = initial_primal_tolerance;
   double intermediate_tol_lambda = initial_dual_tolerance;

   double lambda = 0.0;
   for (int k = 1; k < max_it; k++)
   {
      double norm_K;
      for (int l = 1; l < max_it; l++)
      {
         BilinearForm a(&state_fes);
         GridFunctionCoefficient diffusion_coeff(&K);
         a.AddDomainIntegrator(new DiffusionIntegrator(diffusion_coeff));
         a.Assemble();
         a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

         // B. Solve state equation
         GSSmoother M((SparseMatrix&)(*A));
         PCG(*A, M, B, X, 0, 800, 1e-12, 0.0);

         // C. Recover state variable
         a.RecoverFEMSolution(X, b, u);

         // H. Constuct gradient function (i.e., -|\nabla u|^2 - λ + β(∫_Ω K dx - V ⋅ vol(\Omega)))
         GradientGridFunctionCoefficient grad_u(&u);
         InnerProductCoefficient norm2_grad_u(grad_u,grad_u);
         ProductCoefficient minus_norm2_grad_u(-1.0,norm2_grad_u);

         // ConstantCoefficient lambda_cf(lambda);
         double c = -lambda + beta * (vol_form(K)/domain_volume - mass_fraction)/domain_volume;
         ConstantCoefficient scalar_terms(c);
         LinearForm d(&control_fes);
         d.AddDomainIntegrator(new DomainLFIntegrator(minus_norm2_grad_u));
         d.AddDomainIntegrator(new DomainLFIntegrator(scalar_terms));
         d.Assemble();


         BilinearForm L2proj(&control_fes);
         InverseIntegrator * m = new InverseIntegrator(new MassIntegrator());
         L2proj.AddDomainIntegrator(m);
         L2proj.Assemble();
         Array<int> empty_list;
         OperatorPtr invM;
         L2proj.FormSystemMatrix(empty_list,invM);   
         invM->Mult(d,grad);
      
         // β(∫_Ω K dx - V ⋅ vol(\Omega)))

         K -= grad;

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
         norm_K = K.ComputeL2Error(tmp)/alpha;
         K_old = K;
         mfem::out << "norm of reduced gradient = " << norm_K << endl;
         mfem::out << "compliance = " << b(u) << endl;

         // R = (x_{k} - x_{k+1})/alpha
         // x_{k+1} = x_k - alpha R
         if (norm_K < intermediate_tol_K)
         {
            break;
         }
      }
      // λ <- λ - β (∫_Ω K dx - V⋅ vol(\Omega))
      double mass = vol_form(K);

      mfem::out << "mass_fraction = " << mass / domain_volume << endl;

      double lambda_inc = mass/domain_volume - mass_fraction;

      lambda -= beta*lambda_inc;

      double norm_gradient = grad.ComputeL2Error(zero);
      mfem::out << "\ngrad norm = " << norm_gradient << endl;

      mfem::out << "lambda_inc = " << lambda_inc << endl;
      mfem::out << "lambda = " << lambda << endl;

      if (visualization)
      {

         sout_u << "solution\n" << mesh << u
               << "window_title 'State u'" << flush;

         sout_K << "solution\n" << mesh << K
                << "window_title 'Control K'" << flush;
      }

      if (abs(lambda_inc) < intermediate_tol_K)
      {
         intermediate_tol_K *= primal_tolerance_decay_factor;
         intermediate_tol_lambda *= dual_tolerance_decay_factor;
      }
      // else
      // {
      //    alpha *= primal_stepsize_decay_factor;
      //    beta *= dual_stepsize_growth_factor;
      // }

      if ((abs(lambda_inc) < tol_lambda) && (norm_K < tol_K))
      {
         break;
      }

   }

   return 0;
}