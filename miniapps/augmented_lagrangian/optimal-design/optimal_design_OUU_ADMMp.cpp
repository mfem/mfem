//                                Solution of distributed control problem
//
// Compile with: make optimal_design
//
// Sample runs:
//    ./optimal_design_OUU_ADMM -alpha 0.1 -beta 0.1 -admm -r 3 -o 2 -gamma 0.01 -sigma 1
//    ./optimal_design_OUU_ADMM -alpha 0.1 -beta 0.1 -admm -r 3 -o 2 -gamma 0.01 -sigma 0.5
//    ./optimal_design_OUU_ADMM -alpha 0.1 -beta 0.1 -admm -r 3 -o 2 -gamma 0.01 -sigma 0.2
//    ./optimal_design_OUU_ADMM -alpha 0.1 -beta 0.1 -r 3 -o 2
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

bool random_seed = true;


// class RandomFunctionCoefficient : public Coefficient
// {
// private:
//    double a = 0.0;
//    double b = 1.0;
//    double x1, y1;
//    double (*Function)(const Vector &, double, double);
// public:
//    RandomFunctionCoefficient(double (*F)(const Vector &, double, double), int seed = 0) 
//    : Function(F)
//    {
//       resample(seed);
//    }
//    virtual double Eval(ElementTransformation &T,
//                        const IntegrationPoint &ip)
//    {
//       Vector transip(3);
//       T.Transform(ip, transip);
//       return ((*Function)(transip, x1, y1));
//    }
//    void resample(int seed = 0)
//    {
//       srand((unsigned)seed);
//       const double max = (double)(RAND_MAX) + 1.;
//       rand();
//       x1 = std::abs(rand()/max) * (b-a) + a;
//       y1 = std::abs(rand()/max) * (b-a) + a;
//    }
// };
class RandomFunctionCoefficient : public Coefficient
{
private:
   double a = 0.0;
   double b = 1.0;
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
   if (r <= 0.5)
   {
      return 1.0;
   }
   else
   {
      return 0.5;
   }
}

double load(const Vector & x)
{
   double x1 = x(0);
   double x2 = x(1);
   double r = sqrt(x1*x1 + x2*x2);
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
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   srand(time(0));

   // 1. Parse command-line options.
   const char *mesh_file = "../../../data/inline-quad.mesh";
   int ref_levels = 2;
   int order = 2;
   bool visualization = true;
   double alpha = 1.0;
   double beta = 1.0;
   double sigma = 1.0;
   double mass_fraction = 0.5;
   double compliance_max = 0.15;
   int max_it = 1e2;
   double tol_K = 1e-3;
   double tol_lambda = 1e-3;
   double K_max = 0.9;
   double K_min = 1e-3;
   int prob = 0;
   int batch_size = 5;
   double gamma = 1.0;
   double theta = 0.5;

   double primal_tolerance_decay_factor = 1.0;
   double dual_tolerance_decay_factor = 1.0;
   double primal_stepsize_decay_factor = 1.0;
   double dual_stepsize_growth_factor = 1.0;
   // double primal_tolerance_decay_factor = 0.5;
   // double dual_tolerance_decay_factor = 0.5;
   // double primal_stepsize_decay_factor = 0.9;
   // double dual_stepsize_growth_factor = 1.0/0.9;

   bool ADMM = false;


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
   args.AddOption(&sigma, "-sigma", "--sigma",
                  "Fractional exponent σ");                  
   args.AddOption(&gamma, "-gamma", "--gamma-penalization",
                  "Step length for γ");                  
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&tol_K, "-tk", "--tol_K",
                  "Exit tolerance for K");     
   args.AddOption(&batch_size, "-bs", "--batch-size",
                  "batch size for stochastic gradient descent.");                             
   args.AddOption(&tol_lambda, "-tl", "--tol_lambda",
                  "Exit tolerance for λ");    
   args.AddOption(&theta, "-theta", "--theta",
                  "Adaptive sampling factor");                                                     
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
   args.AddOption(&ADMM, "-admm", "--admm", "-no-admm",
                  "--no-admm",
                  "Enable or disable ADMM method.");
   args.AddOption(&random_seed, "-rs", "--random-seed", "-no-rs",
                  "--no-random-seed",
                  "Enable or disable GLVis visualization.");                 

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   double initial_primal_tolerance = tol_K;
   double initial_dual_tolerance = tol_lambda;


   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 5. Define the vector finite element spaces representing the state variable u,
   //    adjoint variable p, and the control variable f.
   H1_FECollection state_fec(order, dim);
   L2_FECollection control_fec(order-1, dim, BasisType::Positive);
   ParFiniteElementSpace state_fes(&pmesh, &state_fec);
   ParFiniteElementSpace control_fes(&pmesh, &control_fec);

   int state_size = state_fes.GetTrueVSize();
   int control_size = control_fes.GetTrueVSize();
   if (myid == 0)
   {
      cout << "Number of state unknowns: " << state_size << endl;
      cout << "Number of control unknowns: " << control_size << endl;
   }

   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_tdof_list;
   state_fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Set the initial guess for f and the boundary conditions for u and p.
   ParGridFunction u(&state_fes);
   ParGridFunction p(&state_fes);
   ParGridFunction K(&control_fes);
   ParGridFunction K_old(&control_fes);
   ParGridFunction K_tilde(&state_fes);
   ParGridFunction lambda_L2(&control_fes);
   u = 0.0;
   p = 0.0;
   K = 1.0;
   K_tilde = 0.5;
   K_old = 0.0;

   // 8. Set up the linear form b(.) for the state and adjoint equations.


   FPDESolver * PoissonSolver = new FPDESolver();
   PoissonSolver->SetMesh(&pmesh);
   PoissonSolver->SetOrder(order);
   PoissonSolver->SetAlpha(1.0);
   PoissonSolver->SetBeta(0.0);
   PoissonSolver->Init();
   PoissonSolver->SetupFEM();

   int seed = (random_seed) ? rand()%100 + myid : myid;
   // RandomFunctionCoefficient load_coeff(randomload,seed);
   RandomFunctionCoefficient load_coeff(randomload);
   PoissonSolver->SetRHSCoefficient(&load_coeff);


   // OperatorPtr A;
   // Vector B, C, X;

   // 9. Define the gradient function
   ParGridFunction grad(&control_fes);
   ParGridFunction avg_grad(&control_fes);

   // 10. Define some tools for later
   ConstantCoefficient zero(0.0);
   ConstantCoefficient one(1.0);
   ParGridFunction onegf(&control_fes);
   onegf = 1.0;
   ParLinearForm vol_form(&control_fes);
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
   lambda_L2 = 0.0;
   for (int k = 1; k < max_it; k++)
   {
      double adaptive_batch_size = batch_size;

      double norm_K;
      for (int l = 1; l < max_it; l++)
      {
         if (myid == 0)
         {
            cout << "Step = " << l << endl;
            cout << "batch_size = " << adaptive_batch_size << endl;
         }
         avg_grad = 0.0;
         double avg_grad_norm = 0.;

         for (int ib = 0; ib<adaptive_batch_size; ib++)
         {
            GridFunctionCoefficient diffusion_coeff(&K);

            PoissonSolver->SetDiffusionCoefficient(&diffusion_coeff);
            load_coeff.resample();
            PoissonSolver->Solve();
            u = *PoissonSolver->GetFEMSolution();
            // H. Construct gradient function (i.e., -|\nabla u|^2 - λ + β⋅(∫_Ω K dx - V ⋅ vol(\Omega)) - λ_L2 + β⋅(K - \tilde{K})

            // -|\nabla u|^2
            GradientGridFunctionCoefficient grad_u(&u);
            InnerProductCoefficient norm2_grad_u(grad_u,grad_u);
            ProductCoefficient minus_norm2_grad_u(-1.0,norm2_grad_u);

            // -λ_L2
            GridFunctionCoefficient lambda_cf(&lambda_L2);
            ProductCoefficient minus_lambda_cf(-1.0,lambda_cf);

            // β⋅(K - \tilde{K})
            GridFunctionCoefficient K_cf(&K);
            GridFunctionCoefficient K_tilde_cf(&K_tilde);
            SumCoefficient beta_K_diff(K_cf,K_tilde_cf, beta, -beta);

            // -λ + β⋅(∫_Ω K dx - V ⋅ vol(\Omega))
            double c = -lambda + beta * (vol_form(K)/domain_volume - mass_fraction)/domain_volume;
            ConstantCoefficient scalar_terms(c);

            ParLinearForm d(&control_fes);
            d.AddDomainIntegrator(new DomainLFIntegrator(minus_norm2_grad_u));
            if (ADMM)
            {
               d.AddDomainIntegrator(new DomainLFIntegrator(minus_lambda_cf));    
               d.AddDomainIntegrator(new DomainLFIntegrator(beta_K_diff));
            }
            d.AddDomainIntegrator(new DomainLFIntegrator(scalar_terms));
            d.Assemble(false);

            ParBilinearForm L2proj(&control_fes);
            InverseIntegrator * m = new InverseIntegrator(new MassIntegrator());
            L2proj.AddDomainIntegrator(m);
            L2proj.Assemble();
            Array<int> empty_list;
            OperatorPtr invM;
            L2proj.FormSystemMatrix(empty_list,invM);   
            invM->Mult(d,grad);

            avg_grad += grad;
            double grad_norm = grad.ComputeL2Error(zero);
            avg_grad_norm += grad_norm*grad_norm;
         }

         avg_grad_norm /= (double)adaptive_batch_size;  
         avg_grad /= (double)adaptive_batch_size;

         double norm_avg_grad = pow(avg_grad.ComputeL2Error(zero),2);
         double variance = (avg_grad_norm - norm_avg_grad)/(adaptive_batch_size - 1);  

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
         norm_K = K.ComputeL2Error(tmp)/alpha;
         K_old = K;
         if (myid == 0)
         {
            mfem::out << "norm of reduced gradient = " << norm_K << endl;
            mfem::out << "compliance = " << (*(PoissonSolver->GetLinearForm()))(u) << endl;
            mfem::out << "variance = " << variance << std::endl;
         }
         // R = (x_{k} - x_{k+1})/alpha
         // x_{k+1} = x_k - alpha R
         if (norm_K < intermediate_tol_K)
         {
            break;
         }

         double ratio = sqrt(abs(variance)) / norm_K ;
         if (myid == 0)
         {
            mfem::out << "ratio = " << ratio << std::endl;
         }
         MFEM_VERIFY(IsFinite(ratio), "ratio not finite");
         if (ratio > theta)
         {
            // adaptive_batch_size = (int)(pow(ratio / theta,2.) * adaptive_batch_size); 
            adaptive_batch_size +=5;
         }
      }
      // λ <- λ - β (∫_Ω K dx - V⋅ vol(\Omega))
      double mass = vol_form(K);

      if (myid == 0)
      {
         mfem::out << "mass_fraction = " << mass / domain_volume << endl;
      }
      double lambda_inc = mass/domain_volume - mass_fraction;

      lambda -= beta*lambda_inc;

      double norm_gradient = avg_grad.ComputeL2Error(zero);
      if (myid == 0)
      {
         mfem::out << "\ngrad norm = " << norm_gradient << endl;
         mfem::out << "lambda_inc = " << lambda_inc << endl;
         mfem::out << "lambda = " << lambda << endl;
      }
      if (visualization)
      {
         
         sout_u << "parallel " << num_procs << " " << myid << "\n";
         sout_u << "solution\n" << pmesh << u
               << "window_title 'State u'" << flush;

         sout_K << "parallel " << num_procs << " " << myid << "\n";
         sout_K << "solution\n" << pmesh << K
                << "window_title 'Control K'" << flush;
      }

      if (ADMM)
      {
         // -\gamma \Delta \tilde{K} + \beta \tilde{K} = \beta K - \lambda
         {
            ParLinearForm d(&state_fes);
            GridFunctionCoefficient K_cf(&K);
            GridFunctionCoefficient lambda_cf(&lambda_L2);
            SumCoefficient rhs(K_cf,lambda_cf,beta,-1.0);

            FPDESolver ProjectionSolver;
            ConstantCoefficient gamma_cf2(pow(gamma,1./sigma));
            ProjectionSolver.SetMesh(&pmesh);
            ProjectionSolver.SetOrder(order);
            ProjectionSolver.SetAlpha(sigma);
            ProjectionSolver.SetBeta(beta);
            ProjectionSolver.Init();
            ProjectionSolver.SetupFEM();
            ProjectionSolver.SetRHSCoefficient(&rhs);
            ProjectionSolver.SetDiffusionCoefficient(&gamma_cf2);
            Array<int> ess_bdr(pmesh.bdr_attributes.Max()); ess_bdr = 0;
            ProjectionSolver.SetEssentialBoundary(ess_bdr);
            ProjectionSolver.Solve();
            K_tilde = *ProjectionSolver.GetFEMSolution();
         }

         ///// UPDATE lambda_L2 <- lambda_L2 - \beta (K - \tilde{K})
         GridFunctionCoefficient K_tilde_cf(&K_tilde);
         ParGridFunction lambda_L2_inc(&control_fes);
         lambda_L2_inc.ProjectCoefficient(K_tilde_cf);
         lambda_L2_inc -= K;
         double norm_lambda_L2_inc = lambda_L2_inc.ComputeL2Error(zero);

         lambda_L2_inc *= beta;
         lambda_L2 += lambda_L2_inc;

         if (myid == 0)
         {
            cout << " norm_lambda_L2_inc = " << norm_lambda_L2_inc << endl;
         }
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