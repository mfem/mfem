//                                Solution of distributed control problem
//
// Compile with: make optimal_design
//
// Sample runs:
//    ./ouu_heat_sink -alpha 0.1 -beta 0.1 -admm -r 3 -o 2 -gamma 0.01 -sigma 1
//    ./ouu_heat_sink -alpha 0.1 -beta 0.1 -admm -r 3 -o 2 -gamma 0.01 -sigma 0.5
//    ./ouu_heat_sink -alpha 0.1 -beta 0.1 -admm -r 3 -o 2 -gamma 0.01 -sigma 0.2
//    ./ouu_heat_sink -alpha 0.1 -beta 0.1 -r 3 -o 2
// Description:  This example solves the following PDE-constrained
//               optimization problem:
//
//         min J(K) = <g,u>
//                            
//                                 Γ_2    
//               _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _   
//              |         |         |         |         |  
//              |         |         |         |         |  
//              |---------|---------|---------|---------|  
//              |         |         |         |         |  
//              |         |         |         |         |  
//              |---------|---------|---------|---------|  
//              |         |         |         |         |  
//              |         |         |         |         |  
//              |---------|---------|---------|---------|  
//              |         |         |         |         |  
//              |         |         |         |         |  
//               ---------------------------------------  
//                  |̂                              |̂  
//                 Γ_1                            Γ_1  
//
//
//
//         subject to   - div( K\nabla u ) = 0    in \Omega
//                                       u = 0    on Γ_1
//                               (K ∇ u)⋅n = g    on Γ_2
//                                   ∇ u⋅n = 0    on ∂Ω\(Γ_1 ∪ Γ_2) 
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

// Let H^1_Γ_1 := {v ∈ H^1(Ω) | v|Γ_1 = 0}
/** The Lagrangian for this problem is
 *    
 *    L(u,K,p,λ) = <g,u> - (K ∇u, ∇p) + <g,v> - λ (∫_Ω K dx - V ⋅ vol(Ω))
 *                                    + β/2 (∫_Ω K dx - V ⋅ vol(Ω))^2    
 *      u, p \in H^1_Γ_1
 *      K \in L^∞(Ω)
 * 
 *  Note that
 * 
 *    ∂_p L = 0        (1)
 *  
 *  delivers the state equation
 *    
 *    (K ∇u, ∇ v) = <g,v> for all v in 
 * 
 *  and
 *  
 *    ∂_u L = 0        (2)
 * 
 *  delivers the adjoint equation (same as the state eqn)
 * 
 *    (∇ p, ∇ v) = <g,v>  for all v H^1_Γ_1
 *    
 *  and at the solutions u=p of (1) and (2), respectively,
 * 
 *  D_K J = D_K L = ∂_u L ∂_K u + ∂_p L ∂_K p
 *                + ∂_K L
 *                = ∂_K L
 *                = (-|∇ u|^2 - λ + β(∫_Ω K dx - V ⋅ vol(Ω)), ⋅)
 * 
 * We update the control K_k with projected gradient descent via
 * 
 *  1. Initialize λ 
 *  2. update until convergence 
 *     K <- P (K + α |∇ u|^2 + λ)
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
   double a = 0.0;
   double b = 1.0;
   double m;
   std::default_random_engine generator;
   std::uniform_real_distribution<double> * distribution;
   double (*Function)(const Vector &, double);
public:
   RandomFunctionCoefficient(double (*F)(const Vector &, double)) 
   : Function(F) 
   {
      distribution = new std::uniform_real_distribution<double> (a,b);
      m = (*distribution)(generator);
   }
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      Vector transip(3);
      T.Transform(ip, transip);
      return ((*Function)(transip, m));
   }
   void resample()
   {
      m = (*distribution)(generator);
   }
};

double randomload(const Vector & X, double m)
{
   double sigma = 0.1;
   double alpha = 1.0/(sigma * sqrt(2.0*M_PI));
   double beta = -0.5*pow( (X(0)-0.5)/sigma,2);
   // double beta = -0.5*pow( (X(0)-m)/sigma,2);
   return alpha * exp(beta);
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

   bool ADMM = false;


   OptionsParser args(argc, argv);
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

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   double initial_primal_tolerance = tol_K;
   double initial_dual_tolerance = tol_lambda;

   ostringstream file_name;
   file_name << "conv_order" << order << "_GD" << ".csv";
   ofstream conv(file_name.str().c_str());
   conv << "Step,    Compliance,    Mass Fraction" << endl;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   mesh.UniformRefinement();

   Vector center;
   for (int i = 0; i<mesh.GetNBE(); i++)
   {
      Element * be = mesh.GetBdrElement(i);
      Array<int> vertices;
      be->GetVertices(vertices);

      double * coords1 = mesh.GetVertex(vertices[0]);
      double * coords2 = mesh.GetVertex(vertices[1]);

      Vector center(2);
      center(0) = 0.5*(coords1[0] + coords2[0]);
      center(1) = 0.5*(coords1[1] + coords2[1]);


      if (abs(center(1) - 1.0) < 1e-10)
      {
         // the top edge
         be->SetAttribute(1);
      }
      else if(abs(center(1)) < 1e-10 && (center(0) < 0.125 || center(0) > 0.875))
      {
         // bottom edge (left and right "corners")
         be->SetAttribute(2);
      }
      else
      {
         be->SetAttribute(3);
      }
   }
   mesh.SetAttributes();

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


   // Auxiliary FE spaces for gradient
   ND_FECollection NDfec(order,dim);
   FiniteElementSpace NDfes(&mesh,&NDfec);


   int state_size = state_fes.GetTrueVSize();
   int control_size = control_fes.GetTrueVSize();
   cout << "Number of state unknowns: " << state_size << endl;
   cout << "Number of control unknowns: " << control_size << endl;

   // 7. Set the initial guess for f and the boundary conditions for u and p.
   GridFunction u(&state_fes);
   GridFunction K(&control_fes);
   GridFunction K_old(&control_fes);
   GridFunction K_tilde(&state_fes);
   GridFunction lambda_L2(&control_fes);
   K = 1.0;
   K_tilde = 0.5;
   K_old = 0.0;

   // 8. Set up the linear form b(.) for the state and adjoint equations.

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   Array<int> inhomogenous_neuman_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[1] = 1;
   inhomogenous_neuman_bdr = 0;
   inhomogenous_neuman_bdr[0] = 1;


   FPDESolver * PoissonSolver = new FPDESolver();
   PoissonSolver->SetMesh(&mesh);
   PoissonSolver->SetOrder(order);
   PoissonSolver->SetAlpha(1.0);
   PoissonSolver->SetBeta(0.0);
   PoissonSolver->SetupFEM();
   RandomFunctionCoefficient load_coeff(randomload);
   PoissonSolver->SetEssentialBoundary(ess_bdr);
   PoissonSolver->SetNeumannBoundary(inhomogenous_neuman_bdr);
   PoissonSolver->SetNeumannData(&load_coeff);
   PoissonSolver->Init();



   OperatorPtr A;
   Vector B, C, X;

   // 9. Define the gradient function
   GridFunction grad(&control_fes);
   GridFunction avg_grad(&control_fes);

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
   socketstream sout_K, sout_u;
   if (visualization)
   {
      sout_K.open(vishost, visport);
      sout_K.precision(8);
      sout_u.open(vishost, visport);
      sout_u.precision(8);
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

   IntegrationRules rules(0, Quadrature1D::GaussLegendre);
   const IntegrationRule &ir = rules.Get(state_fes.GetFE(0)->GetGeomType(),
                               2 * order - 1);

   for (int k = 1; k < max_it; k++)
   {
      double adaptive_batch_size = batch_size;

      double norm_K;
      for (int l = 1; l < max_it; l++)
      {
         cout << "Step = " << l << endl;
         cout << "batch_size = " << adaptive_batch_size << endl;
         avg_grad = 0.0;
         double avg_grad_norm = 0.;
         double avg_compliance = 0.;

         GridFunctionCoefficient diffusion_coeff(&K);
         for (int ib = 0; ib<adaptive_batch_size; ib++)
         {

            PoissonSolver->SetDiffusionCoefficient(&diffusion_coeff);
            load_coeff.resample();
            PoissonSolver->Solve();
            u = *PoissonSolver->GetFEMSolution();

            cout << "norm of u = " << u.Norml2() << endl;
            // H. Construct gradient function (i.e., -|\nabla u|^2 - λ + β⋅(∫_Ω K dx - V ⋅ vol(\Omega)) - λ_L2 + β⋅(K - \tilde{K})

            // -|\nabla u|^2
            GradientGridFunctionCoefficient grad_u(&u);


            MixedBilinearForm agrad(&state_fes, &NDfes);
            agrad.AddDomainIntegrator(new GradientInterpolator());
            agrad.Assemble();
            GridFunction grad_gf(&NDfes);
            agrad.Mult(u,grad_gf);

            MixedBilinearForm inner_prod(&NDfes,&control_fes);
            VectorGridFunctionCoefficient grad_cf(&grad_gf);
            inner_prod.AddDomainIntegrator(new VectorInnerProductInterpolator(grad_cf));
            inner_prod.Assemble();
            GridFunction norm2_grad(&control_fes);
            inner_prod.Mult(grad_gf,norm2_grad);
            GridFunctionCoefficient norm2_grad_cf(&norm2_grad);
            ProductCoefficient minus_norm2_grad_u(-1.0,norm2_grad_cf);

            // InnerProductCoefficient norm2_grad_u(grad_u,grad_u);
            // ProductCoefficient minus_norm2_grad_u(-1.0,norm2_grad_u);

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

            LinearForm d(&control_fes);
            d.AddDomainIntegrator(new DomainLFIntegrator(minus_norm2_grad_u,&ir));
            if (ADMM)
            {
               d.AddDomainIntegrator(new DomainLFIntegrator(minus_lambda_cf,&ir));    
               d.AddDomainIntegrator(new DomainLFIntegrator(beta_K_diff,&ir));
            }
            d.AddDomainIntegrator(new DomainLFIntegrator(scalar_terms,&ir));
            d.Assemble(false);

            BilinearForm L2proj(&control_fes);
            InverseIntegrator * m = new InverseIntegrator(new MassIntegrator());
            m->SetIntegrationRule(ir);
            L2proj.AddDomainIntegrator(m);
            L2proj.Assemble();
            Array<int> empty_list;
            OperatorPtr invM;
            L2proj.FormSystemMatrix(empty_list,invM);   
            invM->Mult(d,grad);

            avg_grad += grad;
            double grad_norm = grad.ComputeL2Error(zero);
            avg_grad_norm += grad_norm*grad_norm;
            avg_compliance += (*(PoissonSolver->GetLinearForm()))(u);

         }
         avg_grad_norm /= (double)adaptive_batch_size;  
         avg_grad /= (double)adaptive_batch_size;
         avg_compliance /= (double)adaptive_batch_size;  


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
         mfem::out << "norm of reduced gradient = " << norm_K << endl;
         mfem::out << "avg_compliance = " << avg_compliance << endl;
         mfem::out << "variance = " << variance << std::endl;

         // R = (x_{k} - x_{k+1})/alpha
         // x_{k+1} = x_k - alpha R
         if (norm_K < intermediate_tol_K)
         {
            break;
         }

         double ratio = sqrt(abs(variance)) / norm_K ;
         mfem::out << "ratio = " << ratio << std::endl;
         MFEM_VERIFY(IsFinite(ratio), "ratio not finite");
         if (ratio > theta)
         {
            // adaptive_batch_size = (int)(pow(ratio / theta,2.) * adaptive_batch_size); 
            adaptive_batch_size +=5;
         }
      }
      // λ <- λ - β (∫_Ω K dx - V⋅ vol(\Omega))
      double mass = vol_form(K);

      mfem::out << "mass_fraction = " << mass / domain_volume << endl;

      double lambda_inc = mass/domain_volume - mass_fraction;

      lambda -= beta*lambda_inc;

      double norm_gradient = avg_grad.ComputeL2Error(zero);
      mfem::out << "\ngrad norm = " << norm_gradient << endl;

      mfem::out << "lambda_inc = " << lambda_inc << endl;
      mfem::out << "lambda = " << lambda << endl;

      if (visualization)
      {
         sout_K << "solution\n" << mesh << K
                << "window_title 'Control K'" << flush;
         sout_u << "solution\n" << mesh << u
                << "window_title 'Solution u'" << flush;       
      }

      cin.get();

      if (ADMM)
      {
         // -\gamma \Delta \tilde{K} + \beta \tilde{K} = \beta K - \lambda
         {
            LinearForm d(&state_fes);
            GridFunctionCoefficient K_cf(&K);
            GridFunctionCoefficient lambda_cf(&lambda_L2);
            SumCoefficient rhs(K_cf,lambda_cf,beta,-1.0);

            FPDESolver ProjectionSolver;
            ConstantCoefficient gamma_cf2(pow(gamma,1./sigma));
            ProjectionSolver.SetMesh(&mesh);
            ProjectionSolver.SetOrder(order);
            ProjectionSolver.SetAlpha(sigma);
            ProjectionSolver.SetBeta(beta);
            ProjectionSolver.Init();
            ProjectionSolver.SetupFEM();
            ProjectionSolver.SetRHSCoefficient(&rhs);
            ProjectionSolver.SetDiffusionCoefficient(&gamma_cf2);
            Array<int> ess_bdr(mesh.bdr_attributes.Max()); ess_bdr = 0;
            ProjectionSolver.SetEssentialBoundary(ess_bdr);
            ProjectionSolver.Solve();
            K_tilde = *ProjectionSolver.GetFEMSolution();

         }

         ///// UPDATE lambda_L2 <- lambda_L2 - \beta (K - \tilde{K})
         GridFunctionCoefficient K_tilde_cf(&K_tilde);
         GridFunction lambda_L2_inc(&control_fes);
         lambda_L2_inc.ProjectCoefficient(K_tilde_cf);
         lambda_L2_inc -= K;
         double norm_lambda_L2_inc = lambda_L2_inc.ComputeL2Error(zero);

         lambda_L2_inc *= beta;
         lambda_L2 += lambda_L2_inc;

         cout << " norm_lambda_L2_inc = " << norm_lambda_L2_inc << endl;
      }

      if (abs(lambda_inc) < intermediate_tol_K)
      {
         intermediate_tol_K *= primal_tolerance_decay_factor;
         intermediate_tol_lambda *= dual_tolerance_decay_factor;
      }

      if ((abs(lambda_inc) < tol_lambda) && (norm_K < tol_K))
      {
         break;
      }

   }

   return 0;
}