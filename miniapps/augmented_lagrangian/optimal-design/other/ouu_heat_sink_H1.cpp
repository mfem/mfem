//
// Compile with: make optimal_design
//
// Sample runs:
//    ./ouu_heat_sink_H1 -gamma 0.1 -epsilon 0.05 -alpha 0.01 -beta 5.0
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
//         and                   ∫_Ω K dx <= V ⋅ vol(\Omega)
//         and                  a <= K(x) <= b

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>
#include <random>
#include "../solvers/fpde.hpp"


using namespace std;
using namespace mfem;

// Let H^1_Γ_1 := {v ∈ H^1(Ω) | v|Γ_1 = 0}
/** The Lagrangian for this problem is
 *    
 *    L(u,K,p,λ) = <g,u> - (K ∇u, ∇p) + <g,v>
 *                + γϵ/2 (∇K, ∇K)
 *                + γ/(2ϵ) ∫_Ω K(1-K) dx
 *                - λ (∫_Ω K dx - V ⋅ vol(Ω))
 *                + β/2 (∫_Ω K dx - V ⋅ vol(Ω))^2    
 *      u, p \in H^1_Γ_1
 *      K \in H^1(Ω)
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
 *                + γϵ(∇ K,∇⋅) + γ/ϵ(1/2-K,⋅)
 * 
 * We update the control K_k with projected gradient descent via
 * 
 *  1. Initialize λ 
 *  2. update until convergence 
 *     K <- P (K - α( γ/ϵ(1/2+K) - λ + β(∫_Ω K dx - V ⋅ vol(Ω)) - R^{-1}( |∇ u|^2 + 2K ) )
 *  3. update λ 
 *     λ <- λ - β (∫_Ω K dx - V ⋅ vol(Ω))
 * 
 * P is the projection operator enforcing a <= K(x) <= b, and α  is a specified
 * step length.
 * 
 */

class RandomFunctionCoefficient : public Coefficient
{
private:
   double a = .49;
   double b = 0.51;
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
//    double beta = -0.5*pow( (X(0)-0.5)/sigma,2);
   double beta = -0.5*pow( (X(0)-m)/sigma,2);
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
   double gamma = 1.0;
   double epsilon = 1.0;
   double theta = 0.5;
   double mass_fraction = 0.5;
   double compliance_max = 0.15;
   int max_it = 1e2;
   double tol_K = 1e-2;
   double tol_lambda = 1e-2;
   double K_max = 1.0;
   double K_min = 1e-3;
   int prob = 0;
   int batch_size_min = 2;

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
   args.AddOption(&max_it, "-./ouu_heat_sink_H1 -gamma 0.1 -epsilon 0.05 -alpha 0.01 -beta 5.0mi", "--max-it",
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
   conv << "Step,    Sample Size,    Compliance,    Mass Fraction" << endl;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   mesh.UniformRefinement();

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
   H1_FECollection control_fec(order-1, dim, BasisType::Positive);
   FiniteElementSpace state_fes(&mesh, &state_fec);
   FiniteElementSpace control_fes(&mesh, &control_fec);

   int state_size = state_fes.GetTrueVSize();
   int control_size = control_fes.GetTrueVSize();
   cout << "Number of state unknowns: " << state_size << endl;
   cout << "Number of control unknowns: " << control_size << endl;

   // 7. Set the initial guess for f and the boundary conditions for u.
   GridFunction u(&state_fes);
   GridFunction K(&control_fes);
   GridFunction K_old(&control_fes);
   u = 0.0;
   K = 0.5;
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


   ConstantCoefficient eps2_cf(epsilon*epsilon);
   FPDESolver * H1Projection = new FPDESolver();
   H1Projection->SetMesh(&mesh);
   H1Projection->SetOrder(order-1);
   H1Projection->SetAlpha(1.0);
   H1Projection->SetBeta(1.0);
   H1Projection->SetDiffusionCoefficient(&eps2_cf);
   Array<int> ess_bdr_K(mesh.bdr_attributes.Max()); ess_bdr_K = 0;
   H1Projection->SetEssentialBoundary(ess_bdr_K);
   H1Projection->Init();
   H1Projection->SetupFEM();

   // b.AddDomainIntegrator(new DomainLFIntegrator(f));
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

   mfem::ParaViewDataCollection paraview_dc("Heat_sink", &mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("soln",&u);
   paraview_dc.RegisterField("dens",&K);
   paraview_dc.Save();

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
   int step = 0;
   double lambda = 0.0;
   for (int k = 1; k < max_it; k++)
   {
      // A. Form state equation

      for (int l = 1; l < max_it; l++)
      {
         step++;
         cout << "Step = " << l << endl;
         cout << "batch_size = " << batch_size << endl;

         avg_grad = 0.0;
         double avg_grad_norm = 0.;
         double avg_compliance = 0.;

         GridFunctionCoefficient diffusion_coeff(&K);
         double mf = vol_form(K)/domain_volume;
         for (int ib = 0; ib<batch_size; ib++)
         {
            PoissonSolver->SetDiffusionCoefficient(&diffusion_coeff);
            load_coeff.resample();
            PoissonSolver->Solve();
            u = *PoissonSolver->GetFEMSolution();

            cout << "norm of u = " << u.Norml2() << endl;

            // H. Constuct gradient function
            // i.e., ∇ J = γ/ϵ (1/2 + K) - λ + β(∫_Ω K dx - V ⋅ vol(Ω)) - R^{-1}(|∇u|^2 + 2γ/ϵ K)
            GradientGridFunctionCoefficient grad_u(&u);
            InnerProductCoefficient norm2_grad_u(grad_u,grad_u);
            SumCoefficient grad_cf(norm2_grad_u,diffusion_coeff,-1.0,-2.0*gamma/epsilon);
            H1Projection->SetRHSCoefficient(&grad_cf);
            H1Projection->Solve();
            
            grad = K;
            grad += (K_max-K_min)/2.0;
            grad *= gamma/epsilon;
            grad += *H1Projection->GetFEMSolution();

            // - λ + β(∫_Ω K dx - V ⋅ vol(\Omega)))
            grad -= lambda;
            grad += beta * (mf - mass_fraction)/domain_volume;

            avg_grad += grad;
            double grad_norm = grad.ComputeL2Error(zero);
            avg_grad_norm += grad_norm*grad_norm;
            avg_compliance += (*(PoissonSolver->GetLinearForm()))(u);

         } // enf of loop through batch samples
         avg_grad_norm /= (double)batch_size;  
         avg_grad /= (double)batch_size;
         avg_compliance /= (double)batch_size;  

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
         mfem::out << "avg_compliance = " << avg_compliance << endl;
         mfem::out << "variance = " << variance << std::endl;
         if (norm_K < tol_K)
         {
            break;
         }


         double ratio = sqrt(abs(variance)) / norm_K ;
         mfem::out << "ratio = " << ratio << std::endl;
         conv << step << ",   " << batch_size << ",   " << avg_compliance << ",   " << mf << endl;
         
         
         MFEM_VERIFY(IsFinite(ratio), "ratio not finite");
         if (ratio > theta)
         {
            batch_size = (int)(pow(ratio / theta,2.) * batch_size); 
         }
        //  else if (ratio < 0.1*theta)
        //  {
        //     batch_size = max(batch_size/2,batch_size_min);
        //  }

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

         paraview_dc.SetCycle(step);
         paraview_dc.SetTime((double)k);
         paraview_dc.Save();
      }

      if (abs(lambda_inc) < tol_lambda)
      {
         break;
      }

   }

   return 0;
}