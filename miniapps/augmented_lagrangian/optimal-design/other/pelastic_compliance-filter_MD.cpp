//
// Compile with: make optimal_design
//
// Sample runs:
// mpirun -np 6 pelastic_compliance-filter_MD -r 4 -o 2 -alpha 1.0 -epsilon 0.01 -mi 50 -mf 0.5 -lambda 0.1 -mu 0.1 -tr 0.0001
// mpirun -np 6 pelastic_compliance-filter_MD -r 5 -o 2 -alpha 10.0 -epsilon 0.01 -mi 50 -mf 0.5 -tr 0.00001
// mpirun -np 8 pelastic_compliance-filter_MD -r 6 -o 2 -alpha 10.0 -epsilon 0.01 -mi 50 -mf 0.5 -tr 0.00001
#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>
#include <random>
#include "../solvers/fpde.hpp"
#include "../solvers/pde_solvers.hpp"


double lnit(double x)
{
   double tol = 1e-12;
   x = min(max(tol,x),1.0-tol);
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

/**
 * @brief Compute the root of the function
 *            f(c) = ∫_Ω expit(lnit(τ) + c) dx - θ vol(Ω)
 */
void projit(GridFunction &tau, double &c, LinearForm &vol_form, double volume_fraction, double tol=1e-12, int max_its=10)
{
   GridFunction ftmp(tau.FESpace());
   GridFunction dftmp(tau.FESpace());
   for (int k=0; k<max_its; k++)
   {
      // Compute f(c) and dfdc(c)
      for (int i=0; i<tau.Size(); i++)
      {
         ftmp[i]  = expit(lnit(tau[i]) + c) - volume_fraction;
         dftmp[i] = dexpitdx(lnit(tau[i]) + c);
      }
      double f = vol_form(ftmp);
      double df = vol_form(dftmp);
      
      MPI_Allreduce(MPI_IN_PLACE,&f,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&df,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

      double dc = -f/df;
      c += dc;
      if (abs(dc) < tol) { break; }
   }
   tau = ftmp;
   tau += volume_fraction;
}


class SIMPCoefficient : public Coefficient
{
protected:
   GridFunction *rho_filter; // grid function
   double min_val;
   double max_val;
   double exponent;

public:
   SIMPCoefficient(GridFunction *rho_filter_, double min_val_= 1e-3, double max_val_=1.0, 
      double exponent_ = 3)
      : rho_filter(rho_filter_), min_val(min_val_), max_val(max_val_),exponent(exponent_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      double val = rho_filter->GetValue(T, ip);
      double coeff = min_val + pow(val,exponent)*(max_val-min_val);
      return coeff;
   }
};


// A Coefficient for computing the components of the stress.
class StrainEnergyDensityCoefficient : public Coefficient
{
protected:
   Coefficient * lambda=nullptr;
   Coefficient * mu=nullptr;
   GridFunction *u = nullptr; // displacement
   GridFunction *rho_filter = nullptr; // filter density
   DenseMatrix grad; // auxiliary matrix, used in Eval
   double exponent;
   double k_min;

public:
   StrainEnergyDensityCoefficient(Coefficient *lambda_, Coefficient *mu_, 
      GridFunction * u_, GridFunction * rho_filter_, double k_min_=1e-6, 
      double exponent_ = 3.0)
      : lambda(lambda_), mu(mu_),  u(u_), rho_filter(rho_filter_), 
          exponent(exponent_), k_min(k_min_)
   {
      MFEM_ASSERT(k_min_ >= 0.0, "k_min must be >= 0");
      MFEM_ASSERT(k_min_ < 1.0,  "k_min must be > 1");
      MFEM_ASSERT(u, "displacement field is not set");
      MFEM_ASSERT(rho_filter, "density field is not set");
   }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      double L = lambda->Eval(T, ip);
      double M = mu->Eval(T, ip);
      u->GetVectorGradient(T, grad);
      double div_u = grad.Trace();
      double density = L*div_u*div_u;
      int dim = T.GetSpaceDim();
      for (int i=0; i<dim; i++)
      {
         for (int j=0; j<dim; j++)
         {
            density += M*grad(i,j)*(grad(i,j)+grad(j,i));
         }
      }
      double val = rho_filter->GetValue(T,ip);

      return -exponent * pow(val, exponent-1.0) * (1-k_min) * density;
   }
};

class VolumeForceCoefficient : public VectorCoefficient
{
private:
   double r;
   Vector center;
   Vector force;
public:
   VolumeForceCoefficient(double r_,Vector &  center_, Vector & force_) : 
   VectorCoefficient(center_.Size()), r(r_), center(center_), force(force_) { }

   virtual void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector xx; xx.SetSize(T.GetDimension());
      T.Transform(ip,xx);
      for(int i=0;i<xx.Size();i++)
      {
         xx[i]=xx[i]-center[i];
      }

      double cr=xx.Norml2();
      V.SetSize(T.GetDimension());
      if (cr <= r) 
      {
         V = force;
      } 
      else
      {
         V = 0.0;
      }
   }

   void Set(double r_,Vector & center_, Vector & force_)
   {
   r=r_;
   center = center_;
   force = force_;
   }
};


using namespace std;
using namespace mfem;

/** The Lagrangian for this problem is
 *    TODO
 * -------------------------------------------------------- 
 * 
 * We update ρ with projected gradient descent via
 * 
 *  1. Initialize ζ, ρ 
 *  while not converged
 *     2. Solve (ϵ² ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)  
 *     3. (λ(ρ̃) ∇⋅u, ∇⋅v) + (2 μ(ρ̃) e(u)), e(v)) = (f,v),
 *        k(ρ̃):= kₘᵢₙ + ρ̃³ (1 - kₘᵢₙ)    
 *        λ(ρ̃):= λ k(ρ̃)    
 *        μ(ρ̃):= μ k(ρ̃)    
 * 
 *     4. Solve (ϵ² ∇ w̃ , ∇ v ) + (w̃ ,v) = (-k'(ρ̃) ( λ(ρ̃) |∇⋅u|² + 2 μ(ρ̃) |e(u)|²),v)
 *     5. Compute gradient in L² w:= M⁻¹ w̃ 
 *     6. update until convergence 
 *       ρ <--- P(ρ - α (w - ζ + β (∫_Ω ρ - V ⋅ vol(Ω)) ) )     
 *              P is the projection operator enforcing 0 <= ρ <= 1
 * 
 *  7. update ζ 
 *     ζ <- ζ - β (∫_Ω K dx - V ⋅ vol(Ω))
 * 
 *  ρ ∈ L² (order p - 1)
 *  ρ̃ ∈ H¹ (order p - 1)
 *  u ∈ (H¹)ᵈ (order p)
 *  w̃ ∈ H¹ (order p - 1)
 *  w ∈ L² (order p - 1)
 */

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();   
   // 1. Parse command-line options.
   // const char *mesh_file = "bar2d.msh";
   int ref_levels = 2;
   int order = 2;
   bool visualization = true;
   double alpha = 1.0;
   double epsilon = 1.0;
   double mass_fraction = 0.4;
   int max_it = 1e2;
   double tol_rho = 5e-2;
   double K_min = 1e-3;
   double lambda = 1.0;
   double mu = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&alpha, "-alpha", "--alpha-step-length",
                  "Step length for gradient descent.");
   args.AddOption(&epsilon, "-epsilon", "--epsilon-thickness",
                  "epsilon phase field thickness");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&tol_rho, "-tr", "--tol_rho",
                  "Exit tolerance for ρ ");     
   args.AddOption(&mass_fraction, "-mf", "--mass-fraction",
                  "Mass fraction for diffusion coefficient.");
   args.AddOption(&lambda, "-lambda", "--lambda",
                  "Lame constant λ");
   args.AddOption(&mu, "-mu", "--mu",
                  "Lame constant μ");                                    
   args.AddOption(&K_min, "-Kmin", "--K-min",
                  "Minimum of density coefficient.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
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
   // Mesh mesh(mesh_file, 1, 1);

   Mesh mesh = Mesh::MakeCartesian2D(3,1,mfem::Element::Type::QUADRILATERAL,true,3.0,1.0);

   int dim = mesh.Dimension();

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


      if (abs(center(0) - 0.0) < 1e-10)
      {
         // the left edge
         be->SetAttribute(1);
      }
      else
      {
         // all other boundaries
         be->SetAttribute(2);
      }
   }
   mesh.SetAttributes();

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 5. Define the vector finite element spaces representing the state variable u,
   //    adjoint variable p, and the control variable f.
   H1_FECollection state_fec(order, dim,BasisType::Positive); // space for u
   H1_FECollection filter_fec(order-1, dim,BasisType::Positive); // space for ρ̃  
   L2_FECollection control_fec(order-1, dim,BasisType::Positive); // space for ρ   
   ParFiniteElementSpace state_fes(&pmesh, &state_fec,dim);
   ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);
   ParFiniteElementSpace control_fes(&pmesh, &control_fec);
   
   HYPRE_BigInt state_size = state_fes.GlobalTrueVSize();
   HYPRE_BigInt control_size = control_fes.GlobalTrueVSize();
   HYPRE_BigInt filter_size = filter_fes.GlobalTrueVSize();
   if (myid==0)
   {
      cout << "Number of state unknowns: " << state_size << endl;
      cout << "Number of filter unknowns: " << filter_size << endl;
      cout << "Number of control unknowns: " << control_size << endl;
   }

   // 7. Set the initial guess for f and the boundary conditions for u.
   ParGridFunction u(&state_fes);
   ParGridFunction rho(&control_fes);
   ParGridFunction rho_old(&control_fes);
   ParGridFunction rho_filter(&filter_fes);
   u = 0.0;
   rho_filter = 0.0;
   rho = 0.5;
   rho_old = 0.5;
   // 8. Set up the linear form b(.) for the state and adjoint equations.
   int maxat = pmesh.bdr_attributes.Max();
   Array<int> ess_bdr(maxat);
   ess_bdr = 0;
   ess_bdr[0] = 1;
   ConstantCoefficient one(1.0);
   ConstantCoefficient lambda_cf(lambda);
   ConstantCoefficient mu_cf(mu);
   LinearElasticitySolver * ElasticitySolver = new LinearElasticitySolver();
   ElasticitySolver->SetMesh(&pmesh);
   ElasticitySolver->SetOrder(state_fec.GetOrder());
   ElasticitySolver->SetupFEM();
   // RandomFunctionCoefficient load_coeff(randomload);
   Vector center(2); center(0) = 2.9; center(1) = 0.5;
   Vector force(2); force(0) = 0.0; force(1) = -1.0;
   double r = 0.05;
   VolumeForceCoefficient vforce_cf(r,center,force);
   ElasticitySolver->SetRHSCoefficient(&vforce_cf);
   ElasticitySolver->SetEssentialBoundary(ess_bdr);

   ConstantCoefficient eps2_cf(epsilon*epsilon);
   FPDESolver * FilterSolver = new FPDESolver();
   FilterSolver->SetMesh(&pmesh);
   FilterSolver->SetOrder(filter_fec.GetOrder());
   FilterSolver->SetAlpha(1.0);
   FilterSolver->SetBeta(1.0);
   FilterSolver->SetDiffusionCoefficient(&eps2_cf);
   Array<int> ess_bdr_filter;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr_filter.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr_filter = 0;
   }
   FilterSolver->SetEssentialBoundary(ess_bdr_filter);
   FilterSolver->Init();
   FilterSolver->SetupFEM();

   ParBilinearForm mass(&control_fes);
   mass.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator(one)));
   mass.Assemble();
   HypreParMatrix M;
   Array<int> empty;
   mass.FormSystemMatrix(empty,M);


   // 9. Define the gradient function
   ParGridFunction w(&control_fes);
   ParGridFunction w_filter(&filter_fes);

   // 10. Define some tools for later
   ConstantCoefficient zero(0.0);
   ParGridFunction onegf(&control_fes);
   onegf = 1.0;
   ParLinearForm vol_form(&control_fes);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble(false);
   double domain_volume = vol_form(onegf);

   // 11. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_u,sout_K,sout_rho;
   if (visualization)
   {
      sout_u.open(vishost, visport);
      sout_rho.open(vishost, visport);
      sout_K.open(vishost, visport);
      sout_u.precision(8);
      sout_rho.precision(8);
      sout_K.precision(8);
   }

   mfem::ParaViewDataCollection paraview_dc("Elastic_compliance", &pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("soln",&u);
   paraview_dc.RegisterField("dens",&rho);

   // 12. AL iterations
   int step = 0;
   double zeta = 0.0;
   for (int k = 1; k < max_it; k++)
   {
      if (k > 1) { alpha *= ((double) k) / ((double) k-1); }
      // if (k > 1) { alpha *= sqrt((double) k) / sqrt((double) k-1); }

      step++;
      // A. Form state equation
      if (myid == 0)
      {
         cout << "\nStep = " << k << endl;
      }
      // Step 2 -  Filter Solve
      // Solve (ϵ^2 ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)  
      GridFunctionCoefficient rho_cf(&rho);
      FilterSolver->SetRHSCoefficient(&rho_cf);
      FilterSolver->Solve();
      rho_filter = *FilterSolver->GetFEMSolution();
      // ------------------------------------------------------------------
      // Step 3 - State Solve
      SIMPCoefficient K_cf(&rho_filter,K_min, 1.0);
      ProductCoefficient lambda_K_cf(lambda_cf,K_cf);
      ProductCoefficient mu_K_cf(mu_cf,K_cf);
      ElasticitySolver->SetLameCoefficients(&lambda_K_cf,&mu_K_cf);
      ParGridFunction K_gf(&control_fes);
      K_gf.ProjectCoefficient(K_cf);
      sout_K << "parallel " << num_procs << " " << myid << "\n";
      sout_K << "solution\n" << pmesh << K_gf
               << "window_title 'Control K'" << flush;         

      ElasticitySolver->Solve();
      u = *ElasticitySolver->GetFEMSolution();
      // ------------------------------------------------------------------
      // Step 4 - Adjoint Solve
      StrainEnergyDensityCoefficient rhs_cf(&lambda_cf,&mu_cf,&u, &rho_filter, K_min);
      FilterSolver->SetRHSCoefficient(&rhs_cf);
      FilterSolver->Solve();
      w_filter = *FilterSolver->GetFEMSolution();
      // Step 5 - get grad of w
      GridFunctionCoefficient w_cf(&w_filter);
      ParLinearForm w_rhs(&control_fes);
      w_rhs.AddDomainIntegrator(new DomainLFIntegrator(w_cf));
      w_rhs.Assemble();
      M.Mult(w_rhs,w);
      // w.ProjectCoefficient(w_cf); // This might need to change to L2-projection
      // ------------------------------------------------------------------

      if (myid == 0)
      {
         cout << "norm of u = " << u.Norml2() << endl;
      }

      // step 6-update  ρ 
      for (int i = 0; i < rho.Size(); i++)
      {
         rho[i] = expit(lnit(rho[i]) - alpha*w[i]);
      }
      projit(rho, zeta, vol_form, mass_fraction);

      GridFunctionCoefficient tmp(&rho_old);
      double norm_rho = rho.ComputeL2Error(tmp)/alpha;
      rho_old = rho;

      double compliance = (*(ElasticitySolver->GetLinearForm()))(u);
      MPI_Allreduce(MPI_IN_PLACE,&compliance,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      if (myid == 0)
      {
         mfem::out << "norm of reduced gradient = " << norm_rho << endl;
         mfem::out << "compliance = " << compliance << endl;
      }

      double mass1 = vol_form(rho);
      if (myid == 0)
      {
         mfem::out << "mass_fraction = " << mass1 / domain_volume << endl;
         mfem::out << "zeta = " << zeta << endl;
      }

      if (visualization)
      {

         sout_u << "parallel " << num_procs << " " << myid << "\n";
         sout_u << "solution\n" << pmesh << u
               << "window_title 'State u'" << flush;

         sout_rho << "parallel " << num_procs << " " << myid << "\n";
         sout_rho << "solution\n" << pmesh << rho
                << "window_title 'Control ρ '" << flush;
     
         paraview_dc.SetCycle(k);
         paraview_dc.SetTime((double)k);
         paraview_dc.Save();
      }
      
      if (norm_rho < tol_rho)
      {
         break;
      }
   }

   return 0;
}