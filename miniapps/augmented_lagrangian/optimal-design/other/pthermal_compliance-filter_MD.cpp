//
// Compile with: make optimal_design
//
// Sample runs:
// mpirun -np 6 ./pthermal_compliance-filter_MD -epsilon 0.01 -alpha 0.01 -r 4 -o 2 -mi 50
// mpirun -np 8 pthermal_compliance-filter_MD -epsilon 0.01 -alpha 0.01 -r 3 -o 2 -mi 100 -dim 3
// mpirun -np 8 pthermal_compliance-filter_MD -epsilon 0.01 -alpha 0.01 -r 2 -o 2 -mi 100 -dim 3 -cylinder
//
//         min J(K) = <g,u>
//                            
//                        Γ_1           Γ_2            Γ_1
//               _ _ _ _ _ _ _ _ _ _ _________ _ _ _ _ _ _ _ _ _ _   
//              |         |         |         |         |         |  
//              |         |         |         |         |         |  
//              |---------|---------|---------|---------|---------|  
//              |         |         |         |         |         |  
//              |         |         |         |         |         |  
//      Γ_1-->  |---------|---------|---------|---------|---------|  <-- Γ_1
//              |         |         |         |         |         |  
//              |         |         |         |         |         |  
//              |---------|---------|---------|---------|---------|  
//              |         |         |         |         |         |  
//              |         |         |         |         |         |  
//               -------------------------------------------------|
//                       |̂                              |̂  
//                      Γ_1                            Γ_1                    
//
//
//         subject to   - div( K\nabla u ) = f    in \Omega
//                                       u = 0    on Γ_2
//                               (K ∇ u)⋅n = 0    on Γ_1
//         and                   ∫_Ω K dx <= V ⋅ vol(\Omega)
//         and                  a <= K(x) <= b

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>
#include <random>
#include "../solvers/fpde.hpp"

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

class DiffusionCoefficient : public Coefficient
{
protected:
   GridFunction *rho_filter; // grid function
   double min_val;
   double max_val;
   double exponent;

public:
   DiffusionCoefficient(GridFunction &rho_filter_, double min_val_= 1e-3, double max_val_=1.0, 
      double exponent_ = 3)
      : rho_filter(&rho_filter_), min_val(min_val_), max_val(max_val_),exponent(exponent_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      double val = rho_filter->GetValue(T, ip);
      double coeff = min_val + pow(val,exponent)*(max_val-min_val);
      return coeff;
   }
};

class GradientRHSCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   GridFunction *rho_filter; // grid function
   double min_val;
   double max_val;
   double exponent;

public:
   GradientRHSCoefficient(GridFunction &u_, GridFunction & rho_filter_, 
      double min_val_= 1e-3, double max_val_=1.0, double exponent_ = 3.0)
      : u(&u_), rho_filter(&rho_filter_), min_val(min_val_), max_val(max_val_), 
         exponent(exponent_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);
      double val = rho_filter->GetValue(T,ip);
      Vector gradu;
      u->GetGradient(T,gradu);
      return -exponent * pow(val, exponent-1.0) * (max_val-min_val) * (gradu * gradu); 
   }
};


using namespace std;
using namespace mfem;

// Let H^1_Γ_1 := {v ∈ H^1(Ω) | v|Γ_1 = 0}
/** The Lagrangian for this problem is
 *    TODO
 * -------------------------------------------------------- 
 * 
 * We update ρ with projected gradient descent via
 * 
 *  1. Initialize λ, ρ 
 *  while not converged
 *     2. Solve (ϵ^2 ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)  
 *     3. Solve (k(ρ̃) ∇ u , ∇ v) = (f,v) , k(ρ̃):= K_min + ρ̃^3 (K_max- Kmin)  
 *     4. Solve (ϵ^2 ∇ w̃ , ∇ v ) + (w̃ ,v) = (-k'(ρ̃) |∇ u|^2 ,v)
 *     5. Compute gradient in L2 w:= M^-1 w̃ 
 *     6. update until convergence 
 *       ρ <--- P(ρ - α (w - λ + β (∫_Ω ρ - V ⋅ vol(Ω)) ) )     
 *              P is the projection operator enforcing 0 <= ρ <= 1
 * 
 *  7. update λ 
 *     λ <- λ - β (∫_Ω K dx - V ⋅ vol(Ω))
 * 
 *  ρ ∈ L^2 (order p - 1)
 *  ρ̃ ∈ H^1 (order p - 1)
 *  u ∈ H^1 (order p)
 *  w̃ ∈ H^1 (order p - 1)
 *  w ∈ L^2 (order p - 1)
 */

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();   
   // 1. Parse command-line options.
   int ref_levels = 2;
   int sref_levels = 0;
   int order = 2;
   bool visualization = true;
   double alpha = 1.0;
   double epsilon = 1.0;
   double mass_fraction = 0.4;
   int max_it = 1e2;
   double tol_rho = 1e-2;
   double K_max = 1.0;
   double K_min = 1e-3;
   bool use_cylinder = false;
   int dim = 2;
   const char *paraview_file = nullptr;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the parallel mesh uniformly.");
   args.AddOption(&sref_levels, "-sr", "--ser-refine",
                  "Number of times to refine the serial mesh uniformly.");                  
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&dim, "-dim", "--dim",
                  "Dimension of the problem.");                  
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
   args.AddOption(&K_max, "-Kmax", "--K-max",
                  "Maximum of diffusion diffusion coefficient.");
   args.AddOption(&K_min, "-Kmin", "--K-min",
                  "Minimum of diffusion diffusion coefficient.");
   args.AddOption(&use_cylinder, "-cylinder", "--use-cylinder", "-no-cylinder",
                  "--no-cylinder", "Use cylinder mesh");                  
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview_file, "-paraview-file", "--paraview-file",
                  "Paraview file name");                  

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

   Mesh * mesh = nullptr;
   if (use_cylinder)
   {
      const char *mesh_file = "cylinder2.mesh";
      mesh = new Mesh(mesh_file,1,1);
   }
   else
   {
      if (dim == 2)
      {
         mesh = new Mesh(Mesh::MakeCartesian2D(7,7,mfem::Element::Type::QUADRILATERAL,true,1.0,1.0));
      }
      else
      {
         mesh = new Mesh(Mesh::MakeCartesian3D(7,7,7,mfem::Element::Type::HEXAHEDRON,true,1.0,1.0,1.0));
      }
   }

   dim = mesh->Dimension();

   double lz = (use_cylinder) ? 2.0 : 1.0;
   double lx = (use_cylinder) ? 0.0883883 : 0.5;
   double ly = (use_cylinder) ? 0.0883883 : 0.5;


   if (dim == 2)
   {
      for (int i = 0; i<mesh->GetNBE(); i++)
      {
         Vector center(dim);
         int bdrgeom = mesh->GetBdrElementBaseGeometry(i);
         ElementTransformation * tr = mesh->GetBdrElementTransformation(i);
         tr->Transform(Geometries.GetCenter(bdrgeom),center);
         if (abs(center(1) - 1.0) < 1e-10 && abs(center(0)-0.5) < 1e-10)
         {
            // top edge
            mesh->SetBdrAttribute(i,2);
         }
         else
         {
            // all other boundaries
            mesh->SetBdrAttribute(i,1);
         }
      }
   }
   else
   {
      for (int i = 0; i<mesh->GetNBE(); i++)
      {
         Vector center(dim);
         int bdrgeom = mesh->GetBdrElementBaseGeometry(i);
         ElementTransformation * tr = mesh->GetBdrElementTransformation(i);
         tr->Transform(Geometries.GetCenter(bdrgeom),center);
         if (abs(center(2) - lz) < 1e-6 && abs(abs(center(0))-lx) < 1e-6 && abs(abs(center(1))-ly) < 1e-6)
         {
            // top face
            mesh->SetBdrAttribute(i,2);
         }
         else
         {
            // all other boundaries
            mesh->SetBdrAttribute(i,1);
         }
      }
   }

   mesh->SetAttributes();

   for (int lev = 0; lev < sref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   for (int lev = 0; lev < ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }

   // 5. Define the vector finite element spaces representing the state variable u,
   //    adjoint variable p, and the control variable f.
   H1_FECollection state_fec(order, dim); // space for u
   H1_FECollection filter_fec(order-1, dim); // space for ρ̃
   L2_FECollection control_fec(order-1, dim,BasisType::Positive); // space for ρ
   ParFiniteElementSpace state_fes(&pmesh, &state_fec);
   ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);
   ParFiniteElementSpace control_fes(&pmesh, &control_fec);

   // For ParaView export of curl
   ND_FECollection ND_fec(order, dim); // space for grad(u)
   ParFiniteElementSpace ND_fes(&pmesh, &ND_fec);
   ParGridFunction grad_u_gf(&ND_fes);
   
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
   if (maxat > 0)
   {
      ess_bdr[maxat-1] = 1;
   }
   ConstantCoefficient one(1.0);
   FPDESolver * PoissonSolver = new FPDESolver();
   PoissonSolver->SetMesh(&pmesh);
   PoissonSolver->SetOrder(state_fec.GetOrder());
   PoissonSolver->SetAlpha(1.0);
   PoissonSolver->SetBeta(0.0);
   PoissonSolver->SetupFEM();
   // RandomFunctionCoefficient load_coeff(randomload);
   PoissonSolver->SetRHSCoefficient(&one);
   PoissonSolver->SetEssentialBoundary(ess_bdr);
   PoissonSolver->Init();

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
   socketstream sout_u,sout_K,sout_rho, sout_rho_filter;
   if (visualization)
   {
      sout_u.open(vishost, visport);
      sout_rho.open(vishost, visport);
      sout_K.open(vishost, visport);
      sout_rho_filter.open(vishost, visport);
      sout_u.precision(8);
      sout_rho.precision(8);
      sout_K.precision(8);
      sout_rho_filter.precision(8);
   }

   
   std::string y("Thermal_compliance_");
   y+=paraview_file;

   mfem::ParaViewDataCollection paraview_dc(y, &pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("soln",&u);
   paraview_dc.RegisterField("rho",&rho);
   paraview_dc.RegisterField("rho_filter",&rho_filter);
   paraview_dc.RegisterField("grad_soln",&grad_u_gf);

   // 12. AL iterations
   int step = 0;
   double lambda = 0.0;
   for (int k = 1; k < max_it; k++)
   {
      if (k > 1) { alpha *= ((double) k) / ((double) k-1); }
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
      DiffusionCoefficient K(rho_filter,K_min, K_max);
      ParGridFunction k_cf(&control_fes);
      k_cf.ProjectCoefficient(K);
      sout_K << "parallel " << num_procs << " " << myid << "\n";
      sout_K << "solution\n" << pmesh << k_cf
               << "window_title 'Control K'" << flush;         

      PoissonSolver->SetDiffusionCoefficient(&K);
      PoissonSolver->Solve();
      u = *PoissonSolver->GetFEMSolution();
      // ------------------------------------------------------------------
      // Step 4 - Adjoint Solve
      GradientRHSCoefficient rhs_cf(u,rho_filter,K_min, K_max);
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
      projit(rho, lambda, vol_form, mass_fraction);

      GridFunctionCoefficient tmp(&rho_old);
      double norm_rho = rho.ComputeL2Error(tmp)/alpha;
      rho_old = rho;

      double compliance = (*(PoissonSolver->GetLinearForm()))(u);
      if (myid == 0)
      {
         mfem::out << "norm of reduced gradient = " << norm_rho << endl;
         mfem::out << "compliance = " << compliance << endl;
      }

      double mass1 = vol_form(rho);
      if (myid == 0)
      {
         mfem::out << "mass_fraction = " << mass1 / domain_volume << endl;
         mfem::out << "lambda = " << lambda << endl;
      }

      if (visualization)
      {

         sout_u << "parallel " << num_procs << " " << myid << "\n";
         sout_u << "solution\n" << pmesh << u
               << "window_title 'State u'" << flush;

         sout_rho << "parallel " << num_procs << " " << myid << "\n";
         sout_rho << "solution\n" << pmesh << rho
                << "window_title 'Density ρ '" << flush;

         sout_rho_filter << "parallel " << num_procs << " " << myid << "\n";
         sout_rho_filter << "solution\n" << pmesh << rho_filter
                << "window_title 'Filtered density ρ̃ '" << flush;
     
         GradientGridFunctionCoefficient grad_u_cf(&u);
         grad_u_gf.ProjectCoefficient(grad_u_cf);

         paraview_dc.SetCycle(step);
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
