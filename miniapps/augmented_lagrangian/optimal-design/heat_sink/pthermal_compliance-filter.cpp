//
// Compile with: make optimal_design
//
// Sample runs:
// mpirun -np 6 ./pthermal_compliance-filter -epsilon 0.01 -alpha 0.1 -beta 5.0 -r 4 -o 2
//
// η ∈ [0.8,0.9]
// mpirun -np 6 ./pthermal_compliance-filter -epsilon 0.01 -alpha 0.05 -beta 10.0 -r 4 -obs 5 -no-simp -theta 0.7 -mi 50


// mpirun -np 6 ./pthermal_compliance-filter -epsilon 0.01 -alpha 0.01 -beta 5.0 -r 5 -o 2 -bs 10 -theta 2.0 -mi 100 -mf 0.4 -no-simp  -paraview
// mpirun -np 6 ./pthermal_compliance-filter -epsilon 0.01 -alpha 0.01 -beta 5.0 -r 4 -o 2 -bs 1 -theta 2.0 -mi 100 -mf 0.4 -paraview
// mpirun -np 6 ./pthermal_compliance-filter -epsilon 0.01 -alpha 0.01 -beta 5.0 -r 4 -o 2 -bs 5 -theta 1.0 -mi 100 -mf 0.4

// run on ruby on 8 nodes
// srun -np 256 ./pthermal_compliance-filter -epsilon 0.01 -alpha 0.01 -beta 5.0 -r 5 -o 2 -bs 5 -theta 1.0 -mi 100 -mf 0.4 -paraview
// srun -np 256 ./pthermal_compliance-filter -epsilon 0.01 -alpha 0.05 -beta 5.0 -r 5 -o 2 -bs 5 -theta 1.0 -mi 100 -mf 0.4 -paraview
// srun -np 256 ./pthermal_compliance-filter -epsilon 0.01 -alpha 0.1 -beta 5.0 -r 5 -o 2 -bs 5 -theta 1.0 -mi 100 -mf 0.4 -paraview
// srun -np 256 ./pthermal_compliance-filter -epsilon 0.01 -alpha 0.1 -beta 5.0 -r 5 -o 2 -bs 5 -theta 2.0 -mi 100 -mf 0.4 -paraview


// Runs for the paper
// Determnistic design
// mpirun -np 8 pthermal_compliance-filter -epsilon 0.01 -beta 2.0 -r 5 -o 2 -bs 1 -mi 100 -mf 0.5 -alpha 0.1 -l1 0.2 -l2 0.2 -ms 1000000 -paraview -save-to-file -prob 0

// Stochastic design
// Adaptive sampling 
// mpirun -np 8 pthermal_compliance-filter -epsilon 0.01 -beta 2.0 -r 5 -o 2 -bs 5 -mi 100 -mf 0.5 -theta 2.5 -alpha 0.1 -l1 0.2 -l2 0.2 -ms 1000000 -paraview -prob 1

// Constant number of samples
// mpirun -np 8 pthermal_compliance-filter -epsilon 0.01 -beta 2.0 -r 5 -o 2 -bs 10 -mi 100 -mf 0.5 -theta -1.0 -alpha 0.1 -l1 0.2 -l2 0.2 -ms 1000000 -paraview -prob 1
// mpirun -np 8 pthermal_compliance-filter -epsilon 0.01 -beta 2.0 -r 5 -o 2 -bs 100 -mi 100 -mf 0.5 -theta -1.0 -alpha 0.1 -l1 0.2 -l2 0.2 -ms 1000000 -paraview -prob 1
// mpirun -np 8 pthermal_compliance-filter -epsilon 0.01 -beta 2.0 -r 5 -o 2 -bs 1000 -mi 100 -mf 0.5 -theta -1.0 -alpha 0.1 -l1 0.2 -l2 0.2 -ms 1000000 -paraview -prob 1
// mpirun -np 8 pthermal_compliance-filter -epsilon 0.01 -beta 2.0 -r 5 -o 2 -bs 10000 -mi 100 -mf 0.5 -theta -1.0 -alpha 0.1 -l1 0.2 -l2 0.2 -ms 1000000 -paraview -prob 1



// Restore from saved design and evaluate compliance on random loads
// mpirun -np 8 pthermal_compliance-filter -epsilon 0.01 -beta 2.0 -r 5 -o 2 -bs 100 -mi 1 -mf 0.5 -theta 2.0 -alpha 0.1 -l1 0.2 -l2 0.2 -ms 1000000 -paraview -restore -prob 1 -m output_design/mesh. -g output_design/rho.




//         min J(K) = <g,u>
//                            
//                        Γ_2           Γ_1            Γ_2
//               _ _ _ _ _ _ _ _ _ _ _________ _ _ _ _ _ _ _ _ _ _   
//              |         |         |         |         |         |  
//              |         |         |         |         |         |  
//              |---------|---------|---------|---------|---------|  
//              |         |         |         |         |         |  
//              |         |         |         |         |         |  
//      Γ_4-->  |---------|---------|---------|---------|---------|  <-- Γ_4
//              |         |         |         |         |         |  
//              |         |         |         |         |         |  
//              |---------|---------|---------|---------|---------|  
//              |         |         |         |         |         |  
//              |         |         |         |         |         |  
//               -------------------------------------------------
//                                        |̂                          
//                                       Γ_3                                   
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
#include "../solvers/pde_solvers.hpp"
#include "../../../spde/spde_solver.hpp"

bool random_seed = false;

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

enum prob_type
{
   deterministic,    
   stochastic     
};
prob_type prob;

enum geom_type
{
   square,    
   sphere     
};
geom_type geom;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();   
   // 1. Parse command-line options.
   int ref_levels = 2;
   int order = 2;
   bool visualization = true;
   bool mirror = false;
   double alpha = 1.0;
   double beta = 1.0;
   double epsilon = 1.0;
   double mass_fraction = 0.4;
   int max_it = 1e2;
   double tol_rho = 1.0;
   // double tol_rho = 5e-2;
   double tol_lambda = 0.0; // Don't exit until budget used up
   // double tol_lambda = 1e-3;
   double K_max = 1.0;
   double K_min = 1e-3;
   int batch_size_min = 2;
   double theta = 0.5;
   int max_cumulative_samples = 1e5;
   double l1 = 0.1, l2 = 0.1, l3 = 1.0;
   double e1 = 0.0, e2 = 0.0, e3 = 0.0;
   bool paraview = false;
   int iprob = 0;
   int igeom = 0;

   // save design to a file
   bool save_to_file = false;
   // restore precomputed design from file
   bool restore_from_file = false;
   const char *saved_meshfile = "output_design/mesh.";
   const char *saved_solfile = "output_design/rho.";

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&alpha, "-alpha", "--alpha-step-length",
                  "Step length for gradient descent.");
   args.AddOption(&beta, "-beta", "--beta-step-length",
                  "Step length for λ"); 
   args.AddOption(&epsilon, "-epsilon", "--epsilon-thickness",
                  "epsilon phase field thickness");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&max_cumulative_samples, "-ms", "--max-samples",
                  "Maximum number of cumulative samples.");
   args.AddOption(&l1, "-l1", "--l1",
                  "Correlation length x");
   args.AddOption(&l2, "-l2", "--l2",
                  "Correlation length y");             
   args.AddOption(&l3, "-l3", "--l3",
                  "Correlation length z");                
   args.AddOption(&e1, "-e1", "--e1",
                  "Rotation angle in x direction");
   args.AddOption(&e2, "-e2", "--e2",
                  "Rotation angle in y direction");             
   args.AddOption(&e3, "-e3", "--e3",
                  "Rotation angle in z direction");                               
   args.AddOption(&tol_rho, "-tr", "--tol_rho",
                  "Exit tolerance for ρ ");     
   args.AddOption(&tol_lambda, "-tl", "--tol_lambda",
                  "Exit tolerance for λ");                                 
   args.AddOption(&mass_fraction, "-mf", "--mass-fraction",
                  "Mass fraction for diffusion coefficient.");
   args.AddOption(&K_max, "-Kmax", "--K-max",
                  "Maximum of diffusion diffusion coefficient.");
   args.AddOption(&K_min, "-Kmin", "--K-min",
                  "Minimum of diffusion diffusion coefficient.");
   args.AddOption(&iprob, "-prob", "--problem", "Problem type"
                  " 0: deterministic, 1: stochastic ");
   args.AddOption(&igeom, "-g", "--geom", "Geometry type"
                  "0: square, 1: sphere");
   args.AddOption(&batch_size_min, "-bs", "--batch-size",
                  "batch size for stochastic gradient descent.");     
   args.AddOption(&theta, "-theta", "--theta-sampling-ratio",
                  "Sampling ratio theta");       
   args.AddOption(&random_seed, "-rs", "--random-seed", "-no-rs",
                  "--no-random-seed",
                  "Enable or disable random seed.");                                                     
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&mirror, "-mirror", "--mirror", "-no-mirror",
                  "--no-mirror",
                  "Enable or disable symmetric optimization.");

   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable paraview export.");       
   args.AddOption(&save_to_file, "-save-to-file", "--save-to-file", "-no-save",
                  "--no-save",
                  "Enable or disable saving design to a file.");                       
   args.AddOption(&restore_from_file, "-restore", "--restore", "-no-restore",
                  "--no-restore",
                  "Enable or disable restore from file.");                         
   args.AddOption(&saved_meshfile, "-m", "--mesh",
                  "Load precomputed design mesh.");
   args.AddOption(&saved_solfile, "-g", "--gf",
                  "Load precomputed design GridFunction.");                  
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

   MFEM_VERIFY((iprob == 0 || iprob == 1), "Wrong choice of problem kind");
   prob = (prob_type)iprob;

   MFEM_VERIFY((igeom == 0 || igeom == 1), "Wrong choice of geometry kind");
   geom = (geom_type)igeom;

   int batch_size = batch_size_min;
   if (theta < 0.0) { theta = std::numeric_limits<double>::infinity(); }

   ostringstream csv_file_name;
   csv_file_name << "conv_order" << order << "_GD_alpha_" << alpha 
                 << "_theta_" << theta << "_l1_" << l1 << "_l2_" << l2 << ".csv";
   ofstream conv(csv_file_name.str().c_str());
   if (myid == 0)
   {
      conv << "Step,  Inner Step,    Sample Size, Cum. Sample Size,"
           << " Compliance,    Mass Fraction, Norm of reduced grad,"
           << " Stationarity Error,"
           << " Lambda " << endl;
   } 

   Mesh *mesh;
   if (geom == geom_type::square)
   {
      if (mirror)
      {
         mesh = new Mesh(Mesh::MakeCartesian2D(3,6,mfem::Element::Type::QUADRILATERAL,true,0.5,1.0));

      }
      else
      {
         mesh = new Mesh(Mesh::MakeCartesian2D(7,7,mfem::Element::Type::QUADRILATERAL,true,1.0,1.0));
      }

   }
   else
   {
      const char *mesh_file = "spherical_surf.msh";
      mesh = new Mesh(mesh_file,1,1);
   }

   int dim = mesh->Dimension();

   if (geom == geom_type::square)
   {
      for (int i = 0; i<mesh->GetNBE(); i++)
      {
         Element * be = mesh->GetBdrElement(i);
         Array<int> vertices;
         be->GetVertices(vertices);

         double * coords1 = mesh->GetVertex(vertices[0]);
         double * coords2 = mesh->GetVertex(vertices[1]);

         Vector center(2);
         center(0) = 0.5*(coords1[0] + coords2[0]);
         center(1) = 0.5*(coords1[1] + coords2[1]);

         if (mirror)
         {
            if (abs(center(1) - 1.0) < 1e-10 && center(0)>=1.0/3.0)
            {
               // top edge on the right
               be->SetAttribute(1);
            }
            else if (abs(center(1) - 1.0) < 1e-10)
            {
               // the rest of top edge
               be->SetAttribute(2);
            }
            else if (abs(center(0)) < 1e-10)
            {
               // left edge
               be->SetAttribute(3);
            }
            else if (abs(center(0)-0.5) < 1e-10)
            {
               // right edge
               be->SetAttribute(4);
            }
            else
            {
               // bottom edge
               be->SetAttribute(3);
            }
         }
         else
         {
            if (abs(center(1) - 1.0) < 1e-10 && abs(center(0)-0.5) < 1e-10)
            {
               // middle of the top edge
               be->SetAttribute(1);
            }
            else if (abs(center(1) - 1.0) < 1e-10)
            {
               // the rest of top edge
               be->SetAttribute(2);

            }
            else if (abs(center(0)) < 1e-10 || abs(center(0)-1.0) < 1e-10)
            {
               // left and right edge
               be->SetAttribute(4);
            }
            else
            {
               // bottom edge
               be->SetAttribute(3);
            }
         }
      }
      mesh->SetAttributes();
   }

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   mesh->Clear();
   delete mesh;

   // 5. Define the vector finite element spaces representing the state variable u,
   //    adjoint variable p, and the control variable f.
   H1_FECollection state_fec(order, dim,BasisType::Positive); // space for u
   H1_FECollection filter_fec(order-1, dim,BasisType::Positive); // space for ρ̃  
   L2_FECollection control_fec(order-1, dim,BasisType::Positive); // space for ρ   
   ParFiniteElementSpace state_fes(&pmesh, &state_fec);
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

   // Setup SPDE for random load generation
   spde::Boundary bc;
   spde::Boundary bc1;
   if (mirror)
   {
      bc1.AddHomogeneousBoundaryCondition(4,spde::BoundaryType::kDirichlet);
   }

   if (Mpi::Root()) 
   {
     bc.PrintInfo();
     bc.VerifyDefinedBoundaries(pmesh);

     bc1.PrintInfo();
     bc1.VerifyDefinedBoundaries(pmesh);
   }
   double nu = 1.0;
   spde::SPDESolver random_load_solver(nu, bc, &state_fes, l1, l2,l3, e1,e2,e3);
   spde::SPDESolver random_load_solver1(nu, bc1, &state_fes, l1, l2,l3, e1,e2,e3);
   random_load_solver.SetPrintLevel(0);
   random_load_solver1.SetPrintLevel(0);

   ParGridFunction load_gf(&state_fes); 
   ParGridFunction load_gf1(&state_fes); 
   random_load_solver.SetupRandomFieldGenerator(myid+1);
   random_load_solver.GenerateRandomField(load_gf);

   random_load_solver1.SetupRandomFieldGenerator(myid+3000);
   random_load_solver1.GenerateRandomField(load_gf1);


   GridFunctionCoefficient load_cf(&load_gf);
   GridFunctionCoefficient load_cf1(&load_gf1);

   // 7. Set the initial guess for f and the boundary conditions for u.
   ParGridFunction u(&state_fes);
   ParGridFunction rho(&control_fes);
   ParGridFunction rho_old(&control_fes);
   ParGridFunction rho_filter(&filter_fes);
   u = 0.0;
   rho_filter = 0.0;
   rho = 0.5;
   rho_old = rho;
   // 8. Set up the linear form b(.) for the state and adjoint equations.
   int maxat = pmesh.bdr_attributes.Max();
   Array<int> ess_bdr(maxat);
   Array<int> ess_bdr1(maxat);
   ess_bdr = 0;
   ess_bdr1 = 0;
   if (maxat > 0)
   {
      if (geom == geom_type::square)
      {
         ess_bdr[0] = 1;
         ess_bdr1[0] = 1;
         ess_bdr1[3] = 1;
      }  
      else
      {
         ess_bdr = 1; // 
      }    
   }
   ConstantCoefficient one(1.0);
   DiffusionSolver * PoissonSolver = new DiffusionSolver();
   PoissonSolver->SetMesh(&pmesh);
   PoissonSolver->SetOrder(state_fec.GetOrder());
   PoissonSolver->SetupFEM();

   DiffusionSolver * PoissonSolver1 = new DiffusionSolver();
   PoissonSolver1->SetMesh(&pmesh);
   PoissonSolver1->SetOrder(state_fec.GetOrder());
   PoissonSolver1->SetupFEM();

   int seed = (random_seed) ? rand()%100 + myid : myid;
   
   PoissonSolver->SetRHSCoefficient(&load_cf);
   PoissonSolver->SetEssentialBoundary(ess_bdr);

   PoissonSolver1->SetRHSCoefficient(&load_cf1);
   PoissonSolver1->SetEssentialBoundary(ess_bdr1);

   ConstantCoefficient eps2_cf(epsilon*epsilon);
   DiffusionSolver * FilterSolver = new DiffusionSolver();
   FilterSolver->SetMesh(&pmesh);
   FilterSolver->SetOrder(filter_fec.GetOrder());
   FilterSolver->SetDiffusionCoefficient(&eps2_cf);
   FilterSolver->SetMassCoefficient(&one);

   Array<int> ess_bdr_filter;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr_filter.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr_filter = 0;
   }
   FilterSolver->SetEssentialBoundary(ess_bdr_filter);
   FilterSolver->SetupFEM();

   // 9. Define the gradient function
   ParGridFunction w(&control_fes);
   ParGridFunction avg_w(&control_fes);
   ParGridFunction stationarity(&control_fes);
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
   socketstream sout_u,sout_rho, sout_rho_filter;
   if (visualization)
   {
      sout_u.open(vishost, visport);
      sout_rho.open(vishost, visport);
      sout_rho_filter.open(vishost, visport);
      sout_u.precision(8);
      sout_rho.precision(8);
      sout_rho_filter.precision(8);
   }

   ParaViewDataCollection * paraview_dc = nullptr;
   
   if (paraview)
   {
      ostringstream paraview_file_name;
      paraview_file_name << "Thermal_compliance_alpha_" << alpha 
                         << "_theta_" << theta << "_l1_" << l1 << "_l2_" << l2
                         << "_e1_" << e1 << "_bs_" << batch_size_min;
      paraview_dc = new ParaViewDataCollection(paraview_file_name.str(), &pmesh);
      paraview_dc->SetPrefixPath("ParaView");
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      // paraview_dc->RegisterField("soln",&u);
      // paraview_dc->RegisterField("dens",&rho);
      paraview_dc->RegisterField("dens_filter",&rho_filter);
   }
   // 12. AL iterations
   int step = 0;
   bool first_iteration = true;
   int cumulative_samples = 0;
   double lambda = 0.0;
   Coefficient * K_cf = nullptr;
   Coefficient * rhs_cf = nullptr;
   for (int k = 1; k <= max_it; k++)
   {
      // A. Form state equation
      if (k > 1) { tol_rho *= (double (k-1))/(double (k)) ; }
      double ratio_avg = 0.0;
      for (int l = 1; l <= max_it; l++)
      {
         step++;
         cumulative_samples += batch_size;
         if (cumulative_samples > max_cumulative_samples) { break; }

         if (myid == 0)
         {
            cout << "\nStep = " << l << endl;
            cout << "Batch Size   = " << batch_size << endl;
            cout << "Cum. Samples = " << cumulative_samples << endl;
         }
         // Step 2 -  Filter Solve
         // Solve (ϵ^2 ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)  
         GridFunctionCoefficient rho_cf(&rho);
         FilterSolver->SetRHSCoefficient(&rho_cf);
         FilterSolver->Solve();
         rho_filter = *FilterSolver->GetFEMSolution();

         // ------------------------------------------------------------------
         // Step 3 - State Solve

         avg_w = 0.0;
         double avg_w_norm = 0.;
         double avg_compliance = 0.;

         double mf = vol_form(rho)/domain_volume;
         K_cf = new DiffusionCoefficient(rho_filter,K_min,K_max);
         rhs_cf = new GradientRHSCoefficient(u,rho_filter,K_min, K_max);
         PoissonSolver->SetDiffusionCoefficient(K_cf);
         PoissonSolver1->SetDiffusionCoefficient(K_cf);
         FilterSolver->SetRHSCoefficient(rhs_cf);
         for (int ib = 0; ib<batch_size; ib++)
         {
            if (prob == prob_type::stochastic)
            {
               random_load_solver.GenerateRandomField(load_gf);
               if (mirror) random_load_solver1.GenerateRandomField(load_gf1);
            }
            else
            {
               load_gf = 1.0;
            }
            PoissonSolver->Solve();
            u = *PoissonSolver->GetFEMSolution();
            if (mirror)
            {
               PoissonSolver1->Solve();
               u += *PoissonSolver1->GetFEMSolution();
               u/=2.0;
            }

            // ------------------------------------------------------------------
            // Step 4 - Adjoint Solve
            FilterSolver->Solve();
            w_filter = *FilterSolver->GetFEMSolution();

            GridFunctionCoefficient w_cf(&w_filter);
            w.ProjectCoefficient(w_cf); // This might need to change to L2-projection
            // ------------------------------------------------------------------
            // step 6-update  ρ 
            w -= lambda;
            w += beta * (mf - mass_fraction)/domain_volume;
            avg_w += w;
            double w_norm = w.ComputeL2Error(zero);
            avg_w_norm += w_norm*w_norm;
            avg_compliance += (*(PoissonSolver->GetParLinearForm()))(u);
         } // end of loop through batch samples

         avg_w_norm /= (double)batch_size;  
         avg_w /= (double)batch_size;

         avg_compliance /= (double)batch_size;  

         double norm_avg_w = pow(avg_w.ComputeL2Error(zero),2);
         double denom = batch_size == 1 ? batch_size : batch_size-1;
         double variance = (avg_w_norm - norm_avg_w)/denom;

         double eta = 1e-6;
         stationarity = 0.0;
         stationarity += rho;
         stationarity.Add(-eta, avg_w);
         stationarity += eta * beta * (mf - mass_fraction)/domain_volume;
         // project
         for (int i = 0; i < stationarity.Size(); i++)
         {
            if (stationarity[i] > 1.0) 
            {
               stationarity[i] = 1.0;
            }
            else if (stationarity[i] < 0.0)
            {
               stationarity[i] = 0.0;
            }
            else
            { // do nothing
            }
         }
         stationarity -= rho;
         stationarity /= eta;
         double stationarity_norm = stationarity.ComputeL2Error(zero);

         rho.Add(-alpha, avg_w);
         // project
         for (int i = 0; i < rho.Size(); i++)
         {
            if (rho[i] > 1.0) 
            {
               rho[i] = 1.0;
            }
            else if (rho[i] < 0.0)
            {
               rho[i] = 0.0;
            }
            else
            { // do nothing
            }
         }

         GridFunctionCoefficient tmp(&rho_old);
         double norm_rho = rho.ComputeL2Error(tmp)/alpha;
         rho_old = rho;
         
         if (myid == 0)
         {
            mfem::out << "norm of reduced gradient = " << norm_rho << endl;
            mfem::out << "avg_compliance = " << avg_compliance << endl;
            mfem::out << "variance = " << variance << std::endl;
            mfem::out << "stationarity = " << stationarity_norm << std::endl;
         }
         if (norm_rho < tol_rho) { break; }

         double ratio = sqrt(abs(variance)) / norm_rho ;
         if (myid == 0)
         {
            mfem::out << "ratio = " << ratio << std::endl;
            conv << step << ",   "
                 << l << ",   " 
                 << batch_size << ",   " 
                 << cumulative_samples << ",   " 
                 << avg_compliance <<  ",   " 
                 << mf << ",   "
                 << norm_rho << ",   "
                 << stationarity_norm << ",   "
                 << lambda << endl;
         }
         MFEM_VERIFY(IsFinite(ratio), "ratio not finite");
         if (myid == 0)
         {
            mfem::out << "ratio_avg = " << ratio_avg << std::endl;
         }
         if (ratio > theta && !first_iteration)
         {
            batch_size = max((int)(pow(ratio / theta,2) * batch_size),batch_size_min); 
         }
         else if (ratio < 0.1 * theta && !first_iteration)
         {
            batch_size = max((int)(pow(ratio / theta,2) * batch_size),batch_size_min); 
         }
         first_iteration = false;

         if (visualization)
         {

            sout_u << "parallel " << num_procs << " " << myid << "\n";
            sout_u << "solution\n" << pmesh << u
                  << "window_title 'State u'" << flush;

            sout_rho << "parallel " << num_procs << " " << myid << "\n";
            sout_rho << "solution\n" << pmesh << rho
                  << "window_title 'Control ρ '" << flush;

            sout_rho_filter << "parallel " << num_procs << " " << myid << "\n";
            sout_rho_filter << "solution\n" << pmesh << rho_filter
                  << "window_title 'Control ρ filter '" << flush;                  

         }
         if (paraview)
         {
            paraview_dc->SetCycle(step);
            paraview_dc->SetTime((double)step);
            paraview_dc->Save();
         }
         delete K_cf;
         delete rhs_cf;
      }
      // λ <- λ - β (∫_Ω K dx - V⋅ vol(\Omega))
      double mass = vol_form(rho);
      if (myid == 0)
      {
         mfem::out << "mass_fraction = " << mass / domain_volume << endl;
      }

      double lambda_inc = mass/domain_volume - mass_fraction;

      lambda -= beta*lambda_inc;
      if (myid == 0)
      {
         mfem::out << "lambda_inc = " << lambda_inc << endl;
         mfem::out << "lambda = " << lambda << endl;
      }

      if (visualization)
      {
         sout_u << "parallel " << num_procs << " " << myid << "\n";
         sout_u << "solution\n" << pmesh << u
               << "window_title 'State u'" << flush;

         sout_rho << "parallel " << num_procs << " " << myid << "\n";
         sout_rho << "solution\n" << pmesh << rho
                << "window_title 'Control ρ '" << flush;
     
      }
      if (paraview)
      {
         paraview_dc->SetCycle(step);
         paraview_dc->SetTime((double)step);
         paraview_dc->Save();
      }

      if (abs(lambda_inc) < tol_lambda) { break; }
      if (cumulative_samples > max_cumulative_samples) { break; }

   }

   if (paraview)
   {
      delete paraview_dc;
   }

   return 0;
}