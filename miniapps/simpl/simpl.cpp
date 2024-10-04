#include "mfem.hpp"
#include "topopt_problems.hpp"

using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   TopoptProblem prob = Cantilever2;
   int ser_ref_levels = 2;
   int par_ref_levels = 4;
   int order_control = 0;
   int order_filter = 1;
   int order_state = 1;
   bool use_glvis = true;
   bool use_paraview = true;

   real_t exponent = 3.0;
   real_t rho0 = 1e-06;
   // problem dependent parmeters.
   // Set to -1 to use default value
   real_t E = -1.0;
   real_t nu = -1.0;
   real_t r_min = -1.0;
   real_t max_vol = -1.0;
   real_t min_vol = -1.0;

   // Stopping-criteria related
   int max_it = 300;
   real_t tol_stationary_rel = 1e-04;
   real_t tol_stationary_abs = 1e-06;
   bool use_bregman_stationary = true;
   real_t tol_obj_diff_rel = 1e-03;
   real_t tol_obj_diff_abs = 1e-03;
   // backtracking related
   int max_it_backtrack = 300;
   bool use_bregman_backtrack = true;
   real_t c1 = 1e-04;

   OptionsParser args(argc, argv);
   // problem
   args.AddOption((int*)&prob, "-p", "--problem",
                  "Topology optimization problem. See, topopt_problems.hpp");
   // FE-related options
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "The number of uniform mesh refinement in serial");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "The number of uniform mesh refinement in parallel");
   args.AddOption(&order_control, "-oc", "--order-control",
                  "Finite element order (polynomial degree) for control variable.");
   args.AddOption(&order_filter, "-of", "--order-filter",
                  "Finite element order (polynomial degree) for filter variable.");
   args.AddOption(&order_state, "-os", "--order-state",
                  "Finite element order (polynomial degree) for state variable.");
   // problem dependent options
   args.AddOption(&r_min, "-fr", "--filter-radius",
                  "Filter radius for Helmholtz filter. eps = r_min / 2 / sqrt(3)");
   args.AddOption(&max_vol, "-maxv", "--max-volume",
                  "Maximum volume");
   args.AddOption(&min_vol, "-minv", "--min-volume",
                  "Miminum volume");
   args.AddOption(&exponent, "-exp", "--simp-exponent",
                  "Penalty parameter for SIMP method");
   args.AddOption(&rho0, "-rho0", "--min-density",
                  "Minimum density for SIMP method");
   args.AddOption(&E, "-E", "--youngs-modulus",
                  "Young's modulus E");
   args.AddOption(&nu, "-nu", "--poisson-ratio",
                  "Poinsson ration nu");

   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iteration for Mirror Descent Step");
   args.AddOption(&max_it_backtrack, "-mi-back", "--max-it-backtrack",
                  "Maximum number of iteration for backtracking");
   args.AddOption(&tol_stationary_rel, "-rtol", "--rel-tol",
                  "Tolerance for relative stationarity error");
   args.AddOption(&tol_stationary_abs, "-atol", "--abs-tol",
                  "Tolerance for absolute stationarity error");
   args.AddOption(&tol_obj_diff_rel, "-rtol-obj", "--rel-tol-obj",
                  "Tolerance for relative successive objective difference");
   args.AddOption(&tol_obj_diff_abs, "-atol-obj", "--abs-tol-obj",
                  "Tolerance for absolute successive objective difference");
   args.AddOption(&use_bregman_backtrack, "-bb", "--bregman-backtrack", "-ab",
                  "--armijo-backtrack",
                  "Option to choose Bregman backtracking algorithm or Armijo backtracking algorithm");
   args.AddOption(&use_bregman_stationary, "-bs", "--bregman-stationarity", "-L2",
                  "--L2-stationarity",
                  "Option to choose Bregman stationarity or L2 stationarity for stopping criteria");

   // visualization related options
   args.AddOption(&use_glvis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&use_paraview, "-pv", "--paraview", "-no-pv",
                  "--no-paraview",
                  "Enable or disable paraview export.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(out);
      }
      return 1;
   }

   Array2D<int> ess_bdr_state;
   Array<int> ess_bdr_filter;
   real_t tot_vol;
   std::unique_ptr<ParMesh> mesh((ParMesh*)GetTopoptMesh(
                                    prob, r_min, tot_vol, min_vol, max_vol,
                                    E, nu, ess_bdr_state, ess_bdr_filter,
                                    ser_ref_levels, par_ref_levels));
   // const real_t lambda = E*nu/((1+nu)*(1-2*nu));
   // const real_t mu = E/(2*(1+nu));
   const real_t lambda = 1.0;
   const real_t mu = 1.0;
   const int dim = mesh->SpaceDimension();
   if (myid == 0)
   {
      args.PrintOptions(out);
   }

   if (Mpi::Root())
   {
      out << "   --lambda " << lambda << std::endl;
      out << "   --mu " << mu << std::endl;
   }

   // Finite Element
   L2_FECollection fec_control(order_control, dim);
   H1_FECollection fec_filter(order_filter, dim);
   H1_FECollection fec_state(order_state, dim);

   ParFiniteElementSpace fes_control(mesh.get(), &fec_control);
   ParFiniteElementSpace fes_filter(mesh.get(), &fec_filter);
   ParFiniteElementSpace fes_state(mesh.get(), &fec_state, dim);

   HYPRE_BigInt num_elem = mesh->GetGlobalNE();
   HYPRE_BigInt control_ndof = fes_control.GlobalTrueVSize();
   HYPRE_BigInt state_ndof = fes_state.GlobalTrueVSize();
   HYPRE_BigInt filter_ndof = fes_filter.GlobalTrueVSize();
   if (Mpi::Root())
   {
      std::cout << "\tThe number of elements: " << num_elem << std::endl
                << "\tThe number of control unknowns: " << control_ndof << std::endl
                << "\tThe number of filter  unknowns: " << filter_ndof << std::endl
                << "\tThe number of state   unknowns: " << state_ndof << std::endl;
   }

   if (Mpi::Root())
   {
      out << "Creating gridfunctions ... " << std::flush;
   }
   ParGridFunction control_gf(&fes_control); control_gf = 0.0;
   ParGridFunction filter_gf(&fes_filter); filter_gf = 0.0;
   ParGridFunction state_gf(&fes_state); state_gf = 0.0;
   ParGridFunction grad_gf(&fes_control); grad_gf = 0.0;
   ParGridFunction grad_filter_gf(&fes_filter); grad_filter_gf = 0.0;
   std::unique_ptr<ParGridFunction> adj_state_gf;
   if (prob < 0)
   {
      adj_state_gf.reset(new ParGridFunction(&fes_state));
      *adj_state_gf = 0.0;
   }

   ParGridFunction control_old_gf(&fes_control);
   ParGridFunction control_eps_gf(&fes_control);
   ParGridFunction grad_old_gf(&fes_control);
   if (Mpi::Root())
   {
      out << "done" << std::endl;
   }

   // elasticity coefficients
   ConstantCoefficient zero_cf(0.0);
   MappedGFCoefficient simp_cf(filter_gf,
   new std::function<real_t(const real_t)>([exponent, rho0](const real_t x) {return simp(x, exponent, rho0);}));
   MappedGFCoefficient der_simp_cf(filter_gf,
   new std::function<real_t(const real_t)>([exponent, rho0](const real_t x) {return der_simp(x, exponent, rho0);}));
   ConstantCoefficient lambda_cf(lambda), mu_cf(mu);
   ProductCoefficient lambda_simp_cf(lambda, simp_cf), mu_simp_cf(mu, simp_cf);

   if (Mpi::Root())
   {
      out << "Creating problems ... " << std::flush;
   }
   // Density
   FermiDiracEntropy entropy;
   MappedGFCoefficient density_cf = entropy.GetBackwardCoeff(control_gf);
   MappedPairedGFCoefficient bregman_diff_old
      = entropy.GetBregman_dual(control_old_gf, control_gf);
   MappedPairedGFCoefficient bregman_diff_eps
      = entropy.GetBregman_dual(control_eps_gf, control_gf);
   DesignDensity density(fes_control, tot_vol, min_vol, max_vol, &entropy);

   // Filter
   HelmholtzFilter filter(fes_filter, ess_bdr_filter, r_min, true);
   filter.GetLinearForm()->AddDomainIntegrator(new DomainLFIntegrator(density_cf));
   filter.SetBStationary(false);
   StrainEnergyDensityCoefficient energy(lambda_cf, mu_cf, der_simp_cf, state_gf,
                                         adj_state_gf.get());
   filter.GetAdjLinearForm()->AddDomainIntegrator(new DomainLFIntegrator(energy));
   filter.SetAdjBStationary(false);

   // elasticity
   ElasticityProblem elasticity(fes_state, ess_bdr_state, lambda_simp_cf,
                                mu_simp_cf, prob < 0);
   elasticity.SetAStationary(false);
   SetupTopoptProblem(prob, elasticity, filter_gf, state_gf);
   DensityBasedTopOpt optproblem(density, control_gf, grad_gf,
                                 filter, filter_gf, grad_filter_gf,
                                 elasticity, state_gf);
   elasticity.SetBStationary(false);
   if (Mpi::Root())
   {
      out << "done" << std::endl;
   }

   real_t step_size = 1.0;
   real_t objval = infinity();
   real_t old_objval = infinity();
   grad_gf = 0.0;

   socketstream sout_filter, sout_state;
   if (use_glvis)
   {
      char vishost[] = "localhost";
      int visport = 19916;

      sout_filter.open(vishost, visport);
      if (!sout_filter)
      {
         use_glvis = false;
         if (Mpi::Root())
         {
            out << "Unable to connect to GLVis server at " << vishost << ':'
                << visport << std::endl;
            out << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         sout_filter.precision(8);
         // Plot magnitude of vector-valued momentum
         sout_filter << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() <<
                     "\n";
         sout_filter << "solution\n" << *mesh << filter_gf;
         sout_filter << "window_title 'filtered density'\n";
         sout_filter << "view 0 0\n";  // view from top
         sout_filter << "keys jlm\n";  // turn off perspective and light, show mesh
         sout_filter << std::flush;
         MPI_Barrier(mesh->GetComm());
      }
      sout_state.open(vishost, visport);
      if (!sout_state)
      {
         use_glvis = false;
         if (Mpi::Root())
         {
            out << "Unable to connect to GLVis server at " << vishost << ':'
                << visport << std::endl;
            out << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         sout_state.precision(8);
         // Plot magnitude of vector-valued momentum
         sout_state << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() <<
                    "\n";
         sout_state << "solution\n" << *mesh << state_gf;
         sout_state << "window_title 'state'\n";
         sout_state << "view 0 0\n";  // view from top
         sout_state << "keys jlm\n";  // turn off perspective and light, show mesh
         sout_state << std::flush;
         MPI_Barrier(mesh->GetComm());
      }
   }

   for (int k=-1; k<max_it; k++)
   {
      control_gf.Add(-step_size, grad_gf);
      old_objval = objval;
      objval = optproblem.Eval();
      if (Mpi::Root())
      {
         out << "\t\tNew Objective: " << objval << std::endl;
         out << "\t\tCurrent Volume: " << optproblem.GetCurrentVolume() << std::endl;
      }

      if (sout_filter.is_open())
      {
         sout_filter << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() <<
                     "\n";
         sout_filter << "solution\n" << *mesh << filter_gf;
         sout_filter << std::flush;
      }
      if (sout_state.is_open())
      {
         sout_state << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() <<
                    "\n";
         sout_state << "solution\n" << *mesh << state_gf;
         sout_state << std::flush;
      }
      optproblem.UpdateGradient();
      step_size += 1;

   }
   return 0;
}
