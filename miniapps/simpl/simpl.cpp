#include "mfem.hpp"
#include "topopt_problems.hpp"
#include "logger.hpp"

using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
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
   real_t tol_stationary_abs = 1e-04;
   real_t eps_stationarity = 1e-04;
   bool use_bregman_stationary = true;
   real_t tol_obj_diff_rel = 1e-06;
   real_t tol_obj_diff_abs = 1e-06;
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
      if (Mpi::Root())
      {
         args.PrintUsage(out);
      }
      return 1;
   }
   std::stringstream filename;
   filename << "SiMPL-" << (use_bregman_backtrack?"B-":"A-");

   Array2D<int> ess_bdr_state;
   Array<int> ess_bdr_filter;
   real_t tot_vol;
   std::unique_ptr<ParMesh> mesh((ParMesh*)GetTopoptMesh(
                                    prob, filename,
                                    r_min, tot_vol, min_vol, max_vol,
                                    E, nu, ess_bdr_state, ess_bdr_filter,
                                    ser_ref_levels, par_ref_levels));
   filename << "-" << ser_ref_levels + par_ref_levels;
   const real_t lambda = E*nu/((1+nu)*(1-2*nu));
   const real_t mu = E/(2*(1+nu));
   const int dim = mesh->SpaceDimension();
   if (Mpi::Root())
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
   ParGridFunction density_gf(&fes_control);
   density_gf = 0.0; // this is only for visualization
   if (Mpi::Root())
   {
      out << "done" << std::endl;
   }

   // elasticity coefficients
   ConstantCoefficient zero_cf(0.0);
   MappedGFCoefficient simp_cf(
      filter_gf, [exponent, rho0](const real_t x)
   {
      return simp(x, exponent, rho0);
   });
   MappedGFCoefficient der_simp_cf(
      filter_gf, [exponent, rho0](const real_t x)
   {
      return der_simp(x, exponent, rho0);
   });
   ConstantCoefficient lambda_cf(lambda), mu_cf(mu);
   ProductCoefficient lambda_simp_cf(lambda, simp_cf), mu_simp_cf(mu, simp_cf);

   if (Mpi::Root())
   {
      out << "Creating problems ... " << std::flush;
   }
   // Density
   FermiDiracEntropy entropy;
   MappedGFCoefficient density_cf = entropy.GetBackwardCoeff(control_gf);
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

   // Backtracking related stuffs
   MappedPairedGFCoefficient bregman_diff_old
      = entropy.GetBregman_dual(control_old_gf, control_gf);
   MappedPairedGFCoefficient bregman_diff_eps
      = entropy.GetBregman_dual(control_eps_gf, control_gf);
   MappedPairedGFCoefficient diff_density_cf(
      control_gf, control_old_gf, [](const real_t x, const real_t y)
   {
      return sigmoid(x)-sigmoid(y);
   });
   MappedGFCoefficient density_eps_dual_cf = entropy.GetBackwardCoeff(
                                                control_eps_gf);
   GridFunctionCoefficient density_eps_primal_cf(&control_eps_gf);
   GridFunctionCoefficient grad_cf(&grad_gf);
   ParGridFunction zero_gf(&fes_control);
   zero_gf = 0.0;
   ParLinearForm diff_density_form(&fes_control);
   diff_density_form.AddDomainIntegrator(new DomainLFIntegrator(diff_density_cf));

   real_t step_size = 1.0;
   GLVis glvis("localhost", 19916, true);
   const char keys[] = "Rjmml****************";
   glvis.Append(control_gf, "control variable", keys);
   glvis.Append(density_gf, "design density", keys);
   glvis.Append(filter_gf, "filtered density", keys);
   glvis.Append(state_gf, "displacement magnitude", keys);

   real_t stationarity0, obj0,
          stationarity_error_L2, stationarity_error_bregman, stationarity_error,
          curr_vol,
          objval(infinity()), old_objval(infinity()), succ_obj_diff(infinity());
   int tot_reeval(0), num_reeval(0);
   int it_md;
   TableLogger logger;
   logger.Append("iteration", it_md);
   logger.Append("volume", curr_vol);
   logger.Append("obj", objval);
   logger.Append("step-size", step_size);
   logger.Append("num-reeval", num_reeval);
   logger.Append("succ-objdiff", succ_obj_diff);
   logger.Append("stnrty-L2", stationarity_error_L2);
   logger.Append("stnrty-B", stationarity_error_bregman);
   logger.SaveWhenPrint(filename.str());
   mfem::ParaViewDataCollection paraview_dc(filename.str(), mesh.get());
   if (use_paraview)
   {
      if (paraview_dc.Error())
      {
         use_paraview=false;
      }
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order_state);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.RegisterField("displacement", &state_gf);
      paraview_dc.RegisterField("density", &density_gf);
      paraview_dc.RegisterField("filtered_density", &filter_gf);
      paraview_dc.Save();
   }

   grad_gf = 0.0;
   for (it_md = 0; it_md<max_it; it_md++)
   {
      if (Mpi::Root()) { out << "Mirror Descent Step " << it_md << std::endl; }
      if (it_md > 1)
      {
         diff_density_form.Assemble();
         grad_old_gf -= grad_gf;
         real_t grad_diffrho = -InnerProduct(MPI_COMM_WORLD,
                                             diff_density_form, grad_old_gf);
         control_old_gf -= control_gf;
         real_t psi_diffrho = -InnerProduct(MPI_COMM_WORLD,
                                            diff_density_form, control_old_gf);
         step_size = std::fabs(psi_diffrho / grad_diffrho);
         if (Mpi::Root())
         {
            out << "   Step size = " << step_size
                << " = | " << psi_diffrho << " / " << grad_diffrho << " |" << std::endl;
         }
      }

      control_old_gf = control_gf;
      old_objval = objval;
      for (num_reeval=0; num_reeval < max_it_backtrack; num_reeval++)
      {
         add(control_old_gf, -step_size, grad_gf, control_gf);
         objval = optproblem.Eval();
         diff_density_form.Assemble();
         real_t grad_diffrho = InnerProduct(MPI_COMM_WORLD,
                                            diff_density_form, grad_gf);
         real_t bregman_diff = zero_gf.ComputeL1Error(bregman_diff_old);
         real_t target_objval = use_bregman_backtrack
                                ? old_objval + grad_diffrho + bregman_diff / step_size
                                : old_objval + c1*grad_diffrho;
         if (Mpi::Root())
         {
            out << "      New    Objective  : " << objval << std::endl;
            out << "      Target Objective  : " << target_objval;
            if (use_bregman_backtrack)
            {
               out << " ( " << old_objval << " + " << grad_diffrho
                   << " + " << bregman_diff << " / " << step_size << " )" << std::endl;
            }
            else
            {
               out << " ( " << old_objval << " + " << c1 << " * " << grad_diffrho << " )" <<
                   std::endl;
            }
         }
         if (objval < target_objval)
         {
            if (Mpi::Root())
            {
               out << "   Backtracking terminated after "
                   << num_reeval << " re-eval" << std::endl;
            }
            tot_reeval += num_reeval;
            break;
         }
         if (Mpi::Root())
         {
            out << "   --Attempt failed" << std::endl;
         }
         step_size *= 0.5;
      }
      succ_obj_diff = old_objval - objval;
      grad_old_gf = grad_gf;
      curr_vol = optproblem.GetCurrentVolume();

      density_gf.ProjectCoefficient(density_cf);
      if (use_glvis) { glvis.Update(); }
      if (use_paraview)
      {
         if (!paraview_dc.Error()) { paraview_dc.Save(); }
         else {use_paraview = false;}
      }

      optproblem.UpdateGradient();

      add(control_gf, -eps_stationarity, grad_gf, control_eps_gf);
      density.ApplyVolumeProjection(control_eps_gf, true);
      stationarity_error_bregman = std::sqrt(zero_gf.ComputeL1Error(
                                                bregman_diff_eps))/eps_stationarity;

      add(density_gf, -eps_stationarity, grad_gf, control_eps_gf);
      density.ApplyVolumeProjection(control_eps_gf, false);
      stationarity_error_L2 = density_gf.ComputeL2Error(
                                 density_eps_primal_cf)/eps_stationarity;

      stationarity_error = use_bregman_stationary
                           ? stationarity_error_bregman : stationarity_error_L2;
      logger.Print(true);
      if (Mpi::Root())
      {
         out << "--------------------------------------------" << std::endl;
      }
      if (it_md == 0)
      {
         stationarity0 = stationarity_error;
         obj0 = objval;
      }
      if ((stationarity_error < tol_stationary_abs ||
           stationarity_error < tol_stationary_rel*stationarity0)
          && (std::abs(objval - old_objval) < tol_obj_diff_abs ||
              std::abs(objval - old_objval) < tol_obj_diff_rel*std::fabs(obj0)))
      {
         break;
      }
   }
   if (Mpi::Root())
   {
      out << filename.str() << " terminated after " << it_md
          << " with " << tot_reeval << " re-eval" << std::endl;
   }
   logger.CloseFile();
   return 0;
}
