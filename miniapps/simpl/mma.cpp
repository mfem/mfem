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
   int vis_steps = 10;
   bool use_paraview = true;
   bool overwrite_paraview = false;
   real_t step_size = -1.0;

   real_t exponent = 3.0;
   real_t rho0 = 1e-06;
   // problem dependent parmeters.
   // Set to -1 to use default value
   real_t E = -1.0;
   real_t nu = -1.0;
   real_t r_min = -1.0;
   real_t max_vol = -1.0;
   real_t min_vol = -1.0;

   // Solid / Void material element attributes
   int solid_attr = 0;
   int void_attr = 0;
   real_t max_latent = 1e18;

   // Stopping-criteria related
   int max_it = 300;
   int min_it = 10;
   real_t tol_stationary_rel = 1e-04;
   real_t tol_stationary_abs = 1e-04;
   real_t eps_stationarity = 1e-04;
   bool use_bregman_stationary = true;
   real_t tol_obj_diff_rel = 5e-05;
   real_t tol_obj_diff_abs = 5e-05;
   real_t tol_kkt_rel = 2e-04;
   real_t tol_kkt_abs = 2e-05;
   // backtracking related
   int max_it_backtrack = 20;
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

   args.AddOption(&max_latent, "-maxl", "--max-latent",
                  "Maximum value for the latent variable to prevent overflow");

   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iteration for Mirror Descent Step");
   args.AddOption(&min_it, "-mini", "--min-it",
                  "Minimum number of iteration for Mirror Descent Step");
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
   args.AddOption(&step_size, "-a0", "--init-step",
                  "Initial step size");
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
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&use_paraview, "-pv", "--paraview", "-no-pv",
                  "--no-paraview",
                  "Enable or disable paraview export.");
   args.AddOption(&overwrite_paraview, "-po", "--paraview-overwrite", "-pn",
                  "--paraview-newiteration",
                  "Overwrites paraview file");
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
   filename << "MMA-";

   Array2D<int> ess_bdr_state;
   Array<int> ess_bdr_filter;
   real_t tot_vol;
   std::unique_ptr<ParMesh> mesh((ParMesh*)GetTopoptMesh(
                                    prob, filename,
                                    r_min, tot_vol, min_vol, max_vol,
                                    E, nu, ess_bdr_state, ess_bdr_filter,
                                    solid_attr, void_attr,
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

   if (Mpi::Root()) { out << "Creating gridfunctions ... " << std::flush; }
   ParGridFunction control_gf(&fes_control); control_gf = 0.0;
   ParGridFunction filter_gf(&fes_filter); filter_gf = 0.0;
   ParGridFunction state_gf(&fes_state); state_gf = 0.0;
   ParGridFunction grad_gf(&fes_control); grad_gf = 0.0;
   ParGridFunction grad_filter_gf(&fes_filter); grad_filter_gf = 0.0;

   // Filter Essential BDR
   Array<int> solid_bdr_filter(ess_bdr_filter);
   for (auto &isSolid:solid_bdr_filter) {isSolid = isSolid==1;};
   ConstantCoefficient one_cf(1.0);
   filter_gf.ProjectBdrCoefficient(one_cf, solid_bdr_filter);

   ParGridFunction control_old_gf(&fes_control);
   ParGridFunction control_eps_gf(&fes_control);
   ParGridFunction grad_old_gf(&fes_control);
   ParGridFunction density_gf(&fes_control);
   density_gf = 0.0; // this is only for visualization
   if (Mpi::Root()) { out << "done" << std::endl; }

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

   if (Mpi::Root()) { out << "Creating problems ... " << std::flush; }
   // Density
   PrimalEntropy entropy;
   control_gf = entropy.forward((min_vol ? min_vol : max_vol)/tot_vol);
   MappedGFCoefficient density_cf = entropy.GetBackwardCoeff(control_gf);
   DesignDensity density(fes_control, tot_vol, min_vol, max_vol, &entropy);
   density.SetVoidAttr(void_attr);
   density.SetSolidAttr(solid_attr);
   // Filter
   HelmholtzFilter filter(fes_filter, ess_bdr_filter, r_min, true);
   filter.GetLinearForm()->AddDomainIntegrator(new DomainLFIntegrator(density_cf));
   filter.SetBStationary(false);
   StrainEnergyDensityCoefficient energy(lambda_cf, mu_cf, der_simp_cf, state_gf,
                                         nullptr);
   filter.GetAdjLinearForm()->AddDomainIntegrator(new DomainLFIntegrator(energy));
   filter.SetAdjBStationary(false);
   if (prob == mfem::ForceInverter2)
   {
      ForceInverterInitialDesign(control_gf, &entropy);
   }
   density.ApplyVolumeProjection(control_gf, true);
   filter.Solve(filter_gf);

   // elasticity
   ElasticityProblem elasticity(fes_state, ess_bdr_state, lambda_simp_cf,
                                mu_simp_cf, prob < 0);
   elasticity.SetAStationary(false);
   SetupTopoptProblem(prob, filter, elasticity, filter_gf, state_gf);
   DensityBasedTopOpt optproblem(density, control_gf, grad_gf,
                                 filter, filter_gf, grad_filter_gf,
                                 elasticity, state_gf);
   if (elasticity.HasAdjoint()) {energy.SetAdjState(optproblem.GetAdjState());}
   if (Mpi::Root()) { out << "done" << std::endl; }

   // Backtracking related stuffs
   MappedPairedGFCoefficient bregman_diff_old
      = entropy.GetBregman_dual(control_old_gf, control_gf);
   MappedPairedGFCoefficient bregman_diff_eps
      = entropy.GetBregman_dual(control_eps_gf, control_gf);
   MappedPairedGFCoefficient diff_density_cf(
      control_gf, control_old_gf, [&entropy](const real_t x, const real_t y)
   {
      return entropy.backward(x)-entropy.backward(y);
   });
   MappedGFCoefficient density_eps_dual_cf = entropy.GetBackwardCoeff(
                                                control_eps_gf);
   real_t avg_grad;
   MappedPairedGFCoefficient KKT_cf(
      control_gf, grad_gf,
      [&avg_grad, &entropy](const real_t x, const real_t g)
   {
      real_t rho_k = entropy.backward(x);
      real_t lambda_k = avg_grad - g;
      // Definition -Brendan
      return std::max(0.0, lambda_k)*(1.0-rho_k)-std::min(0.0, lambda_k)*rho_k;
      // Definition -Thomas
      // return lambda_k
      //        - std::min(0.0, rho_k     + lambda_k)
      //        - std::max(0.0, rho_k - 1 + lambda_k);
   });
   ParBilinearForm mass(&fes_control);
   mass.AddDomainIntegrator(new MassIntegrator());
   mass.Assemble();
   mass.Finalize();
   ParGridFunction kkt_gf(&fes_control);
   std::unique_ptr<HypreParMatrix> Mass(mass.ParallelAssemble());

   GridFunctionCoefficient density_eps_primal_cf(&control_eps_gf);
   GridFunctionCoefficient grad_cf(&grad_gf);
   ParGridFunction zero_gf(&fes_control);
   zero_gf=1.0;
   ParGridFunction dv(&fes_control);
   Mass->Mult(zero_gf, dv);
   zero_gf = 0.0;
   ParLinearForm diff_density_form(&fes_control);
   diff_density_form.AddDomainIntegrator(new DomainLFIntegrator(diff_density_cf));

   std::unique_ptr<GLVis> glvis;
   if (use_glvis)
   {
      glvis.reset(new GLVis ("localhost", 19916, true));
      const char keys[] = "Rjmml****************";
      glvis->Append(control_gf, "control variable", keys);
      density_gf.ProjectCoefficient(density_cf);
      glvis->Append(density_gf, "design density", keys);
      glvis->Append(filter_gf, "filtered density", keys);
      glvis->Append(state_gf, "displacement magnitude", keys);
      glvis->Append(kkt_gf, "KKT", keys);
      if (elasticity.HasAdjoint()) {glvis->Append(optproblem.GetAdjState(), "adjoint displacement", keys);}
   }

   real_t stationarity0, obj0,
          stationarity_error_L2, stationarity_error_bregman, stationarity_error,
          curr_vol,
          objval(infinity()), old_objval(infinity()), succ_obj_diff(infinity());
   real_t kkt, kkt0;
   int tot_reeval(0), num_reeval(0);
   int it_mma;
   TableLogger logger;
   logger.Append("it", it_mma);
   logger.Append("volume", curr_vol);
   logger.Append("obj", objval);
   logger.Append("step-size", step_size);
   logger.Append("num-reeval", num_reeval);
   logger.Append("succ-objdiff", succ_obj_diff);
   logger.Append("stnrty-L2", stationarity_error_L2);
   logger.Append("stnrty-B", stationarity_error_bregman);
   logger.Append("kkt", kkt);
   logger.SaveWhenPrint(filename.str());
   std::unique_ptr<ParaViewDataCollection> paraview_dc;
   if (use_paraview)
   {
      paraview_dc.reset(new mfem::ParaViewDataCollection(filename.str(), mesh.get()));
      if (paraview_dc->Error()) { use_paraview=false; }
      else
      {
         paraview_dc->SetPrefixPath("ParaView");
         paraview_dc->SetLevelsOfDetail(order_state);
         paraview_dc->SetDataFormat(VTKFormat::BINARY);
         paraview_dc->SetHighOrderOutput(true);
         // paraview_dc->RegisterField("displacement", &state_gf);
         // paraview_dc->RegisterField("density", &density_gf);
         paraview_dc->RegisterField("filtered_density", &filter_gf);
      }
   }
   ParGridFunction M_grad_gf(&fes_control), lower(&fes_control),
                   upper(&fes_control);
   real_t max_ch(0.1);
   MMAOpt mma(mesh->GetComm(), control_gf.Size(), 1, control_gf);
   optproblem.Eval();
   grad_gf = 0.0;
   Vector con(1);
   real_t volume_correction;
   for (it_mma = 0; it_mma<max_it; it_mma++)
   {
      if (Mpi::Root()) { out << "MMA Step " << it_mma << std::endl; }

      control_old_gf = control_gf;
      for (int i=0; i<control_gf.Size(); i++)
      {
         lower[i] = std::max(0.0, control_gf[i] - max_ch);
         upper[i] = std::min(1.0, control_gf[i] + max_ch);
      }
      con[0] = optproblem.GetCurrentVolume() - max_vol;
      mma.Update(it_mma, grad_gf, con, dv, lower, upper, control_gf);

      old_objval = objval;
      objval = optproblem.Eval();
      succ_obj_diff = old_objval - objval;
      grad_old_gf = grad_gf;
      curr_vol = optproblem.GetCurrentVolume();

      density_gf.ProjectCoefficient(density_cf);
      if (it_mma % vis_steps == 0)
      {
         if (use_glvis) { glvis->Update(); }
         if (use_paraview && !(paraview_dc->Error()))
         {
            if (!overwrite_paraview)
            {
               paraview_dc->SetCycle(it_mma);
               paraview_dc->SetTime(it_mma);
            }
            paraview_dc->Save();
         }
         else {use_paraview = false;}
      }

      optproblem.UpdateGradient();
      avg_grad = InnerProduct(fes_control.GetComm(), grad_gf, dv)/tot_vol;

      real_t dummy1, dummy2;
      control_eps_gf = control_gf;
      density.ProjectedStep(control_eps_gf, eps_stationarity, grad_gf, dummy1,
                            dummy2);
      stationarity_error_bregman = std::sqrt(zero_gf.ComputeL1Error(
                                                bregman_diff_eps))/eps_stationarity;

      stationarity_error_L2 = stationarity_error_bregman;
      kkt = zero_gf.ComputeL1Error(KKT_cf) + (max_vol - curr_vol)/max_vol;
      zero_gf.ComputeElementL1Errors(KKT_cf, kkt_gf);

      stationarity_error = use_bregman_stationary
                           ? stationarity_error_bregman : stationarity_error_L2;
      logger.Print(true);
      if (Mpi::Root())
      {
         out << "--------------------------------------------" << std::endl;
      }
      if (it_mma == 0)
      {
         stationarity0 = stationarity_error;
         kkt0 = kkt;
      }
      if ((kkt < tol_kkt_rel*kkt0) || (kkt < tol_kkt_abs))
      {
         if (it_mma > min_it) { break; }
      }
   }
   if (Mpi::Root())
   {
      out << filename.str() << " terminated after " << it_mma
          << " with " << tot_reeval << " re-eval" << std::endl;
   }
   if (use_paraview && !paraview_dc->Error())
   {
      if (!overwrite_paraview)
      {
         paraview_dc->SetCycle(it_mma);
         paraview_dc->SetTime(it_mma);
      }
      paraview_dc->Save();
   }
   logger.CloseFile();
   return 0;
}
