#include "mfem.hpp"
#include "miniapps/simpl/topopt.hpp"
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
   int KKT_type = 1;
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
   real_t max_step_size = infinity();
   real_t entropyPenalty = -1.0;

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
   real_t tol_rel = 1e-05;
   real_t tol_abs = 1e-05;
   // backtracking related
   int max_it_backtrack = 20;
   bool use_bregman_backtrack = true;
   bool use_L2_stationarity = false;
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
   args.AddOption(&tol_rel, "-rtol", "--rel-tol",
                  "Tolerance for relative KKT residual");
   args.AddOption(&tol_abs, "-atol", "--abs-tol",
                  "Tolerance for absolute KKT residual");
   args.AddOption(&KKT_type, "-kkt-type", "--kkt-type",
                  "KKT type: 1) Brendan, 2) Thomas");
   args.AddOption(&step_size, "-a0", "--init-step",
                  "Initial step size");
   args.AddOption(&use_bregman_backtrack, "-bb", "--bregman-backtrack", "-ab",
                  "--armijo-backtrack",
                  "Option to choose Bregman backtracking algorithm or Armijo backtracking algorithm");
   args.AddOption(&use_L2_stationarity, "-L2", "--L2-stationarity", "-kkt",
                  "--kkt-residual",
                  "Option to use L2 stationarity for the stopping criteria. KKT is the default");

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
   filename << "SiMPL-" << (use_bregman_backtrack?"B-":"A-");

   Array2D<int> ess_bdr_state;
   Array<int> ess_bdr_filter;
   real_t tot_vol;
   std::unique_ptr<ParMesh> mesh((ParMesh*)GetTopoptMesh(
                                    prob, filename,
                                    r_min, tot_vol, min_vol, max_vol,
                                    E, nu, ess_bdr_state, ess_bdr_filter,
                                    solid_attr, void_attr,
                                    ser_ref_levels, par_ref_levels));
   filename << "-" << ser_ref_levels + par_ref_levels << "-" << order_control;
   if (use_L2_stationarity)
   {
      filename << "-L2";
   }
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

   const int nrObj = std::max(1, prob / 100);

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
                << "\tThe number of state   unknowns: " << state_ndof << std::endl
                << "\tThe number of objectives      : " << nrObj << std::endl;
   }

   if (Mpi::Root()) { out << "Creating gridfunctions ... " << std::flush; }
   ParGridFunction control_gf(&fes_control); control_gf = 0.0;
   ParGridFunction filter_gf(&fes_filter); filter_gf = 0.0;
   std::vector<std::unique_ptr<GridFunction>> state_gf(nrObj);
   for (int i=0; i<nrObj; i++) { state_gf[i].reset(new ParGridFunction(&fes_state)); *state_gf[i] = 0.0; }
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
   FermiDiracEntropy entropy;
   PrimalEntropy entropy_primal;
   entropy.SetFiniteLowerBound(-max_latent);
   entropy.SetFiniteUpperBound(+max_latent);
   control_gf = entropy.forward((min_vol ? min_vol : max_vol)/tot_vol);
   MappedGFCoefficient density_cf = entropy.GetBackwardCoeff(control_gf);
   DesignDensity density(fes_control, tot_vol, min_vol, max_vol, &entropy);
   DesignDensity density_primal(fes_control, tot_vol, min_vol, max_vol,
                                &entropy_primal);
   density.SetVoidAttr(void_attr);
   density.SetSolidAttr(solid_attr);
   // Filter
   HelmholtzFilter filter(fes_filter, ess_bdr_filter, r_min, true);
   filter.GetLinearForm()[0]->AddDomainIntegrator(new DomainLFIntegrator(
                                                     density_cf));
   filter.SetBStationary(false);
   std::vector<std::unique_ptr<StrainEnergyDensityCoefficient>> energy(nrObj);
   for (int i=0; i<state_gf.size(); i++)
   {
      energy[i].reset(new StrainEnergyDensityCoefficient(
                         lambda_cf, mu_cf, der_simp_cf, *state_gf[i], nullptr));
      filter.GetAdjLinearForm()[0]->AddDomainIntegrator(new DomainLFIntegrator(
                                                           *energy[i]));
   }
   if (prob == mfem::ForceInverter2)
   {
      ForceInverterInitialDesign(control_gf, &entropy);
   }
   if (prob == mfem::ForceInverter3)
   {
      ForceInverter3InitialDesign(control_gf, &entropy);
   }

   // elasticity
   ElasticityProblem elasticity(fes_state, ess_bdr_state, lambda_simp_cf,
                                mu_simp_cf, prob < 0, nrObj);
   elasticity.SetAStationary(false);
   SetupTopoptProblem(prob, filter, elasticity, filter_gf, state_gf);
   DensityBasedTopOpt optproblem(density, control_gf, grad_gf,
                                 filter, filter_gf, grad_filter_gf,
                                 elasticity, state_gf);
   if (elasticity.HasAdjoint())
   {
      for (int i=0; i<state_gf.size(); i++)
      {
         energy[i]->SetAdjState(*optproblem.GetAdjState()[i]);
      }
   }
   if (Mpi::Root()) { out << "done" << std::endl; }

   // Backtracking related stuffs
   MappedPairedGFCoefficient bregman_diff_old
      = entropy.GetBregman_dual(control_old_gf, control_gf);
   MappedPairedGFCoefficient bregman_diff_eps
      = entropy.GetBregman_dual(control_eps_gf, control_gf);
   MappedPairedGFCoefficient primal_diff_eps
      = entropy_primal.GetBregman_dual(control_eps_gf, density_gf);
   MappedPairedGFCoefficient diff_density_cf(
      control_gf, control_old_gf, [&entropy](const real_t x, const real_t y)
   {
      return entropy.backward(x)-entropy.backward(y);
   });
   MappedGFCoefficient density_eps_dual_cf = entropy.GetBackwardCoeff(
                                                control_eps_gf);
   real_t avg_grad;
   real_t step_size_old;
   real_t lambda_V;
   real_t volume_correction;
   MappedPairedGFCoefficient KKT_cf;
   KKT_cf.SetGridFunction(&control_eps_gf, &control_gf);
   switch (KKT_type)
   {
      case 1:
      {
         KKT_cf.SetFunction([&avg_grad, &entropy, &step_size](const real_t x,
                                                              const real_t x_old)
         {
            real_t rho_k = entropy.backward(x_old);
            real_t lambda_k = (x - x_old)/step_size; // -grad F(rho_{k-1})
            // Definition -Brendan
            return std::max(std::max(-lambda_k*rho_k, lambda_k*(1-rho_k)), 0.0);
            // return std::max(0.0, lambda_k)*(1.0-rho_k)-std::min(0.0, lambda_k)*rho_k;
            // Definition -Thomas
            // More robust when gradient has large magnitude.
            return std::fabs(lambda_k
                             - std::min(0.0, rho_k     + lambda_k)
                             - std::max(0.0, rho_k - 1 + lambda_k));
            // return lambda_k*(1.0-2*rho_k) >= 0;
         });
         break;
      }
      case 2:
      {
         KKT_cf.SetFunction([&avg_grad, &entropy, &step_size](const real_t x,
                                                              const real_t x_old)
         {
            real_t rho_k = entropy.backward(x_old);
            real_t lambda_k = (x - x_old)/step_size; // -grad F(rho_{k-1})
            // Definition -Thomas
            // More robust when gradient has large magnitude.
            return std::fabs(lambda_k
                             - std::min(0.0, rho_k     + lambda_k)
                             - std::max(0.0, rho_k - 1 + lambda_k));
            // return lambda_k*(1.0-2*rho_k) >= 0;
         });
         break;
      }
      default: MFEM_ABORT("Undefined KKT type");
   }
   ParBilinearForm mass(&fes_control);
   mass.AddDomainIntegrator(new MassIntegrator());
   mass.Assemble();
   mass.Finalize();
   ParGridFunction kkt_gf(&fes_control);
   std::unique_ptr<HypreParMatrix> Mass(mass.ParallelAssemble());

   GridFunctionCoefficient density_eps_primal_cf(&control_eps_gf);
   GridFunctionCoefficient grad_cf(&grad_gf);
   ParGridFunction zero_gf(&fes_control), one_gf(&fes_control);
   zero_gf=0.0; one_gf=1.0;
   ParGridFunction dv(&fes_control);
   Mass->Mult(one_gf, dv);
   ParLinearForm diff_density_form(&fes_control);
   diff_density_form.AddDomainIntegrator(new DomainLFIntegrator(diff_density_cf));

   std::unique_ptr<GLVis> glvis;
   if (use_glvis)
   {
      glvis.reset(new GLVis ("localhost", 19916, true));
      const char keys[] = "Rjmml****************";
      glvis->Append(control_gf, "control variable", keys);
      density_gf.ProjectCoefficient(density_cf);
      // glvis->Append(density_gf, "design density", keys);
      glvis->Append(filter_gf, "filtered density", keys);
      // glvis->Append(state_gf, "displacement magnitude", keys);
      glvis->Append(kkt_gf, "KKT", keys);
      // if (elasticity.HasAdjoint()) {glvis->Append(optproblem.GetAdjState(), "adjoint displacement", keys);}
   }

   real_t stationarity0, obj0,
          stationarity,
          curr_vol,
          objval(infinity()), old_objval(infinity()), succ_obj_diff(infinity());
   real_t kkt, kkt0(infinity());
   int tot_reeval(0), num_reeval(0);
   int it_md;
   TableLogger logger;
   logger.Append("it", it_md);
   logger.Append("volume", curr_vol);
   logger.Append("obj", objval);
   logger.Append("step-size", step_size);
   logger.Append("num-reeval", num_reeval);
   logger.Append("succ-objdiff", succ_obj_diff);
   logger.Append("stnrty-L2", stationarity);
   logger.Append("kkt", kkt);
   logger.Append("tot_reeval", tot_reeval);
   logger.Append("volume_correction", volume_correction);
   logger.Append("penalty", entropyPenalty);
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
         paraview_dc->RegisterField("density", &control_gf);
         paraview_dc->RegisterField("filtered_density", &filter_gf);
      }
   }
   grad_gf = 0.0;
   ParGridFunction dual_B(&fes_control);
   real_t dual_V;
   for (it_md = 0; it_md<max_it; it_md++)
   {
      step_size_old = step_size;
      if (Mpi::Root()) { out << "Mirror Descent Step " << it_md << std::endl; }
      if (it_md == 1 && step_size == -1.0)
      {
         step_size = 1.0 / grad_gf.ComputeMaxError(zero_cf);
      }
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
         // step_size = std::sqrt(step_size*step_size_old);
         if (Mpi::Root())
         {
            out << "   Step size = " << step_size
                << " = | " << psi_diffrho << " / " << grad_diffrho << " |" << std::endl;
         }
         step_size = std::sqrt(step_size * step_size_old);
         if (!isfinite(step_size))
         {
            break;
         }
      }
      step_size = std::min(step_size, max_step_size);

      control_old_gf = control_gf;
      old_objval = objval;
      bool converged = false;
      entropyPenalty = -1.0;
      for (num_reeval=0; num_reeval < max_it_backtrack; num_reeval++)
      {
         control_gf = control_old_gf;
         density.ProjectedStep(control_gf, step_size, grad_gf, volume_correction,
                               curr_vol, &entropyPenalty);
         objval = optproblem.Eval();
         // kkt = zero_gf.ComputeL1Error(KKT_cf);
         // if (it_md > 1 && !use_L2_stationarity && (kkt < tol_rel*kkt0 || kkt < tol_abs))
         // {
         //    converged = true;
         //    break;
         // }
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
         if (objval <= target_objval)
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
      if (num_reeval == max_it_backtrack)
      {
         if (succ_obj_diff < 1e-08)
         {
            if (Mpi::Root())
            {
               out << " Failed to find feasible direction and successive difference is too small. Terminate SiMPL"
                   << std::endl;;
            }
            break;
         }
      }
      succ_obj_diff = old_objval - objval;
      grad_old_gf = grad_gf;

      density_gf.ProjectCoefficient(density_cf);

      optproblem.UpdateGradient();
      real_t dummy2;

      control_eps_gf.ProjectCoefficient(density_cf);
      density_primal.ProjectedStep(control_eps_gf, 1.0, grad_gf, lambda_V, dummy2);
      stationarity = control_eps_gf.ComputeL2Error(density_cf);
      control_eps_gf = control_gf;
      density.ProjectedStep(control_eps_gf, step_size > 0 ? step_size : 1.0, grad_gf,
                            lambda_V, dummy2);
      kkt = zero_gf.ComputeL1Error(KKT_cf);
      // control_eps_gf = control_gf;
      // density.ProjectedStep(control_eps_gf, 1.0, grad_gf, lambda_V, dummy2);
      // kkt = zero_gf.ComputeL1Error(KKT_cf);
      // zero_gf.ComputeElementMaxErrors(KKT_cf, kkt_gf);
      // kkt = ComputeKKT(control_gf, grad_gf, entropy, solid_attr, void_attr,
      //                  min_vol, max_vol, curr_vol,
      //                  one_gf, zero_gf, dv, kkt_gf, dual_V);

      if (it_md == 0)
      {
         stationarity0 = stationarity;
         kkt0 = kkt;
      }
      logger.Print(true);
      if (Mpi::Root())
      {
         out << "--------------------------------------------" << std::endl;
      }
      if (it_md % vis_steps == 0)
      {
         kkt_gf.ProjectCoefficient(KKT_cf);
         if (use_glvis) { glvis->Update(); }
         if (use_paraview && !(paraview_dc->Error()))
         {
            if (!overwrite_paraview)
            {
               paraview_dc->SetCycle(it_md);
               paraview_dc->SetTime(it_md);
            }
            paraview_dc->Save();
         }
         else {use_paraview = false;}
      }
      if (it_md > min_it)
      {
         if (use_L2_stationarity && (stationarity < tol_rel*stationarity0 ||
                                     stationarity < tol_abs))
         {
            break;
         }
         if (!use_L2_stationarity && (kkt < tol_rel*kkt0 || kkt < tol_abs))
         {
            break;
         }
      }
   }
   if (Mpi::Root())
   {
      out << filename.str() << " terminated after " << it_md
          << " with " << tot_reeval << " re-eval" << std::endl;
   }
   if (use_glvis)
   {
      glvis->Update();
   }
   if (use_paraview && !paraview_dc->Error())
   {
      if (!overwrite_paraview)
      {
         paraview_dc->SetCycle(it_md);
         paraview_dc->SetTime(it_md);
      }
      paraview_dc->Save();
   }
   logger.CloseFile();
   return 0;
}

