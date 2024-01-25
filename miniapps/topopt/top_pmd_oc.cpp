// Compliancemi minimization with projected mirror descent OC in parallel
// where Bregman divergence is enduced from the Shannon entropy
// and the descent direction is motivated from OC, log(1/-∇F).
//
//                  minimize F(ρ) = ∫_Ω f⋅u dx over ρ ∈ L¹(Ω)
//
//                  subject to
//
//                    -Div(r(ρ̃)Cε(u)) = f       in Ω + BCs
//                    -ϵ²Δρ̃ + ρ̃ = ρ             in Ω + Neumann BCs
//                    0 ≤ ρ ≤ 1                 in Ω
//                    ∫_Ω ρ dx = θ vol(Ω)
//
//              Here, r(ρ̃) = ρ₀ + ρ̃³ (1-ρ₀) is the solid isotropic material
//              penalization (SIMP) law, C is the elasticity tensor for an
//              isotropic linearly elastic material, ϵ > 0 is the design
//              length scale, and 0 < θ < 1 is the volume fraction.
//
//              Update is done by
//
//              ρ_new = clip(exp(ψ_new)) = clip(exp(ψ_cur - α d_cur + c))
//              with d_cur = log(1/-∇F(ρ_cur))
//
//              where c is a constant volume correction. The step size α is
//              determined by a generalized Barzilai-Borwein method with
//              Armijo condition check
//
//              BB:        α_init = |(δψ, δρ) / (δd, δρ)|
//
//              Armijo:   F(ρ(α)) ≤ F(ρ_cur) + c_1 (∇F(ρ_cur), ρ(α) - ρ_cur)
//                        with ρ(α) = clip(exp(ψ_cur - α d_cur + c))

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include "topopt.hpp"
#include "helper.hpp"
#include "prob_elasticity.hpp"

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int ref_levels = 6;
   int order = 1;
   // filter radius. Use problem-dependent default value if not provided.
   // See switch statements below
   double filter_radius = -1;
   // Volume fraction. Use problem-dependent default value if not provided.
   // See switch statements below
   double vol_fraction = -1;
   int max_it = 2e2;
   double rho_min = 1e-06;
   double exponent = 3.0;
   double lambda = 1.0;
   double mu = 1.0;
   double c1 = 1e-04;
   bool glvis_visualization = true;
   bool save = false;
   bool paraview = true;
   double tol_stationarity = 5e-05;
   double tol_compliance = 5e-05;

   ostringstream filename_prefix;
   filename_prefix << "PMD_OC-";

   int problem = ElasticityProblem::Cantilever;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem number: 0) Cantilever, 1) MBB, 2) LBracket.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&filter_radius, "-fr", "--filter-radius",
                  "Length scale for ρ.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&vol_fraction, "-vf", "--volume-fraction",
                  "Volume fraction for the material density.");
   args.AddOption(&lambda, "-lambda", "--lambda",
                  "Lamé constant λ.");
   args.AddOption(&mu, "-mu", "--mu",
                  "Lamé constant μ.");
   args.AddOption(&rho_min, "-rmin", "--rho-min",
                  "Minimum of density coefficient.");
   args.AddOption(&glvis_visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good()) {args.PrintUsage(mfem::out);}


   std::unique_ptr<Mesh> mesh;
   Array2D<int> ess_bdr;
   Array<int> ess_bdr_filter;
   std::unique_ptr<VectorCoefficient> vforce_cf;
   std::string prob_name;
   GetElasticityProblem((ElasticityProblem)problem, filter_radius, vol_fraction,
                        mesh, vforce_cf,
                        ess_bdr, ess_bdr_filter,
                        prob_name, ref_levels);
   filename_prefix << prob_name;
   int dim = mesh->Dimension();
   const int num_el = mesh->GetNE();

   mfem::out << "\n"
             << "Compliance Minimization with Projected Mirror Descent OC.\n"
             << "Problem: " << filename_prefix.str() << "\n"
             << "The number of elements: " << num_el << "\n"
             << "Order: " << order << "\n"
             << "Volume Fraction: " << vol_fraction << "\n"
             << "Filter Radius: " << filter_radius << "\n"
             << "Maximum iteration: " << max_it << "\n"
             << "GLVis: " << glvis_visualization << "\n"
             << "Paraview: " << paraview << std::endl;

   if (glvis_visualization && dim == 3)
   {
      glvis_visualization = false;
      paraview = true;
      mfem::out << "GLVis for 3D is disabled. Use ParaView" << std::endl;
   }

   if (save)
   {
      ostringstream meshfile;
      meshfile << filename_prefix.str() << "-" << ref_levels << ".mesh";
      ofstream mesh_ofs(meshfile.str().c_str());
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
   }

   // 4. Define the necessary finite element spaces on the mesh->
   H1_FECollection state_fec(order, dim); // space for u
   H1_FECollection filter_fec(order, dim); // space for ρ̃
   L2_FECollection control_fec(order-1, dim,
                               BasisType::GaussLobatto); // space for ψ
   FiniteElementSpace state_fes(mesh.get(), &state_fec,dim);
   FiniteElementSpace filter_fes(mesh.get(), &filter_fec);
   FiniteElementSpace control_fes(mesh.get(), &control_fec);

   int state_size = state_fes.GetTrueVSize();
   int control_size = control_fes.GetTrueVSize();
   int filter_size = filter_fes.GetTrueVSize();
   mfem::out << "\n"
             << "Number of state unknowns: " << state_size << "\n"
             << "Number of filter unknowns: " << filter_size << "\n"
             << "Number of control unknowns: " << control_size << std::endl;

   // 5. Set the initial guess for ρ.
   SIMPProjector simp_rule(exponent, rho_min);
   HelmholtzFilter filter(filter_fes, filter_radius/(2.0*sqrt(3.0)),
                          ess_bdr_filter);
   LatentDesignDensity density(control_fes, filter_fes, filter, vol_fraction,
                               ShannonEntropy, log_d, exp_d, false, true);

   ConstantCoefficient lambda_cf(lambda), mu_cf(mu);
   ParametrizedElasticityEquation elasticity(state_fes,
                                             density.GetFilteredDensity(), simp_rule, lambda_cf, mu_cf, *vforce_cf, ess_bdr);
   TopOptProblem optprob(elasticity.GetLinearForm(), elasticity, density, false,
                         true);

   GridFunction &u = optprob.GetState();
   GridFunction &rho_filter = density.GetFilteredDensity();
   // 10. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_SIMP, sout_r;
   std::unique_ptr<GridFunction> designDensity_gf, rho_gf;
   if (glvis_visualization)
   {
      designDensity_gf.reset(new GridFunction(&filter_fes));
      rho_gf.reset(new GridFunction(&filter_fes));
      designDensity_gf->ProjectCoefficient(simp_rule.GetPhysicalDensity(
                                              density.GetFilteredDensity()));
      rho_gf->ProjectCoefficient(density.GetDensityCoefficient());
      sout_SIMP.open(vishost, visport);
      if (sout_SIMP.is_open())
      {
         sout_SIMP.precision(8);
         sout_SIMP << "solution\n" << *mesh << *designDensity_gf
                   << "window_title 'Design density r(ρ̃) - PMD_OC "
                   << problem << "'\n"
                   << "keys Rjl***************\n"
                   << flush;
      }
      sout_r.open(vishost, visport);
      if (sout_r.is_open())
      {
         sout_r.precision(8);
         sout_r << "solution\n" << *mesh << *rho_gf
                << "window_title 'Raw density ρ - PMD_OC "
                << problem << "'\n"
                << "keys Rjl***************\n"
                << flush;
      }
   }
   std::unique_ptr<ParaViewDataCollection> pd;
   if (paraview)
   {
      pd.reset(new ParaViewDataCollection(filename_prefix.str(), mesh.get()));
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("state", &u);
      pd->RegisterField("psi", &density.GetGridFunction());
      pd->RegisterField("frho", &rho_filter);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0);
      pd->Save();
   }

   // 11. Iterate
   GridFunction &grad(optprob.GetGradient());
   GridFunction &psi(density.GetGridFunction());
   GridFunction old_grad(&control_fes), old_psi(&control_fes);

   LinearForm diff_rho_form(&control_fes);
   std::unique_ptr<Coefficient> diff_rho(optprob.GetDensityDiffCoeff(old_psi));
   diff_rho_form.AddDomainIntegrator(new DomainLFIntegrator(*diff_rho));

   mfem::out << "\n"
             << "Initialization Done." << "\n"
             << "Start Projected Mirror Descent OC Step." << "\n" << std::endl;

   double compliance = optprob.Eval();
   double step_size(0), volume(density.GetDomainVolume()*vol_fraction),
          stationarityError(infinity()), stationarityError_bregman(infinity());
   int num_reeval(0);
   double old_compliance;

   TableLogger logger;
   logger.Append(std::string("Volume"), volume);
   logger.Append(std::string("Compliance"), compliance);
   logger.Append(std::string("Stationarity"), stationarityError);
   logger.Append(std::string("Re-evel"), num_reeval);
   logger.Append(std::string("Step Size"), step_size);
   logger.Append(std::string("Stationarity-Bregman"), stationarityError_bregman);
   logger.Print();

   optprob.UpdateGradient();
   bool converged = false;
   for (int k = 0; k < max_it; k++)
   {
      // Step 1. Compute Step size
      if (k == 0) { step_size = 1.0; }
      else
      {
         diff_rho_form.Assemble();
         old_psi -= psi;
         old_grad -= grad;
         step_size = std::fabs(diff_rho_form(old_psi)  / diff_rho_form(old_grad));
      }

      // Step 2. Store old data
      old_compliance = compliance;
      old_psi = psi;
      old_grad = grad;

      // Step 3. Step and upate gradient
      num_reeval = Step_Armijo(optprob, old_psi, grad, diff_rho_form, c1, step_size);
      compliance = optprob.GetValue();
      volume = density.GetVolume();
      optprob.UpdateGradient();

      // Step 4. Visualization
      if (glvis_visualization)
      {
         if (sout_SIMP.is_open())
         {
            designDensity_gf->ProjectCoefficient(simp_rule.GetPhysicalDensity(
                                                    density.GetFilteredDensity()));
            sout_SIMP << "solution\n" << *mesh << *designDensity_gf
                      << flush;
         }
         if (sout_r.is_open())
         {
            rho_gf->ProjectCoefficient(density.GetDensityCoefficient());
            sout_r << "solution\n" << *mesh << *rho_gf
                   << flush;
         }
      }
      if (paraview)
      {
         pd->SetCycle(k);
         pd->SetTime(k);
         pd->Save();
      }

      // Check convergence
      stationarityError = density.StationarityErrorL2(grad);
      stationarityError_bregman = density.StationarityError(grad);

      logger.Print();

      if (stationarityError < tol_stationarity &&
          std::fabs(old_compliance - compliance) < tol_compliance)
      {
         converged = true;
         mfem::out << "Total number of iteration = " << k + 1 << std::endl;
         break;
      }
   }
   if (!converged)
   {
      mfem::out << "Total number of iteration = " << max_it << std::endl;
      mfem::out << "Maximum iteration reached." << std::endl;
   }
   if (save)
   {
      ostringstream solfile, solfile2;
      solfile << filename_prefix.str() << "-" << ref_levels << "-0.gf";
      solfile2 << filename_prefix.str() << "-" << ref_levels << "-f.gf";
      ofstream sol_ofs(solfile.str().c_str());
      sol_ofs.precision(8);
      sol_ofs << psi;

      ofstream sol_ofs2(solfile2.str().c_str());
      sol_ofs2.precision(8);
      sol_ofs2 << density.GetFilteredDensity();
   }

   return 0;
}
