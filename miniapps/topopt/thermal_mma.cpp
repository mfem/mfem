// Thermal compliancemi minimization with projected mirror descent
//
//                  minimize F(ρ) = ∫_Ω fu dx over ρ ∈ L¹(Ω)
//
//                  subject to
//
//                    - ∇⋅(K∇u) = f             in Ω + BCs
//                    -ϵ²Δρ̃ + ρ̃ = ρ             in Ω + Neumann BCs
//                    0 ≤ ρ ≤ 1                 in Ω
//                    ∫_Ω ρ dx = θ vol(Ω)
//
//              Here, r(ρ̃) = ρ₀ + ρ̃³ (1-ρ₀) is the solid isotropic material
//              penalization (SIMP) law, K is the diffusion coefficient,
//              ϵ > 0 is the design length scale, and 0 < θ < 1 is the volume fraction.
//
//              Update is done by
//
//              ρ_new = sigmoid(ψ_new) = sigmoid(ψ_cur - α ∇F(ρ_cur) + c)
//
//              where c is a constant volume correction. The step size α is
//              determined by a generalized Barzilai-Borwein method with
//              Armijo condition check
//
//              BB:        α_init = |(δψ, δρ) / (δ∇F(ρ), δρ)|
//
//              Armijo:   F(ρ(α)) ≤ F(ρ_cur) + c_1 (∇F(ρ_cur), ρ(α) - ρ_cur)
//                        with ρ(α) = sigmoid(ψ_cur - α∇F(ρ_cur) + c)
//
// [1] Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., & Sigmund, O.
//    (2011). Efficient topology optimization in MATLAB using 88 lines of
//    code. Structural and Multidisciplinary Optimization, 43(1), 1-16.
// [2] Keith, B. and Surowiec, T. (2023) Proximal Galerkin: A structure-
//     preserving finite element method for pointwise bound constraints.
//     arXiv:2307.12444 [math.NA]
// [3] Lazarov, B. S., & Sigmund, O. (2011). Filters in topology optimization
//     based on Helmholtz‐type differential equations. International Journal
//     for Numerical Methods in Engineering, 86(6), 765-781.

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include "topopt.hpp"
#include "helper.hpp"
#include "MMA_serial.hpp"

using namespace std;
using namespace mfem;

enum Problem
{
   HeatSink
};


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
   double c1 = 1e-04;
   bool glvis_visualization = true;
   bool save = false;
   bool paraview = true;
   double tol_stationarity = 1e-04;
   double tol_compliance = 5e-05;
   double K = 1.0;
   double mv = 0.2;

   ostringstream filename_prefix;
   filename_prefix << "PMD-";

   int problem = Problem::HeatSink;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem number: 0) Heat Sink.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&filter_radius, "-fr", "--filter-radius",
                  "Length scale for ρ.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&vol_fraction, "-vf", "--volume-fraction",
                  "Volume fraction for the material density.");
   args.AddOption(&K, "-K", "--K",
                  "Diffusion coefficient K.");
   args.AddOption(&rho_min, "-rmin", "--rho-min",
                  "Minimum of density coefficient.");
   args.AddOption(&glvis_visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good()) {args.PrintUsage(mfem::out);}


   Mesh mesh;
   Array2D<int> ess_bdr;
   Array<int> ess_bdr_filter;
   double r = 0.05;
   std::unique_ptr<Coefficient> vforce_cf;
   string mesh_file;
   switch (problem)
   {
      case Problem::HeatSink:
         if (filter_radius < 0) { filter_radius = 0.1; }
         if (vol_fraction < 0) { vol_fraction = 0.2; }
         mesh = mesh.MakeCartesian2D(4, 4, mfem::Element::Type::QUADRILATERAL, true, 1.0,
                                     1.0);
         mesh.MarkBoundary(
         [](const Vector &x) {return std::fabs(x[0] - 0.5) < 0.25 && x[1] > 0.5; },
         5);
         mesh.SetAttributes();
         ess_bdr.SetSize(1, 5);
         ess_bdr = 0;
         ess_bdr(0, 4) = 1;
         ess_bdr_filter.SetSize(2);
         ess_bdr_filter = 0;

         vforce_cf.reset(new ConstantCoefficient(1.0));
         filename_prefix << "HeatSink-";
         break;

      default:
         mfem_error("Undefined problem.");
   }
   mesh.SetAttributes();
   int dim = mesh.Dimension();
   const int num_el = mesh.GetNE() * (int)std::pow(2, dim*ref_levels);
   filename_prefix << ref_levels;

   mfem::out << "\n"
             << "Compliance Minimization with Projected Mirror Descent.\n"
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

   // 3. Refine the mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   if (save)
   {
      ostringstream meshfile;
      meshfile << filename_prefix.str() << ".mesh";
      ofstream mesh_ofs(meshfile.str().c_str());
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);
   }

   // 4. Define the necessary finite element spaces on the mesh->
   H1_FECollection state_fec(order, dim); // space for u
   H1_FECollection filter_fec(order, dim); // space for ρ̃
   L2_FECollection control_fec(order-1, dim,
                               BasisType::GaussLobatto); // space for ψ
   FiniteElementSpace state_fes(&mesh, &state_fec);
   FiniteElementSpace filter_fes(&mesh, &filter_fec);
   FiniteElementSpace control_fes(&mesh, &control_fec);

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
   PrimalDesignDensity density(control_fes, filter, vol_fraction);

   ConstantCoefficient K_cf(K);
   ParametrizedDiffusionEquation diffusion(state_fes,
                                           density.GetFilteredDensity(), simp_rule, K_cf, *vforce_cf, ess_bdr);
   TopOptProblem optprob(diffusion.GetLinearForm(), diffusion, density, false,
                         false);

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
         sout_SIMP << "solution\n" << mesh << *designDensity_gf
                   << "window_title 'Design density r(ρ̃) - PMD "
                   << problem << "'\n"
                   << "keys Rjl***************\n"
                   << flush;
      }
      sout_r.open(vishost, visport);
      if (sout_r.is_open())
      {
         sout_r.precision(8);
         sout_r << "solution\n" << mesh << *rho_gf
                << "window_title 'Raw density ρ - PMD "
                << problem << "'\n"
                << "keys Rjl***************\n"
                << flush;
      }
   }
   std::unique_ptr<ParaViewDataCollection> pd;
   if (paraview)
   {
      pd.reset(new ParaViewDataCollection(filename_prefix.str(), &mesh));
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
   GridFunction &rho(density.GetGridFunction());
   GridFunction old_rho(&control_fes), dv(&control_fes);
   for (int i=0; i<mesh.GetNE(); i++) { dv[i] = mesh.GetElementVolume(i); }

   LinearForm diff_rho_form(&control_fes);
   std::unique_ptr<Coefficient> diff_rho(optprob.GetDensityDiffCoeff(old_rho));
   diff_rho_form.AddDomainIntegrator(new DomainLFIntegrator(*diff_rho));

   mfem::out << "\n"
             << "Initialization Done." << "\n"
             << "Start Method of Moving Asymptotes Step." << "\n" << std::endl;

   double compliance = optprob.Eval();
   double volume(density.ComputeVolume()), stationarityError(infinity());
   double old_compliance;

   TableLogger logger;
   logger.Append(std::string("Volume"), volume);
   logger.Append(std::string("Compliance"), compliance);
   logger.Append(std::string("Stationarity"), stationarityError);
   logger.SaveWhenPrint(filename_prefix.str());
   logger.Print();

   optprob.UpdateGradient();
   MMA mma(rho.Size(), 1);
   GridFunction lower(rho), upper(rho);
   bool converged = false;
   for (int k = 0; k < max_it; k++)
   {
      // Store old data
      old_compliance = compliance;
      // Step and upate gradient

      lower = rho; lower -= mv; lower.ApplyMap([](double x) {return std::max(0.0, x);});
      upper = rho; upper += mv; upper.ApplyMap([](double x) {return std::min(1.0, x);});
      double con = density.GetVolume()  - (density.GetDomainVolume()*vol_fraction);

      out << con << std::endl;
      mma.Update(rho, grad, con, dv, lower, upper);
      volume = density.ComputeVolume();
      compliance = optprob.Eval();
      optprob.UpdateGradient();

      // Step 4. Visualization
      if (glvis_visualization)
      {
         if (sout_SIMP.is_open())
         {
            designDensity_gf->ProjectCoefficient(simp_rule.GetPhysicalDensity(
                                                    density.GetFilteredDensity()));
            sout_SIMP << "solution\n" << mesh << *designDensity_gf
                      << flush;
         }
         if (sout_r.is_open())
         {
            rho_gf->ProjectCoefficient(density.GetDensityCoefficient());
            sout_r << "solution\n" << mesh << *rho_gf
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
      stationarityError = density.StationarityError(grad);

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
      sol_ofs << rho;

      ofstream sol_ofs2(solfile2.str().c_str());
      sol_ofs2.precision(8);
      sol_ofs2 << density.GetFilteredDensity();
   }

   return 0;
}
