//                              MFEM Example 37
//
//
// Compile with: make ex37
//
// Sample runs:
//     ex37 -alpha 10
//     ex37 -alpha 10 -pv
//     ex37 -kappa 0.1 -mu 0.1
//     ex37 -o 2 -alpha 5.0 -mi 50 -vf 0.4 -ntol 1e-5
//     ex37 -r 6 -o 1 -alpha 25.0 -epsilon 0.02 -mi 50 -ntol 1e-5
//
//
// Description: This example code demonstrates the use of MFEM to solve a
//              density-filtered [3] topology optimization problem. The
//              objective is to minimize the compliance
//
//                  minimize ∫_Ω f⋅u dx over u ∈ [H¹(Ω)]² and ρ ∈ L¹(Ω)
//
//                  subject to
//
//                    -Div(r(ρ̃)Cε(u)) = f       in Ω + BCs
//                    -ϵ²Δρ̃ + ρ̃ = ρ             in Ω + Neumann BCs
//                    0 ≤ ρ ≤ 1                 in Ω
//                    ∫_Ω ρ dx = θ vol(Ω)
//
//              Here, r(ρ̃) = ρ₀ + ρ̃³ (1-ρ₀) is the solid isotropic material
//              penalization (SIMP) law, C is the diffusion tensor for an
//              isotropic linearly elastic material, ϵ > 0 is the design
//              length scale, and 0 < θ < 1 is the volume fraction.
//
//              The problem is discretized and gradients are computing using
//              finite elements [1]. The design is optimized using an entropic
//              mirror descent algorithm introduced by Keith and Surowiec [2]
//              that is tailored to the bound constraint 0 ≤ ρ ≤ 1.
//
//              This example highlights the ability of MFEM to deliver high-
//              order solutions to inverse design problems and showcases how
//              to set up and solve PDE-constrained optimization problems
//              using the so-called reduced space approach.
//
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

using namespace std;
using namespace mfem;

/**
 * ---------------------------------------------------------------
 *                      ALGORITHM PREAMBLE
 * ---------------------------------------------------------------
 *
 *  The Lagrangian for this problem is
 *
 *          L(u,ρ,ρ̃,w,w̃) = (f,u) - (r(ρ̃) C ε(u),ε(w)) + (f,w)
 *                       - (ϵ² ∇ρ̃,∇w̃) - (ρ̃,w̃) + (ρ,w̃)
 *
 *  where
 *
 *    r(ρ̃) = ρ₀ + ρ̃³ (1 - ρ₀)       (SIMP rule)
 *
 *    ε(u) = (∇u + ∇uᵀ)/2           (symmetric gradient)
 *
 *    C e = λtr(e)I + 2μe           (isotropic material)
 *
 *  NOTE: The Lame parameters can be computed from Young's modulus E
 *        and Poisson's ratio ν as follows:
 *
 *             λ = E ν/((1+ν)(1-2ν)),      μ = E/(2(1+ν))
 *
 * ---------------------------------------------------------------
 *
 *  Discretization choices:
 *
 *     u ∈ V ⊂ (H¹)ᵈ (order p)
 *     ψ ∈ L² (order p - 1), ρ = sigmoid(ψ)
 *     ρ̃ ∈ H¹ (order p)
 *     w ∈ V  (order p)
 *     w̃ ∈ H¹ (order p)
 *
 * ---------------------------------------------------------------
 *                          ALGORITHM
 * ---------------------------------------------------------------
 *
 *  Update ρ with projected mirror descent via the following algorithm.
 *
 *  1. Initialize ψ = inv_sigmoid(vol_fraction) so that ∫ sigmoid(ψ) = θ vol(Ω)
 *
 *  While not converged:
 *
 *     2. Solve filter equation ∂_w̃ L = 0; i.e.,
 *
 *           (ϵ² ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)   ∀ v ∈ H¹.
 *
 *     3. Solve primal problem ∂_w L = 0; i.e.,
 *
 *      (λ r(ρ̃) ∇⋅u, ∇⋅v) + (2 μ r(ρ̃) ε(u), ε(v)) = (f,v)   ∀ v ∈ V.
 *
 *     NB. The dual problem ∂_u L = 0 is the negative of the primal problem due to symmetry.
 *
 *     4. Solve for filtered gradient ∂_ρ̃ L = 0; i.e.,
 *
 *      (ϵ² ∇ w̃ , ∇ v ) + (w̃ ,v) = (-r'(ρ̃) ( λ |∇⋅u|² + 2 μ |ε(u)|²),v)   ∀ v ∈ H¹.
 *
 *     5. Project the gradient onto the discrete latent space; i.e., solve
 *
 *                         (G,v) = (w̃,v)   ∀ v ∈ L².
 *
 *     6. Bregman proximal gradient update; i.e.,
 *
 *                            ψ ← ψ - αG + c,
 *
 *     where α > 0 is a step size parameter and c ∈ R is a constant ensuring
 *
 *                     ∫_Ω sigmoid(ψ - αG + c) dx = θ vol(Ω).
 *
 *  end
 *
 */

enum Problem
{
   HeatSink
};

int main(int argc, char *argv[])
{

   // 1. Parse command-line options.
   int ref_levels = 5;
   int order = 1;
   double epsilon = 1e-2;
   double vol_fraction = 0.5;
   int max_it = 2e2;
   double rho_min = 1e-3;
   double exponent = 3.0;
   double kappa = 1.0;
   double mu = 1.0;
   double c1 = 1e-04;
   bool glvis_visualization = true;
   bool save = true;
   bool paraview = true;

   ostringstream solfile, solfile2, meshfile, paraview_folder;

   int problem = Problem::HeatSink;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem number: 0) Cantilever, 1) MBB, 2) LBracket.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&epsilon, "-epsilon", "--epsilon-thickness",
                  "Length scale for ρ.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&vol_fraction, "-vf", "--volume-fraction",
                  "Volume fraction for the material density.");
   args.AddOption(&kappa, "-kappa", "--kappa",
                  "Diffusion coefficient.");
   args.AddOption(&rho_min, "-rmin", "--psi-min",
                  "Minimum of density coefficient.");
   args.AddOption(&glvis_visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   Mesh mesh;
   Array2D<int> ess_bdr;
   Array<int> ess_bdr_filter;
   std::unique_ptr<Coefficient> vforce_cf;

   switch (problem)
   {
      case Problem::HeatSink:
         mesh = mesh.MakeCartesian2D(4, 4, mfem::Element::Type::QUADRILATERAL, true, 1.0,
                                     1.0);
         MarkBoundary(
            mesh,
         [](const Vector &x) {return std::fabs(x[0]-0.5) < 0.25 && x[1] > 0.5; },
         1);
         MarkBoundary(
            mesh,
         [](const Vector &x) {return !(std::fabs(x[0]-0.5) < 0.25 && x[1] > 0.5); },
         2);
         mesh.SetAttributes();
         ess_bdr.SetSize(1, 2);
         ess_bdr_filter.SetSize(2);
         ess_bdr = 0;
         ess_bdr(0, 0) = 1;
         ess_bdr_filter = 0;

         vforce_cf.reset(new ConstantCoefficient(1.0));
         solfile << "HeatSink-";
         solfile2 << "HeatSink-";
         meshfile << "HeatSink";
         paraview_folder << "HeatSink";
         break;
      default:
         mfem_error("Undefined problem.");
   }

   int dim = mesh.Dimension();
   double h = std::pow(mesh.GetElementVolume(0), 1.0 / dim);

   // 3. Refine the mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
      h *= 0.5;
   }

   // 4. Define the necessary finite element spaces on the mesh.
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
   mfem::out << "Number of state unknowns: " << state_size << std::endl;
   mfem::out << "Number of filter unknowns: " << filter_size << std::endl;
   mfem::out << "Number of control unknowns: " << control_size << std::endl;
   // 5. Set the initial guess for ρ.
   SIMPProjector simp_rule(exponent, rho_min);
   HelmholtzFilter filter(filter_fes, epsilon, ess_bdr_filter);
   SigmoidDesignDensity density(control_fes, filter, filter_fes, vol_fraction);

   ConstantCoefficient kappa_cf(kappa);
   ParametrizedDiffusionEquation diffusion(state_fes,
                                           density.GetFilteredDensity(), simp_rule, kappa_cf, *vforce_cf, ess_bdr);

   TopOptProblem optprob(diffusion.GetLinearForm(), diffusion, density, false,
                         true);


   meshfile << "-" << ref_levels;
   solfile << ref_levels << "-";
   solfile2 << ref_levels << "-";
   solfile << "PMD-0.gf";
   solfile2 << "PMD-f.gf";
   meshfile << ".mesh";

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
      sout_SIMP.precision(8);
      sout_SIMP << "solution\n" << mesh << *designDensity_gf
                << "window_title 'Design density r(ρ̃) - PMD "
                << problem << "'\n"
                << "keys Rjl***************\n"
                << flush;
      sout_r.open(vishost, visport);
      sout_r.precision(8);
      sout_r << "solution\n" << mesh << *rho_gf
             << "window_title 'Raw density ρ - PMD "
             << problem << "'\n"
             << "keys Rjl***************\n"
             << flush;
   }
   std::unique_ptr<ParaViewDataCollection> pd;
   if (paraview)
   {
      pd.reset(new ParaViewDataCollection("TopPMD", &mesh));
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

   mfem::ParaViewDataCollection paraview_dc("ex37", &mesh);
   ofstream mesh_ofs(meshfile.str().c_str());
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   // 11. Iterate
   GridFunction &grad(optprob.GetGradient()), &psi(density.GetGridFunction());
   GridFunction old_grad(&control_fes), old_psi(&control_fes);
   old_psi = psi; old_grad = grad;

   LinearForm diff_rho_form(&control_fes);
   std::unique_ptr<Coefficient> diff_rho(optprob.GetDensityDiffCoeff(old_psi));
   diff_rho_form.AddDomainIntegrator(new DomainLFIntegrator(*diff_rho));

   int k;
   double step_size;
   double compliance = optprob.Eval();
   double old_compliance;
   optprob.UpdateGradient();
   out << std::setw(10) << "Volume" << ",\t"
       << std::setw(10) << "Compliance" << ",\t"
       << std::setw(10) << "Stationarity" << ",\t"
       << std::setw(10) << "Re-eval" << ",\t"
       << std::setw(10) << "Step Size" << ",\t"
       << std::setw(10) << "Stationarity-Bregman"
       << std::endl;
   for (k = 1; k <= max_it; k++)
   {
      diff_rho_form.Assemble();
      old_psi -= psi;
      old_grad -= grad;
      // Compute Step size
      if (k == 1) { step_size = 1.0; }
      else
      {
         step_size = std::fabs(diff_rho_form(old_psi)  / diff_rho_form(old_grad));
      }

      // Store old data
      old_compliance = compliance;
      old_psi = psi;
      old_grad = grad;

      // Step and upate gradient
      int num_check = Step_Armijo(optprob, compliance, c1, step_size);
      compliance = optprob.GetValue();
      optprob.UpdateGradient();

      // Visualization
      if (glvis_visualization)
      {
         designDensity_gf->ProjectCoefficient(simp_rule.GetPhysicalDensity(
                                                 density.GetFilteredDensity()));
         rho_gf->ProjectCoefficient(density.GetDensityCoefficient());
         sout_SIMP << "solution\n" << mesh << *designDensity_gf
                   << flush;
         sout_r << "solution\n" << mesh << *rho_gf
                << flush;
      }
      if (paraview)
      {
         pd->SetCycle(k);
         pd->SetTime(k);
         pd->Save();
      }

      // Check convergence
      double stationarityError = density.StationarityErrorL2(grad);
      double stationarityError_bregman = density.StationarityError(grad);

      out << std::setw(10) << density.GetVolume() << ",\t"
          << std::setw(10) << optprob.GetValue() << ",\t"
          << std::setw(10) << stationarityError << ",\t"
          << std::setw(10) << num_check << ",\t"
          << std::setw(10) << step_size << ",\t"
          << std::setw(10) << stationarityError_bregman
          << std::endl;

      if (stationarityError < 1e-05 && std::fabs(old_compliance - compliance) < 5e-05)
      {
         break;
      }
   }
   if (save)
   {
      ofstream sol_ofs(solfile.str().c_str());
      sol_ofs.precision(8);
      sol_ofs << psi;

      ofstream sol_ofs2(solfile2.str().c_str());
      sol_ofs2.precision(8);
      sol_ofs2 << density.GetFilteredDensity();
   }
   out << "Total number of iteration = " << k << std::endl;

   return 0;
}
