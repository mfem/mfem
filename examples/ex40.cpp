//                              MFEM Example 37
//
//
// Compile with: make ex40
//
// Sample runs:
//     ex40 -alpha 10
//     ex40 -alpha 10 -pv
//     ex40 -lambda 0.1 -mu 0.1
//     ex40 -o 2 -alpha 5.0 -mi 50 -vf 0.4 -ntol 1e-5
//     ex40 -r 6 -o 1 -alpha 25.0 -epsilon 0.02 -mi 50 -ntol 1e-5
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
//              penalization (SIMP) law, C is the elasticity tensor for an
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
#include "ex37.hpp"

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

int main(int argc, char *argv[])
{

   // 1. Parse command-line options.
   int ref_levels = 7;
   int order = 1;
   double alpha = 1.0;
   double epsilon = 0.02;
   double vol_fraction = 0.5;
   int max_it = 1e3;
   double itol = 1e-2;
   double ntol = 1e-6;
   double rho_min = 1e-6;
   double exponent = 3.0;
   double lambda = 1.0;
   double mu = 1.0;
   double c1 = 1e-04;
   double c2 = 0.9;
   bool glvis_visualization = true;
   bool paraview_output = false;
   double mv = 0.2;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&alpha, "-alpha", "--alpha-step-length",
                  "Step length for gradient descent.");
   args.AddOption(&epsilon, "-epsilon", "--epsilon-thickness",
                  "Length scale for ρ.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&ntol, "-ntol", "--rel-tol",
                  "Normalized exit tolerance.");
   args.AddOption(&itol, "-itol", "--abs-tol",
                  "Increment exit tolerance.");
   args.AddOption(&vol_fraction, "-vf", "--volume-fraction",
                  "Volume fraction for the material density.");
   args.AddOption(&lambda, "-lambda", "--lambda",
                  "Lamé constant λ.");
   args.AddOption(&mu, "-mu", "--mu",
                  "Lamé constant μ.");
   args.AddOption(&rho_min, "-rmin", "--psi-min",
                  "Minimum of density coefficient.");
   args.AddOption(&glvis_visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview_output, "-pv", "--paraview", "-no-pv",
                  "--no-paraview",
                  "Enable or disable ParaView output.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

   // 3. Problem definition
   const int numel = std::pow(2, ref_levels);
   Mesh mesh = Mesh::MakeCartesian2D(3*numel,numel,
                                     mfem::Element::Type::QUADRILATERAL,true,
                                     3*numel,numel);
   int dim = mesh.Dimension();
   int maxat = mesh.bdr_attributes.Max();
   Array<int> ess_bdr(maxat);
   ess_bdr = 0;
   ess_bdr[3] = 1;
   Array<int> ess_bdr_filter(maxat);
   ess_bdr_filter = 0;

   Vector center(2); center(0) = 2.9; center(1) = 0.5;
   Vector force(2); force(0) = 0.0; force(1) = -1.0;
   double r = 0.05;
   VolumeForceCoefficient vforce_cf(r,center,force);

   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection state_fec(order, dim); // space for u
   H1_FECollection filter_fec(order, dim); // space for ρ̃
   L2_FECollection control_fec(order-1, dim,
                               BasisType::GaussLobatto); // space for ψ
   FiniteElementSpace state_fes(&mesh, &state_fec,dim);
   FiniteElementSpace filter_fes(&mesh, &filter_fec);
   FiniteElementSpace control_fes(&mesh, &control_fec);

   int state_size = state_fes.GetTrueVSize();
   int control_size = control_fes.GetTrueVSize();
   int filter_size = filter_fes.GetTrueVSize();
   mfem::out << "Number of state unknowns: " << state_size << std::endl;
   mfem::out << "Number of filter unknowns: " << filter_size << std::endl;
   mfem::out << "Number of control unknowns: " << control_size << std::endl;

   // 5. Set the initial guess for ρ.
   GridFunction rho(&control_fes);
   rho = vol_fraction;
   GridFunction rho_old(rho);
   GridFunctionCoefficient rho_cf(&rho);

   GridFunction frho(&filter_fes);
   frho = vol_fraction;
   MappedGridFunctionCoefficient SIMP_cf(&frho, [rho_min,
   exponent](double x) {return simp(x, rho_min, exponent, 1.0); });

   GridFunction u(&state_fes);
   u = 0.0;

   ConstantCoefficient one(1.0), zero(0.0), eps_cf(epsilon), lambda_cf(lambda), mu_cf(mu);
   StrainEnergyDensityCoefficient strainEnergyDensity_cf(&lambda_cf, &mu_cf, &u, &frho, rho_min, exponent);
   
   ProductCoefficient SIMP_lam(lambda, SIMP_cf), SIMP_mu(mu, SIMP_cf);


   // 10. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_r;
   if (glvis_visualization)
   {
      GridFunction designDensity_gf(&control_fes);
      designDensity_gf.ProjectCoefficient(SIMP_cf);
      sout_r.open(vishost, visport);
      sout_r.precision(8);
      sout_r << "solution\n" << mesh << designDensity_gf
             << "window_title 'Design density r(ρ̃)'\n"
             << "keys Rj***************\n"
             << flush;
   }

   mfem::ParaViewDataCollection paraview_dc("ex40", &mesh);
   if (paraview_output)
   {
      // rho_gf.ProjectCoefficient(rho);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.RegisterField("displacement",&u);
      paraview_dc.RegisterField("density",&rho);
      paraview_dc.RegisterField("filtered_density",&frho);
      paraview_dc.Save();
   }
   // 11. Iterate
   for (int k = 1; k <= max_it; k++)
   {
      mfem::out << "\nStep = " << k << std::endl;


      mfem::out << "volume fraction = " <<  obj.GetVolume() / domain_volume <<
                std::endl;
      mfem::out << "compliance = " <<  compliance << std::endl;
      mfem::out << "current step size = " <<  lineSearch.GetStepSize() << std::endl;

      // Compute ||ρ - ρ_old|| in control fes.
      // double norm_increment = zerogf.ComputeL1Error(succ_diff_rho);

      double norm_increment = rho_old.Normlinf();
      
      

      mfem::out << "norm of the increment = " << norm_increment << endl;

      if (glvis_visualization)
      {
         GridFunction designDensity_gf(&control_fes);
         designDensity_gf.ProjectCoefficient(SIMP_cf);
         sout_r << "solution\n" << mesh << designDensity_gf
                << flush;

         ostringstream sol_name;
         sol_name << "sol-" << k << ".gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(8);
         sol_ofs << designDensity_gf;
      }

      if (paraview_output)
      {
         paraview_dc.SetCycle(k);
         paraview_dc.SetTime((double)k);
         paraview_dc.Save();
      }

      if (norm_increment < itol)
      {
         break;
      }
   }

   return 0;
}
