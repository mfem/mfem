//                              MFEM Example 37
//
//
// Compile with: make ex37
//
// Sample runs:
//     ex37 -alpha 10
//     ex37 -alpha 10 -pv
//     ex37 -lambda 0.1 -mu 0.1
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


enum LineSearchMethod
{
   ArmijoBackTracking,
   BregmanBBBackTracking
};

enum Problem
{
   Inverter
};

int main(int argc, char *argv[])
{

   // 1. Parse command-line options.
   int ref_levels = 3;
   int order = 1;
   double alpha = 1.0;
   double epsilon = 1e-2;
   double vol_fraction = 0.5;
   int max_it = 1e3;
   double itol = 1e-3;
   double ntol = 1e-6;
   double rho_min = 1e-6;
   double exponent = 3.0;
   double lambda = 1.0;
   double mu = 1.0;
   double c1 = 1e-04;
   bool glvis_visualization = true;
   bool save = true;

   ostringstream solfile, solfile2, meshfile;

   int lineSearchMethod = LineSearchMethod::BregmanBBBackTracking;
   int problem = Problem::Inverter;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem number: 0) Inverter.");
   args.AddOption(&lineSearchMethod, "-lm", "--line-method",
                  "Line Search Method: 0) BackTracking, 1) BregmanBB, 2) BregmanBB + BackTracking.");
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
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

   Mesh mesh;
   Array2D<int> ess_bdr;
   Array<int> input_bdr, output_bdr;
   Array<int> ess_bdr_filter;
   double input_spring, output_spring;
   Vector input_direction, output_direction;
   string mesh_file;
   switch (problem)
   {
      case Problem::Inverter:
      //                      o o o o o o o o o o o o roller (4)
      //                      -----------------------
      // input (0.1cm) (2) -> |                     | <- output (0.1cm) (3)
      //                      |                     |
      //    fixed (0.1cm) (1) |                     |
      //                      -----------------------
         mesh = mesh.MakeCartesian2D(20, 10, mfem::Element::Type::QUADRILATERAL, true,
                                     2.0,
                                     1.0);
         input_spring = 0.1;
         output_spring = 0.1;
         input_direction.SetSize(2);
         output_direction.SetSize(2);
         input_direction[0] = 1.0;
         output_direction[0] = -1.0;
         ess_bdr.SetSize(3, 5); ess_bdr_filter.SetSize(5);
         input_bdr.SetSize(5); output_bdr.SetSize(5);
         ess_bdr = 0; ess_bdr_filter = 0;
         input_bdr = 0; output_bdr = 0;
         ess_bdr(1, 3) = 1; // roller - y directional fixed
         ess_bdr(2, 0) = 1; // fixed
         input_bdr[1] = 1; output_bdr[2] = 1;
         for (int i = 0; i<mesh.GetNBE(); i++)
         {
            Element * be = mesh.GetBdrElement(i);
            Array<int> vertices;
            be->GetVertices(vertices);

            double * coords1 = mesh.GetVertex(vertices[0]);
            double * coords2 = mesh.GetVertex(vertices[1]);

            Vector fc(2);
            fc(0) = 0.5*(coords1[0] + coords2[0]);
            fc(1) = 0.5*(coords1[1] + coords2[1]);

            if (fabs(fc(0) - 0.0) < 1e-10 & fc(1) - 0.1 < 0)
            {
               // left bottom -> Fixed
               be->SetAttribute(1);
            }
            else if (fabs(fc(0) - 0.0) < 1e-10 & fc(1) - 0.9 > 0)
            {
               // left top -> input
               be->SetAttribute(2);
            }
            else if (fabs(fc(0) - 2.0) < 1e-10 & fc(1) - 0.9 > 0)
            {
               // right top -> output
               be->SetAttribute(3);
            }
            else if (fabs(fc(1) - 1.0) < 1e-10)
            {
               // top -> roller
               be->SetAttribute(4);
            }
            else
            {
               // free boundary
               be->SetAttribute(5);
            }
         }
         solfile << "Inverter-";
         solfile2 << "Inverter-";
         meshfile << "Inverter";
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
   epsilon = 4 * h;

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
   GridFunction psi(&control_fes);
   GridFunction psi_old(&control_fes);
   psi = inv_sigmoid(vol_fraction);
   psi_old = inv_sigmoid(vol_fraction);

   // ρ = sigmoid(ψ)
   MappedGridFunctionCoefficient rho(&psi, sigmoid);
   // Interpolation of ρ = sigmoid(ψ) in control fes (for ParaView output)
   GridFunction rho_gf(&control_fes);
   // ρ - ρ_old = sigmoid(ψ) - sigmoid(ψ_old)
   DiffMappedGridFunctionCoefficient succ_diff_rho(&psi, &psi_old, sigmoid);
   LinearForm succ_diff_rho_form(&control_fes);
   succ_diff_rho_form.AddDomainIntegrator(new DomainLFIntegrator(succ_diff_rho));

   // 9. Define some tools for later
   ConstantCoefficient one(1.0);
   GridFunction zero_gf(&control_fes);
   zero_gf = 0.0;
   LinearForm vol_form(&control_fes);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble();
   double domain_volume = vol_form.Sum();
   const double target_volume = domain_volume * vol_fraction;
   ConstantCoefficient lambda_cf(lambda), mu_cf(mu);
   CompliantMechanism obj(&lambda_cf, &mu_cf, epsilon,
                          &rho, target_volume,
                          ess_bdr,
                          input_bdr, output_bdr,
                          input_spring, output_spring,
                          input_direction, output_direction,
                          &state_fes,
                          &filter_fes, exponent, rho_min);
   obj.SetGridFunction(&psi);
   LineSearchAlgorithm *lineSearch;
   switch (lineSearchMethod)
   {
      case LineSearchMethod::ArmijoBackTracking:
         lineSearch = new BackTracking(obj, alpha, 2.0, c1, 10, infinity());
         solfile << "EXP-";
         solfile2 << "EXP-";
         break;
      case LineSearchMethod::BregmanBBBackTracking:
         lineSearch = new BackTrackingLipschitzBregmanMirror(
            obj, succ_diff_rho_form, *(obj.Gradient()), psi, c1, 1.0, 1e-10, infinity());
         solfile << "BB-";
         solfile2 << "BB-";
         break;
      default:
         mfem_error("Undefined linesearch method.");
   }
   solfile << "0.gf";
   solfile2 << "f.gf";
   meshfile << ".mesh";

   MappedGridFunctionCoefficient &designDensity = obj.GetDesignDensity();
   GridFunction designDensity_gf(&filter_fes);
   designDensity_gf = pow(vol_fraction, exponent);
   // designDensity_gf.ProjectCoefficient(designDensity);

   GridFunction &u = *obj.GetDisplacement();
   GridFunction &rho_filter = *obj.GetFilteredDensity();

   // 10. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_SIMP, sout_r;
   if (glvis_visualization)
   {
      sout_SIMP.open(vishost, visport);
      sout_SIMP.precision(8);
      sout_SIMP << "solution\n" << mesh << designDensity_gf
                << "window_title 'Design density r(ρ̃) - MD "
                << problem << " " << lineSearchMethod << "'\n"
                << "keys Rjl***************\n"
                << flush;
      sout_r.open(vishost, visport);
      sout_r.precision(8);
      sout_r << "solution\n" << mesh << rho_gf
             << "window_title 'Raw density ρ - MD "
             << problem << " " << lineSearchMethod << "'\n"
             << "keys Rjl***************\n"
             << flush;
   }

   mfem::ParaViewDataCollection paraview_dc("ex37", &mesh);
   ofstream mesh_ofs(meshfile.str().c_str());
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   // 11. Iterate
   obj.Eval();
   GridFunction d(&control_fes);
   int k;
   for (k = 1; k <= max_it; k++)
   {
      // mfem::out << "\nStep = " << k << std::endl;

      d = *obj.Gradient();
      d.Neg();
      double compliance = lineSearch->Step(psi, d);
      double norm_increment = zero_gf.ComputeLpError(1, succ_diff_rho);
      psi_old = psi;

      // mfem::out << "volume fraction = " <<  obj.GetVolume() / domain_volume <<
      //           std::endl;
      mfem::out <<  ", " << compliance << ", ";
      mfem::out << norm_increment << std::endl;

      if (glvis_visualization)
      {
         designDensity_gf.ProjectCoefficient(designDensity);
         sout_SIMP << "solution\n" << mesh << designDensity_gf
                   << flush;
         rho_gf.ProjectCoefficient(rho);
         sout_r << "solution\n" << mesh << rho_gf
                << flush;
      }
      if (norm_increment < itol)
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
      sol_ofs2 << *obj.GetFilteredDensity();
   }
   out << "Total number of iteration = " << k << std::endl;
   delete lineSearch;

   return 0;
}
