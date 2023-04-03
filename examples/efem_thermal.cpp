//                              MFEM Example 35
//
//
// Compile with: make ex35
//
// Sample runs:
//     ex35 -alpha 10
//     ex35 -lambda 0.1 -mu 0.1
//     ex35 -r 5 -o 2 -alpha 5.0 -epsilon 0.01 -mi 50 -mf 0.5 -tol 1e-5
//     ex35 -r 6 -o 1 -alpha 10.0 -epsilon 0.01 -mi 50 -mf 0.5 -tol 1e-5
//
//
// Description: This example code demonstrates the use of MFEM to solve a
//              density-filtered [3] topology optimization problem. The
//              objective is to minimize the thermal compliance
//
//                  minimize ∫_Ω f u dx over u ∈ H¹(Ω) and ρ ∈ L²(Ω)
//
//                  subject to
//
//                    -∇⋅(r(ρ̃)∇ u) = f       in Ω + BCs
//                    -ϵ²Δρ̃ + ρ̃ = ρ          in Ω + Neumann BCs
//                    0 ≤ ρ ≤ 1              in Ω
//                    u ≤ 1                  in Ω
//                    ∫_Ω ρ dx = θ vol(Ω)
//
//              Here, r(ρ̃) = ρ₀ + ρ̃³ (1-ρ₀) is the solid isotropic material
//              penalization (SIMP) law, ϵ > 0 is the design length scale,
//              and 0 < θ < 1 is the volume fraction. Note that we have
//
//              More specifically, we have f = 0 in an insulated rectagular
//              domain Ω = (0, 20) x (0, 20) where the left middle section
//              {x = 0} x (9, 11) is held at temperature 1.
//
//                                INSULATED
//                       ---------------------------  20
//                       |                         |
//                       |                         |
//                       * -                       |
//                 u = 1 * |  2                    |
//                       * -                       |
//                       |                         |
//                       |                         |
//                       ---------------------------   0
//                       0                         20
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
// [2] Keith, B. and Surowiec, T. (2023) The entropic finite element method
//     (in preparation).
// [3] Lazarov, B. S., & Sigmund, O. (2011). Filters in topology optimization
//     based on Helmholtz‐type differential equations. International Journal
//     for Numerical Methods in Engineering, 86(6), 765-781.

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include "efem.hpp"

using namespace std;
using namespace mfem;
/**
 * ---------------------------------------------------------------
 *                      ALGORITHM PREAMBLE
 * ---------------------------------------------------------------
 *
 *  The Lagrangian for this problem is
 *
 *          L(u,ρ,ρ̃,w,w̃) = (f,u) + (r(ρ̃)∇u, ∇w) - (f,w) + ϵ^2(∇ρ̃, ∇w̃) + (ρ̃ - ρ, w̃)
 *                       + α_u * D≤(u, uk) + α_ρ * (D≥(ρ, ρk) + D≤(ρ, ρk))
 *
 *  where
 *
 *    r(ρ̃) = ρ₀ + ρ̃³ (1 - ρ₀)            (SIMP rule)
 *
 *    D≥(x, y) = ∫ xlog(x/y) - (x - y)   (Lower Bound, away from 0)
 *
 *    D≤(x, y) = D≥(1 - x, 1 - y)        (Upper Bound, away from 1)
 *
 * ---------------------------------------------------------------
 *
 *  Discretization choices:
 *
 *     u ∈ Vh ⊂ H¹ (order p)
 *     w ∈ Vh ⊂ H¹ (order p)
 *     ρ̃ ∈ Vl ⊂ H¹ (order p)
 *     w̃ ∈ Vl ⊂ H¹ (order p)
 *     ψ ∈ Wl ⊂ L² (order p - 1)
 *
 *  where ρ = sigmoid(ψ) so that 0≤ρ≤1 is strongly enforced
 *
 * ---------------------------------------------------------------
 *                          ALGORITHM
 * ---------------------------------------------------------------
 *
 *  Update ψ with projected mirror descent via the following algorithm.
 *
 *  0. Initialize density field ψ = sigmoid⁻¹(θ) so that ∫ρ = ∫sigmoid(ψ) = θ|Ω|
 *
 *  While not converged:
 *
 *     1. Solve filter equation ∂_w̃ L = 0; i.e.,
 *
 *           (ϵ² ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)   ∀ v ∈ Vl.
 *
 *     2. Solve primal problem ∂_w L = 0; i.e.,
 *
 *                    (r(ρ̃) ∇u, ∇v) = (f,v)   ∀ v ∈ Vh.
 *
 *     3. Solve dual problem ∂_u L = 0; i.e.,
 *
 *                    (r(ρ̃) ∇w, ∇v) = (f,v) + (log(u/uk), v)    ∀ v ∈ Vh.
 *
 *        NOTE: When there is no constraint u≤1, then w = u.
 *              In that case, we do not have to solve the dual problem.
 *
 *     4. Solve for filtered gradient ∂_ρ̃ L = 0; i.e.,
 *
 *      (ϵ² ∇ w̃ , ∇ v ) + (w̃ ,v) = ( r'(ρ̃) (∇ u ⋅ ∇ w), v)   ∀ v ∈ Vl.
 *
 *     5. Set intermediate variable ψ⋆ = ψ - α w̃.
 *
 *     6. Update ψ by ψ = proj(ψ⋆) = ψ⋆ + c where c is chosen to be
 *
 *                ∫ sigmoid(ψ⋆ + c) = θ|Ω|.
 *
 *  end
 *
 */


/**
 * @brief log(max(a, tol)) - log(max(b, tol))
 *
 */
class SafeLogDiffGridFunctionCoefficient : public
   SafeLogarithmicGridFunctionCoefficient
{
private:
   SafeLogarithmicGridFunctionCoefficient
   *gf_other; // gridfunction log(b) to be subtracted

public:

   /**
    * @brief log(max(a, tol)) - log(max(b, tol))
    *
    */
   SafeLogDiffGridFunctionCoefficient(GridFunction *a,
                                      GridFunction *b, const double tolerance):
      SafeLogarithmicGridFunctionCoefficient(a, tolerance),
      gf_other(new SafeLogarithmicGridFunctionCoefficient(b, tolerance)) {}

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      return SafeLogarithmicGridFunctionCoefficient::Eval(T, ip)
             - gf_other->Eval(T, ip);
   }
};

/**
 * @brief
 *
 */
class SIMPDerEnergyCoefficient : public GridFunctionCoefficient
{
private:
   SIMPDerCoefficient *r_prime_rho;
   GradientGridFunctionCoefficient *gradu;
   GradientGridFunctionCoefficient *gradw;
   Vector gradu_val, gradw_val;
   double a;
public:

   SIMPDerEnergyCoefficient(GridFunction *rho_filter, const double exponent,
                            const double rho_min, GridFunction *u,
                            GridFunction *w):GridFunctionCoefficient()
   {
      r_prime_rho = new SIMPDerCoefficient(rho_filter, exponent, rho_min);
      gradu = new GradientGridFunctionCoefficient(u);
      gradw = new GradientGridFunctionCoefficient(w);
      const int dim = u->FESpace()->GetMesh()->Dimension();
      gradu_val = Vector(dim);
      gradw_val = Vector(dim);
   }

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      gradu->Eval(gradu_val, T, ip);
      gradw->Eval(gradw_val, T, ip);
      return -a*r_prime_rho->Eval(T, ip)*(gradu_val*gradw_val);
   }

   void SetAlpha(const double alpha)
   {
      a = alpha;
   }
};

inline void clip(GridFunction &gf, const double lower, const double upper)
{
   for (auto &x : gf)
   {
      x = min(max(x, lower), upper);
   }
}

int main(int argc, char *argv[])
{

   // 0 - 1. Parse command-line options.
   int ref_levels = 4; // The number of initial mesh refinement
   int order = 2; // Polynomial order p. State - p, Design - p - 1, Filter - p
   bool visualization = true;
   double alpha0 = 1.0; // Update rule
   double epsilon = 0.01; // Design parameter, ϵ.
   double mass_fraction = 0.3; // mass fraction, θ.
   int max_it = 1e2; // projected mirror gradient maximum iteration
   double tol = 1e-4; // Projected mirror gradient tolerance
   double rho_min = 1e-6; // SIMP ρ0
   double exponent = 3; // SIMP exponent

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&alpha0, "-alpha", "--alpha-step-length",
                  "Step length for gradient descent.");
   args.AddOption(&epsilon, "-epsilon", "--epsilon-thickness",
                  "epsilon phase field thickness");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&tol, "-tol", "--tol",
                  "Exit tolerance for ρ ");
   args.AddOption(&mass_fraction, "-mf", "--mass-fraction",
                  "Mass fraction for diffusion coefficient.");
   args.AddOption(&rho_min, "-rmin", "--rho-min",
                  "Minimum of density coefficient.");
   args.AddOption(&exponent, "-exp", "--exponent",
                  "SIMP exponent.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh = Mesh::MakeCartesian2D(20,20,mfem::Element::Type::QUADRILATERAL,true,
                                     20.0,20.0);

   int dim = mesh.Dimension();

   // 2. Set BCs.
   for (int i = 0; i<mesh.GetNBE(); i++)
   {
      Element * be = mesh.GetBdrElement(i);
      Array<int> vertices;
      be->GetVertices(vertices);

      double * coords1 = mesh.GetVertex(vertices[0]);
      double * coords2 = mesh.GetVertex(vertices[1]);

      Vector center(2);
      center(0) = 0.5*(coords1[0] + coords2[0]);
      center(1) = 0.5*(coords1[1] + coords2[1]);

      if (abs(center(1) - 10) < 1 && center(0) < 1e-12)
      {
         // the left center
         be->SetAttribute(1);
      }
      else
      {
         // all other boundaries
         be->SetAttribute(2);
      }
   }
   mesh.SetAttributes();

   // 3. Refine the mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   const int filter_order = order - 1;


   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection state_fec(order, dim); // space for u
   H1_FECollection filter_fec(order, dim); // space for ρ̃
   L2_FECollection control_fec(order-1, dim); // space for ρ
   FiniteElementSpace state_fes(&mesh, &state_fec);
   FiniteElementSpace filter_fes(&mesh, &filter_fec);
   FiniteElementSpace control_fes(&mesh, &control_fec);

   int state_size = state_fes.GetTrueVSize();
   int control_size = control_fes.GetTrueVSize();
   int filter_size = filter_fes.GetTrueVSize();
   cout << "Number of state unknowns: " << state_size << endl;
   cout << "Number of filter unknowns: " << filter_size << endl;
   cout << "Number of control unknowns: " << control_size << endl;

   // 5. Set the initial guess for ρ.
   GridFunction u(&state_fes);
   GridFunction w(&state_fes);
   GridFunction psi(&control_fes);
   GridFunction rho_filter(&filter_fes);
   GridFunction w_filter(&filter_fes);

   ExponentialGridFunctionCoefficient rho(&psi);
   u = 1.0;
   w = 1.0;
   rho_filter = 0.0;
   w_filter = 0.0;
   psi = invsigmoid(mass_fraction);

   GridFunction u_old(u);
   GridFunction psi_old(psi);

   // 6. Set-up the physics solver.

   // 6 - 1. State problem LHS
   int maxat = mesh.bdr_attributes.Max();
   Array<int> ess_bdr(maxat);
   ess_bdr = 0;
   ess_bdr[0] = 1;
   SIMPCoefficient r_rho_filter(&rho_filter, exponent, rho_min);
   BilinearForm diffForm(&state_fes);
   diffForm.AddDomainIntegrator(new DiffusionIntegrator(r_rho_filter));
   EllipticSolver state_solver(&state_fes, &diffForm, ess_bdr);

   // 6 - 2. State problem RHS
   LinearForm state_RHS(&state_fes); // (f,v). Assemble only once.
   ConstantCoefficient zero(0.0);
   state_RHS.AddDomainIntegrator(new DomainLFIntegrator(zero));
   state_RHS.Assemble(); // Assemble only once
   LinearForm state_dual_RHS(&state_fes);
   state_dual_RHS.AddDomainIntegrator(new DomainLFIntegrator(zero));
   SafeLogDiffGridFunctionCoefficient logdiffu(&u, &u_old, 1e-12);
   state_dual_RHS.AddDomainIntegrator(new DomainLFIntegrator(logdiffu));

   // 6 - 3. Filter problem LHS
   ess_bdr = 0;
   BilinearForm epsDiffForm(&state_fes);
   ConstantCoefficient eps_squared(epsilon*epsilon);
   epsDiffForm.AddDomainIntegrator(new DiffusionIntegrator(eps_squared));
   EllipticSolver filter_solver(&filter_fes, &epsDiffForm, ess_bdr);

   // 6 - 4. Filter problem RHS
   LinearForm filter_RHS(&filter_fes);
   filter_RHS.AddDomainIntegrator(new DomainLFIntegrator(rho));
   LinearForm filter_dual_RHS(&filter_fes);
   SIMPDerEnergyCoefficient r_energy(&rho_filter, exponent, rho_min, &u, &w);
   filter_dual_RHS.AddDomainIntegrator(new DomainLFIntegrator(r_energy));
   EllipticSolver filter_Solver(&filter_fes, &epsDiffForm, ess_bdr);

   LinearForm volForm(&filter_fes);
   ConstantCoefficient one(1.0);
   volForm.AddDomainIntegrator(new DomainLFIntegrator(one, 0, 0));
   volForm.Assemble();
   const double vol = volForm.Sum();
   SigmoidDensityProjector volProj(&control_fes, mass_fraction, vol);

   BilinearForm invMass(&control_fes);
   invMass.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator()));
   invMass.Assemble();
   SparseMatrix invM;
   Array<int> empty;
   invMass.FormSystemMatrix(empty,invM);
   GridFunctionCoefficient w_filter_cf(&w_filter);
   LinearForm w_filter_load(&control_fes);
   w_filter_load.AddDomainIntegrator(new DomainLFIntegrator(w_filter_cf));

   // 10. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_u,sout_r,sout_rho;
   if (visualization)
   {
      sout_u.open(vishost, visport);
      sout_rho.open(vishost, visport);
      sout_r.open(vishost, visport);
      sout_u.precision(8);
      sout_rho.precision(8);
      sout_r.precision(8);
   }

   mfem::ParaViewDataCollection paraview_dc("Elastic_compliance", &mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("displacement",&u);
   paraview_dc.RegisterField("density",&rho);
   paraview_dc.RegisterField("filtered_density",&rho_filter);

   // 11. Iterate
   double c0 = 0.0;
   for (int k = 1; k < max_it; k++)
   {
      const double alpha = alpha0*k;

      cout << "\nStep = " << k << endl;

      // Step 1 - Filter solve
      // Solve (ϵ^2 ∇ ρ̃, ∇ v) + (ρ̃,v) = (ρ,v)
      filter_RHS.Assemble();
      filter_solver.Solve(&filter_RHS, &rho_filter);


      // Step 2 - Primal solve
      // Solve (r(ρ̃) ∇ u, ∇ v) = (f, v)
      // No need for assemble state_RHS because it does not change.
      state_solver.Solve(&state_RHS, &u);

      // Step 3 - Dual solve
      // Solve (r(ρ̃) ∇ u, ∇ v) = (f, v) + (log(u/uk), v)
      state_dual_RHS.Assemble();
      state_solver.Solve(&state_dual_RHS, &w);

      // Step 4 - Dual filter solve
      // Solve (ϵ^2 ∇ w̃, ∇ v) + (w̃, v) = -(r'(ρ̃)(∇ u ⋅ ∇ w), v)
      filter_dual_RHS.Assemble();
      filter_solver.Solve(&filter_dual_RHS, &w_filter);

      // Step 5 - Get ψ⋆ = ψ - w̃
      w_filter_load.Assemble();
      psi_old = psi;
      invM.AddMult(w_filter_load, psi, -alpha);

      // Step 6 - ψ = proj(ψ⋆)
      clip(psi, -100,
           100); // bound psi so that 0≈sigmoid(-100) < rho < sigmoid(100)≈1
      volProj.Apply(psi, 20);

      if (visualization)
      {
         sout_u << "solution\n" << mesh << u
                << "window_title 'Displacement u'" << flush;

         GridFunction rho_gf()
         sout_rho << "solution\n" << mesh << rho
                  << "window_title 'Control variable ρ'" << flush;

         sout_r << "solution\n" << mesh << rho_filter
                << "window_title 'Design density r(ρ̃)'" << flush;

         paraview_dc.SetCycle(k);
         paraview_dc.SetTime((double)k);
         paraview_dc.Save();
      }

      if (norm_reduced_gradient < tol)
      {
         break;
      }
   }

   delete ElasticitySolver;
   delete FilterSolver;

   return 0;
}