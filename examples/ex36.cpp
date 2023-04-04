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
//              objective is to minimize the compliance
//
//                  minimize ∫_Ω f⋅u dx over u ∈ [H¹(Ω)]² and ρ ∈ L²(Ω)
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
// [2] Keith, B. and Surowiec, T. (2023) The entropic finite element method
//     (in preparation).
// [3] Lazarov, B. S., & Sigmund, O. (2011). Filters in topology optimization
//     based on Helmholtz‐type differential equations. International Journal
//     for Numerical Methods in Engineering, 86(6), 765-781.

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include "ex35.hpp"

/**
 * @brief Nonlinear projection of 0 < τ < 1 onto the subspace
 *        ∫_Ω τ dx = θ vol(Ω) as follows.
 *
 *        1. Compute the root of the R → R function
 *            f(c) = ∫_Ω expit(lnit(τ) + c) dx - θ vol(Ω)
 *        2. Set τ ← expit(lnit(τ) + c).
 *
 */
// void projit(GridFunction &tau, double &c, LinearForm &vol_form,
//             double volume_fraction, double tol=1e-12, int max_its=10)
// {
//    GridFunction ftmp(tau.FESpace());
//    GridFunction dftmp(tau.FESpace());
//    for (int k=0; k<max_its; k++)
//    {
//       // Compute f(c) and dfdc(c)
//       for (int i=0; i<tau.Size(); i++)
//       {
//          ftmp[i]  = expit(lnit(tau[i]) + c) - volume_fraction;
//          dftmp[i] = dexpitdx(lnit(tau[i]) + c);
//       }
//       double f = vol_form(ftmp);
//       double df = vol_form(dftmp);

//       double dc = -f/df;
//       c += dc;
//       if (abs(dc) < tol) { break; }
//    }
//    tau = ftmp;
//    tau += volume_fraction;
// }
class MappedGridFunctionCoefficient : public GridFunctionCoefficient
{
private:
   std::__1::function<double(const double)> fun;
public:
   MappedGridFunctionCoefficient(GridFunction *gf,
                                 std::__1::function<double(const double)> fun_):GridFunctionCoefficient(gf),
      fun(fun_) {}

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      return fun(GridFunctionCoefficient::Eval(T, ip));
   }
};
class GridFunctionPlusCoefficient : public GridFunctionCoefficient
{
private:
   Coefficient *cf;
public:
   GridFunctionPlusCoefficient(GridFunction *gf,
                               Coefficient *coeff): GridFunctionCoefficient(gf), cf(coeff) {}
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      const double gf_val = GridFunctionCoefficient::Eval(T, ip);
      const double cf_val = cf->Eval(T, ip);
      return gf_val + cf_val;
   }
};

/**
 * @brief Volume compliant projection operator for ψ
 * 
 * @param psi ψ, where ρ = sigmoid(ψ) 
 * @param target_volume Target ∫ρ
 * @param tol Tolerance for Newton method
 * @param max_its Maximum iteration for Newton method
 */
void projit(GridFunction &psi, double target_volume,
            const double tol=1e-12, const int max_its=30)
{  
   // ρ = sigmoid(ψ).
   MappedGridFunctionCoefficient rho(&psi, [](const double x) {return expit(x);});
   LinearForm rho_form(psi.FESpace()); // ∫ ρ
   rho_form.AddDomainIntegrator(new DomainLFIntegrator(rho));

   // dsigmoid = d(sigmoid(ψ))/dψ
   MappedGridFunctionCoefficient dsigmoid(&psi, [](const double x) {return dexpitdx(x);});
   LinearForm dsigmoid_form(psi.FESpace()); // ∫ dsigmoid(ψ)
   dsigmoid_form.AddDomainIntegrator(new DomainLFIntegrator(dsigmoid));

   // Newton method with respect to f(c) = ∫sigmoid(ψ + c) - θ|Ω|
   //
   //   ψ_new = ψ_old - f(ψ_old) / f'(ψ_old)
   //         = ψ_old - (∫sigmoid(ψ) - θ|Ω|) / ∫dsigmoid(ψ)
   for (int i=0; i<max_its; i++)
   {
      // Compute ∫ρ with updated ψ
      rho_form.Assemble();
      const double f = rho_form.Sum() - target_volume; // ∫sigmoid(ψ) - θ|Ω|
      // Compute ∫dsigmoid(ψ) with updated ψ
      dsigmoid_form.Assemble();
      const double df = dsigmoid_form.Sum(); // ∫dsigmoid(ψ)
      // Newton increment
      const double dc = - f / df;

      // For debugging, put assert here.
      MFEM_ASSERT(isfinite(dc), "Newton increment is not finite.");
      psi += dc;
      // tolerance check
      if (abs(dc) < tol)
      {
         return;
      }
   }
   // If you are here, then Newton method failed to converge
   mfem_error("Projection failed to converge");
}

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
 *     ρ ∈ L² (order p - 1)
 *     ρ̃ ∈ H¹ (order p - 1)
 *     w ∈ V  (order p)
 *     w̃ ∈ H¹ (order p - 1)
 *
 * ---------------------------------------------------------------
 *                          ALGORITHM
 * ---------------------------------------------------------------
 *
 *  Update ρ with projected mirror descent via the following algorithm.
 *
 *  1. Initialize density field 0 < ρ(x) < 1.
 *
 *  While not converged:
 *
 *     2. Solve filter equation ∂_w̃ L = 0; i.e.,
 *
 *           (ϵ² ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)   ∀ v ∈ H¹.
 *
 *     3. Solve primal problem ∂_w L = 0; i.e.,
 *
 *      (λ(ρ̃) ∇⋅u, ∇⋅v) + (2 μ(ρ̃) ε(u), ε(v)) = (f,v)   ∀ v ∈ V,
 *
 *     where λ(ρ̃) := λ r(ρ̃) and  μ(ρ̃) := μ r(ρ̃).
 *
 *     NB. The dual problem ∂_u L = 0 is the same as the primal problem due to symmetry.
 *
 *     4. Solve for filtered gradient ∂_ρ̃ L = 0; i.e.,
 *
 *      (ϵ² ∇ w̃ , ∇ v ) + (w̃ ,v) = (-r'(ρ̃) ( λ(ρ̃) |∇⋅u|² + 2 μ(ρ̃) |ε(u)|²),v)   ∀ v ∈ H¹.
 *
 *     5. Construct gradient G ∈ L²; i.e.,
 *
 *                         (G,v) = (w̃,v)   ∀ v ∈ L².
 *
 *     6. Mirror descent update until convergence; i.e.,
 *
 *                      ρ ← projit(expit(linit(ρ) - αG)),
 *
 *     where
 *
 *          α > 0                            (step size parameter)
 *
 *          expit(x) = eˣ/(1+eˣ)             (sigmoid)
 *
 *          linit(y) = ln(y) - ln(1-y)       (inverse of sigmoid)
 *
 *     and projit is a (compatible) projection operator enforcing ∫_Ω ρ dx = θ vol(Ω).
 *
 *  end
 *
 */

int main(int argc, char *argv[])
{

   // 1. Parse command-line options.
   int ref_levels = 5;
   int order = 2;
   bool visualization = true;
   double alpha0 = 1.0;
   double epsilon = 0.01;
   double mass_fraction = 0.3;
   int max_it = 1e3;
   double tol = 1e-4;
   double rho_min = 1e-3;
   double exponent = 3;

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

   Mesh mesh = Mesh::MakeCartesian2D(1,1,mfem::Element::Type::QUADRILATERAL,true,
                                     1.0,1.0);

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

      if (abs(center(0) - 0.0) < 1e-10)
      {
         // the left edge
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
   ConstantCoefficient zero(0.0);
   ConstantCoefficient one(1.0);

   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection state_fec(order, dim); // space for u
   H1_FECollection filter_fec(order, dim); // space for ρ̃
   L2_FECollection control_fec(order-1, dim,
                               BasisType::GaussLobatto); // space for ρ
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
   GridFunction psi(&control_fes);
   GridFunction rho_filter_old(&filter_fes);
   GridFunction rho_filter(&filter_fes);
   u = 0.0;
   rho_filter = mass_fraction;
   psi = lnit(mass_fraction);
   rho_filter_old = mass_fraction;

   MappedGridFunctionCoefficient rho(&psi, [](const double x) {return expit(x);});

   // 6. Set-up the physics solver.
   int maxat = mesh.bdr_attributes.Max();
   Array<int> ess_bdr(maxat);
   ess_bdr = 0;
   ess_bdr[0] = 1;
   DiffusionSolver * diffusionSolver = new DiffusionSolver();
   diffusionSolver->SetMesh(&mesh);
   diffusionSolver->SetOrder(state_fec.GetOrder());
   diffusionSolver->SetRHSCoefficient(&one);
   diffusionSolver->SetDiffusionCoefficient(&one);
   diffusionSolver->SetMassCoefficient(&one);
   diffusionSolver->SetEssentialBoundary(ess_bdr);
   diffusionSolver->SetupFEM();

   // 7. Set-up the filter solver.
   Array<int> ess_bdr_filter(maxat);
   ess_bdr_filter = 0;
   ConstantCoefficient eps2_cf(epsilon*epsilon);
   DiffusionSolver * filterSolver = new DiffusionSolver();
   filterSolver->SetMesh(&mesh);
   filterSolver->SetOrder(filter_fec.GetOrder());
   filterSolver->SetDiffusionCoefficient(&eps2_cf);
   filterSolver->SetMassCoefficient(&one);
   filterSolver->SetEssentialBoundary(ess_bdr_filter);
   filterSolver->SetupFEM();

   BilinearForm mass(&control_fes);
   mass.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator(one)));
   mass.Assemble();
   SparseMatrix M;
   Array<int> empty;
   mass.FormSystemMatrix(empty,M);

   // 8. Define the Lagrange multiplier and gradient functions
   GridFunction grad(&control_fes);
   GridFunction w_filter(&filter_fes);

   // 9. Define some tools for later
   GridFunction onegf(&control_fes);
   onegf = 1.0;
   LinearForm vol_form(&control_fes);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble();
   double domain_volume = vol_form(onegf);

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

      sout_u << "solution\n" << mesh << u;
      sout_u << "view 0 0\n";  // view from top
      sout_u << "keys jl********\n";  // turn off perspective and light
      sout_u << "window_title 'Temperature u'";
      sout_u.flush();

      GridFunction rho_gf(&control_fes);
      rho_gf.ProjectCoefficient(rho);

      sout_rho << "solution\n" << mesh << rho_gf;
      sout_rho << "view 0 0\n";  // view from top
      sout_rho << "keys jl********\n";  // turn off perspective and light
      sout_rho << "window_title 'Density ρ'";
      sout_rho.flush();

      sout_r << "solution\n" << mesh << rho_filter;
      sout_r << "view 0 0\n";  // view from top
      sout_r << "keys jl********\n";  // turn off perspective and light
      sout_r << "window_title 'Filtered density ρ̃'";
      sout_r.flush();
   }

   // mfem::ParaViewDataCollection paraview_dc("Elastic_compliance", &mesh);
   // paraview_dc.SetPrefixPath("ParaView");
   // paraview_dc.SetLevelsOfDetail(order);
   // paraview_dc.SetCycle(0);
   // paraview_dc.SetDataFormat(VTKFormat::BINARY);
   // paraview_dc.SetHighOrderOutput(true);
   // paraview_dc.SetTime(0.0);
   // paraview_dc.RegisterField("displacement",&u);
   // paraview_dc.RegisterField("density",&rho);
   // paraview_dc.RegisterField("filtered_density",&rho_filter);

   // 11. Iterate
   int step = 0;
   double c0 = 0.0;
   GridFunction zero_gf(&control_fes);
   for (int k = 1; k < max_it; k++)
   {
      const double alpha = alpha0 * k;

      cout << "\nStep = " << k << endl;

      // Step 1 - Filter solve
      // Solve (ϵ^2 ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)
      // GridFunctionCoefficient rho_cf(&rho);
      rho_filter_old = rho_filter;
      filterSolver->SetRHSCoefficient(&rho);
      filterSolver->Solve();
      rho_filter = *filterSolver->GetFEMSolution();

      // Step 2 - State solve
      // Solve (r(ρ̃) ∇u, ∇v) = (f,v)
      SIMPInterpolationCoefficient SIMP_cf(&rho_filter, rho_min, 1.0);
      GridFunctionPlusCoefficient u_plus_SIMP(&u, &SIMP_cf);
      diffusionSolver->SetDiffusionCoefficient(&u_plus_SIMP);
      diffusionSolver->Solve();
      u = *diffusionSolver->GetFEMSolution();

      // Step 3 - Adjoint filter solve
      // Solve (ϵ² ∇ w̃, ∇ v) + (w̃ ,v) = (-r'(ρ̃) (|∇ u|²),v)
      DiffusionEnergyCoefficient rhs_cf(&one, &u, &rho_filter,
                                        rho_min);
      filterSolver->SetRHSCoefficient(&rhs_cf);
      filterSolver->Solve();
      w_filter = *filterSolver->GetFEMSolution();

      // Step 4 - Compute gradient
      // Solve G = M⁻¹w̃
      GridFunctionCoefficient w_cf(&w_filter);
      LinearForm w_rhs(&control_fes);
      w_rhs.AddDomainIntegrator(new DomainLFIntegrator(w_cf));
      w_rhs.Assemble();
      M.Mult(w_rhs,grad);

      // Step 5 - Update design variable ψ ← projit(ψ - αG + c)
      //          where c is a constant so that
      //
      //               ∫ρ = ∫sigmoid(ψ) = θ|Ω|
      grad *= alpha;
      psi -= grad;
      projit(psi, mass_fraction*domain_volume);


      // Step 6 - Compute other quantities
      GridFunctionCoefficient rho_filter_cf(&rho_filter);
      const double norm_reduced_gradient = rho_filter_old.ComputeL2Error(
                                              rho_filter_cf)/alpha;
      const double compliance = (*(diffusionSolver->GetLinearForm()))(u);
      const double material_volume = zero_gf.ComputeL2Error(rho);
      mfem::out << "norm of reduced gradient = " << norm_reduced_gradient << endl;
      mfem::out << "compliance = " << compliance << endl;
      mfem::out << "mass_fraction = " << material_volume / domain_volume << endl;

      if (visualization)
      {
         sout_u << "solution\n" << mesh << u
                << "window_title 'Displacement u'" << flush;

         GridFunction rho_gf(&control_fes);
         rho_gf.ProjectCoefficient(rho);
         sout_rho << "solution\n" << mesh << rho_gf
                  << "window_title 'Control variable ρ'" << flush;

         GridFunction r_gf(&filter_fes);
         r_gf.ProjectCoefficient(SIMP_cf);
         sout_r << "solution\n" << mesh << r_gf
                << "window_title 'Design density r(ρ̃)'" << flush;
      }

      if (norm_reduced_gradient < tol && k > 1)
      {
         break;
      }
   }

   delete diffusionSolver;
   delete filterSolver;

   return 0;
}