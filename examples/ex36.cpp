// Thermal compliance - Fixed Point
//
// min (f, u)
// s.t -∇⋅(r(ρ̃)∇u) = f in Ω = (0, 20) × (0, 20)
//               u = 0 on Γ = (9, 11) × {y = 20}
//            n⋅∇u = 0 on ∂Ω \ Γ
//        -ϵΔρ̃ + ρ̃ = ρ in Ω
//            n⋅∇ρ̃ = 0 on ∂Ω
//           0 ≤ ρ ≤ 1 a.e. Ω
//
// L = (f, u) - (r(ρ̃)∇u, ∇v) + (f, v)
//    + (ϵ∇ρ̃, ∇λ̃) + (ρ̃, λ̃) - (ρ, λ)
//    + α(logit(ρ) - logit(ρ_k)

#include "mfem.hpp"
#include "proximalGalerkin.hpp"

// Solution variables
class Vars { public: enum {u, f_rho, psi, f_lam, numVars}; };

void clip_abs(mfem::Vector &x, const double max_abs_val)
{
   for (auto &val : x) { val = std::min(max_abs_val, std::max(-max_abs_val, val)); }
}

void clip(mfem::Vector &x, const double min_val, const double max_val)
{
   for (auto &val : x) { val = std::min(max_val, std::max(min_val, val)); }
}

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int problem = 0;
   const char *mesh_file = "../data/rect_with_top_fixed.mesh";
   int ref_levels = 1;
   int order = 1;
   const char *device_config = "cpu";
   bool visualization = true;
   double alpha0 = 1.0;
   double epsilon = 1e-04;
   double rho0 = 1e-6;
   int simp_exp = 3;
   double max_psi = 1e07;

   int maxit_penalty = 10000;
   int maxit_newton = 100;
   double tol_newton = 1e-8;
   double tol_penalty = 1e-6;


   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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

   Device device(device_config);
   device.Print();

   // 2. Input data (mesh, source, ...)
   Mesh mesh(mesh_file);
   int dim = mesh.Dimension();
   const int max_attributes = mesh.bdr_attributes.Max();

   double volume = 0.0;
   for (int i=0; i<mesh.GetNE(); i++) { volume += mesh.GetElementVolume(i); }

   for (int i=0; i<ref_levels; i++) { mesh.UniformRefinement(); }

   // Essential boundary for each variable (numVar x numAttr)
   Array2D<int> ess_bdr(Vars::numVars, max_attributes);
   ess_bdr = 0;
   ess_bdr(Vars::u, 0) = true;

   // Source and fixed temperature
   ConstantCoefficient heat_source(1.0);
   ConstantCoefficient u_bdr(0.0);
   const double volume_fraction = 0.7;
   const double target_volume = volume * volume_fraction;

   // 3. Finite Element Spaces and discrete solutions
   FiniteElementSpace fes_H1_Qk2(&mesh, new H1_FECollection(order + 2, dim,
                                                            mfem::BasisType::GaussLobatto));
   FiniteElementSpace fes_H1_Qk1(&mesh, new H1_FECollection(order + 1, dim,
                                                            mfem::BasisType::GaussLobatto));
   FiniteElementSpace fes_H1_Qk0(&mesh, new H1_FECollection(std::max(order + 0,1),
                                                            dim,
                                                            mfem::BasisType::GaussLobatto));
   FiniteElementSpace fes_L2_Qk2(&mesh, new L2_FECollection(order + 2, dim,
                                                            mfem::BasisType::GaussLobatto));
   FiniteElementSpace fes_L2_Qk1(&mesh, new L2_FECollection(order + 1, dim,
                                                            mfem::BasisType::GaussLobatto));
   FiniteElementSpace fes_L2_Qk0(&mesh, new L2_FECollection(order + 0, dim,
                                                            mfem::BasisType::GaussLobatto));

   Array<FiniteElementSpace*> fes(Vars::numVars);
   fes[Vars::u] = &fes_H1_Qk1;
   fes[Vars::f_rho] = &fes_H1_Qk1;
   fes[Vars::psi] = &fes_L2_Qk0;
   fes[Vars::f_lam] = fes[Vars::f_rho];

   Array<int> offsets = getOffsets(fes);
   BlockVector sol(offsets), delta_sol(offsets);
   sol = 0.0;
   delta_sol = 0.0;

   GridFunction u(fes[Vars::u], sol.GetBlock(Vars::u));
   GridFunction psi(fes[Vars::psi], sol.GetBlock(Vars::psi));
   GridFunction f_rho(fes[Vars::f_rho], sol.GetBlock(Vars::f_rho));
   GridFunction f_lam(fes[Vars::f_lam], sol.GetBlock(Vars::f_lam));

   GridFunction psi_k(fes[Vars::psi]);

   // Project solution
   Array<int> ess_bdr_u;
   ess_bdr_u.MakeRef(ess_bdr[Vars::u], max_attributes);
   u.ProjectBdrCoefficient(u_bdr, ess_bdr_u);
   psi = logit(volume_fraction);
   f_rho = volume_fraction;

   // 4. Define preliminary coefficients
   ConstantCoefficient eps_cf(epsilon);
   ConstantCoefficient alpha_k(alpha0);
   ConstantCoefficient one_cf(1.0);

   auto simp_cf = SIMPCoefficient(&f_rho, simp_exp, rho0);
   auto dsimp_cf = DerSIMPCoefficient(&f_rho, simp_exp, rho0);
   auto d2simp_cf = Der2SIMPCoefficient(&f_rho, simp_exp, rho0);
   auto rho_cf = SigmoidCoefficient(&psi);
   auto dsigmoid_cf = DerSigmoidCoefficient(&psi);

   GridFunctionCoefficient u_cf(&u);
   GridFunctionCoefficient f_rho_cf(&f_rho);
   GridFunctionCoefficient f_lam_cf(&f_lam);
   GridFunctionCoefficient psi_cf(&psi);
   GridFunctionCoefficient psi_k_cf(&psi_k);

   GradientGridFunctionCoefficient Du(&u);
   GradientGridFunctionCoefficient Df_rho(&f_rho);
   GradientGridFunctionCoefficient Df_lam(&f_lam);

   InnerProductCoefficient squared_normDu(Du, Du);

   ProductCoefficient alph_f_lam(alpha_k, f_lam_cf);
   ProductCoefficient neg_alph_f_lam(-1.0, alph_f_lam);
   ProductCoefficient neg_f_lam(-1.0, f_lam_cf);
   ProductCoefficient neg_simp(-1.0, simp_cf);
   ProductCoefficient neg_dsimp(-1.0, dsimp_cf);
   ProductCoefficient dsimp_times2(2.0, dsimp_cf);
   ProductCoefficient dsimp_squared_normDu(dsimp_cf, squared_normDu);
   ProductCoefficient neg_dsimp_squared_normDu(neg_dsimp, squared_normDu);
   ProductCoefficient d2simp_squared_normDu(d2simp_cf, squared_normDu);
   ProductCoefficient neg_dsigmoid(-1.0, dsigmoid_cf);

   ScalarVectorProductCoefficient neg_simp_Du(neg_simp, Du);
   ScalarVectorProductCoefficient neg_eps_Df_rho(-epsilon, Df_rho);
   ScalarVectorProductCoefficient neg_eps_Df_lam(-epsilon, Df_lam);
   ScalarVectorProductCoefficient dsimp_Du(dsimp_cf, Du);
   ScalarVectorProductCoefficient dsimp_Du_times2(dsimp_times2, Du);

   SumCoefficient diff_filter(rho_cf, f_rho_cf, 1.0, -1.0);
   SumCoefficient diff_psi_k(psi_k_cf, psi_cf, 1.0, -1.0);
   SumCoefficient diff_psi_grad(diff_psi_k, alph_f_lam, 1.0, -1.0);

   // 5. Define global system for newton iteration
   BlockLinearSystem newtonSystem(offsets, fes, ess_bdr);
   newtonSystem.own_blocks = true;
   for (int i=0; i<Vars::numVars; i++)
   {
      newtonSystem.SetDiagBlockMatrix(i, new BilinearForm(fes[i]));
   }

   // Equation u
   newtonSystem.GetDiagBlock(Vars::u)->AddDomainIntegrator(
      // A += (r(ρ̃^i)∇δu, ∇v)
      new DiffusionIntegrator(simp_cf)
   );
   newtonSystem.GetLinearForm(Vars::u)->AddDomainIntegrator(
      new DomainLFIntegrator(heat_source)
   );

   // Equation ρ̃
   newtonSystem.GetDiagBlock(Vars::f_rho)->AddDomainIntegrator(
      new DiffusionIntegrator(eps_cf)
   );
   newtonSystem.GetDiagBlock(Vars::f_rho)->AddDomainIntegrator(
      new MassIntegrator(one_cf)
   );
   newtonSystem.GetLinearForm(Vars::f_rho)->AddDomainIntegrator(
      new DomainLFIntegrator(rho_cf)
   );

   // Equation ψ
   newtonSystem.GetDiagBlock(Vars::psi)->AddDomainIntegrator(
      new MassIntegrator(one_cf)
   );
   newtonSystem.GetLinearForm(Vars::psi)->AddDomainIntegrator(
      new DomainLFIntegrator(psi_k_cf)
   );
   newtonSystem.GetLinearForm(Vars::psi)->AddDomainIntegrator(
      new DomainLFIntegrator(neg_alph_f_lam)
   );

   // Equation λ̃
   newtonSystem.GetDiagBlock(Vars::f_lam)->AddDomainIntegrator(
      new DiffusionIntegrator(eps_cf)
   );
   newtonSystem.GetDiagBlock(Vars::f_lam)->AddDomainIntegrator(
      new MassIntegrator(one_cf)
   );
   newtonSystem.GetLinearForm(Vars::f_lam)->AddDomainIntegrator(
      new DomainLFIntegrator(dsimp_squared_normDu)
   );


   Array<int> ordering(0); // ordering of solving the equation
   ordering.Append(Vars::f_rho);
   ordering.Append(Vars::u);
   ordering.Append(Vars::f_lam);
   ordering.Append(Vars::psi);
   // 6. Penalty Iteration
   for (int k=0; k<maxit_penalty; k++)
   {
      mfem::out << "Iteration " << k + 1 << std::endl;
      alpha_k.constant = alpha0*(k+1); // update α_k
      psi_k = psi; // update ψ_k
      for (int j=0; j<maxit_newton; j++) // Newton Iteration
      {
         mfem::out << "\tNewton Iteration " << j + 1 << ": ";
         // delta_sol = 0.0; // initialize newton difference
         // newtonSystem.Assemble(delta_sol); // Update system with current solution
         // newtonSystem.PCG(delta_sol); // Solve system
         delta_sol = sol;
         newtonSystem.SolveDiag(sol, ordering, true);
         delta_sol -= sol; // Update solution
         // newton successive difference
         const double diff_newton = std::sqrt(
                                       std::pow(delta_sol.Norml2(), 2) / delta_sol.Size()
                                    );
         // Project solution
         // NOTE: Newton stopping criteria cannot see this update. Should I consider this update?
         const double current_volume_fraction = VolumeProjection(psi,
                                                                 target_volume) / volume;
         clip_abs(psi, max_psi);
         mfem::out << std::scientific << diff_newton << ", ∫ρ / |Ω| = " << std::fixed
                   << current_volume_fraction << std::endl;

         if (diff_newton < tol_newton)
         {
            break;
         }
      } // end of Newton iteration
      const double diff_penalty = std::sqrt(
                                     psi_k.DistanceSquaredTo(psi) / delta_sol.Size()
                                  );
      mfem::out << "||ψ - ψ_k|| = " << std::scientific << diff_penalty << std::endl
                << std::endl;
      if (diff_penalty < tol_penalty)
      {
         break;
      }
   } // end of penalty iteration

   return 0;
}