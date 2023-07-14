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
//    + (ϵ∇ρ̃, ∇λ̃) + (ρ̃, λ̃) - (S(ψ), λ)
//    + α⁻¹D(S(ψ), S(ψ_k))

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
   int ref_levels = 2;
   int order = 0;
   const char *device_config = "cpu";
   bool visualization = true;
   double alpha0 = 1.0;
   double epsilon = 1e-04;
   double rho0 = 1e-6;
   int simp_exp = 3;
   double max_psi = 1e07;

   int maxit_penalty = 10000;
   int maxit_newton = 1;
   double tol_newton = 1e-6;
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
   Mesh mesh;
   switch (problem)
   {
      case 0:
         mesh = Mesh::MakeCartesian2D(32, 32, mfem::Element::QUADRILATERAL, false,
                                      1.0, 1.0);
         break;
      case 1:
         mesh = Mesh(mesh_file);
         break;
      default:
         mfem_error("Undefined Problem");
   }
   // Mesh mesh(mesh_file);
   int dim = mesh.Dimension();
   const int max_attributes = mesh.bdr_attributes.Max();

   double volume = 0.0;
   for (int i=0; i<mesh.GetNE(); i++) { volume += mesh.GetElementVolume(i); }

   for (int i=0; i<ref_levels; i++) { mesh.UniformRefinement(); }

   // Essential boundary for each variable (numVar x numAttr)
   Array2D<int> ess_bdr(Vars::numVars, max_attributes);
   ess_bdr = 0;
   switch (problem)
   {
      case 0:
         ess_bdr(Vars::u, 2) = true;
         ess_bdr(Vars::u, 3) = true;
         break;
      case 1:
         ess_bdr(Vars::u, 0) = true;
         break;
   }

   // Source and fixed temperature
   ConstantCoefficient heat_source(1.0);
   switch (problem)
   {
      case 0:
         heat_source.constant = 1e-02;
         break;
      case 1:
         heat_source.constant = 1e-03;
         break;
   }
   ConstantCoefficient u_bdr(0.0);
   const double volume_fraction = 0.4;
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
   GridFunction zero_gf(&fes_L2_Qk2);
   zero_gf = 0.0;

   auto simp_cf = SIMPCoefficient(&f_rho, simp_exp, rho0);
   auto dsimp_cf = DerSIMPCoefficient(&f_rho, simp_exp, rho0);
   auto d2simp_cf = Der2SIMPCoefficient(&f_rho, simp_exp, rho0);
   auto rho_cf = SigmoidCoefficient(&psi);
   auto rho_k_cf = SigmoidCoefficient(&psi_k);
   auto dsigmoid_cf = DerSigmoidCoefficient(&psi);

   GridFunctionCoefficient u_cf(&u);
   GridFunctionCoefficient f_rho_cf(&f_rho);
   GridFunctionCoefficient f_lam_cf(&f_lam);
   GridFunctionCoefficient psi_cf(&psi);
   GridFunctionCoefficient psi_k_cf(&psi_k);

   SumCoefficient diff_rho(rho_cf, rho_k_cf, 1.0, -1.0);

   GradientGridFunctionCoefficient Du(&u);
   GradientGridFunctionCoefficient Df_rho(&f_rho);
   GradientGridFunctionCoefficient Df_lam(&f_lam);

   InnerProductCoefficient squared_normDu(Du, Du);

   ProductCoefficient alph_f_lam(alpha_k, f_lam_cf);
   ProductCoefficient alph_f_lam_dsigmoid(alph_f_lam, dsigmoid_cf);
   ProductCoefficient psi_k_dsigmoid(psi_k_cf, dsigmoid_cf);
   ProductCoefficient dsimp_squared_normDu(dsimp_cf, squared_normDu);

   // 5. Define global system for newton iteration
   BlockLinearSystem fixedPointSystem(offsets, fes, ess_bdr);
   fixedPointSystem.own_blocks = true;
   for (int i=0; i<Vars::numVars; i++)
   {
      fixedPointSystem.SetDiagBlockMatrix(i, new BilinearForm(fes[i]));
   }

   // Equation u
   fixedPointSystem.GetDiagBlock(Vars::u)->AddDomainIntegrator(
      // A += (r(ρ̃^i)∇δu, ∇v)
      new DiffusionIntegrator(simp_cf)
   );
   fixedPointSystem.GetLinearForm(Vars::u)->AddDomainIntegrator(
      new DomainLFIntegrator(heat_source)
   );

   // Equation ρ̃
   fixedPointSystem.GetDiagBlock(Vars::f_rho)->AddDomainIntegrator(
      new DiffusionIntegrator(eps_cf)
   );
   fixedPointSystem.GetDiagBlock(Vars::f_rho)->AddDomainIntegrator(
      new MassIntegrator(one_cf)
   );
   fixedPointSystem.GetLinearForm(Vars::f_rho)->AddDomainIntegrator(
      new DomainLFIntegrator(rho_cf)
   );

   // Equation ψ
   fixedPointSystem.GetDiagBlock(Vars::psi)->AddDomainIntegrator(
      new MassIntegrator(dsigmoid_cf)
   );
   fixedPointSystem.GetLinearForm(Vars::psi)->AddDomainIntegrator(
      new DomainLFIntegrator(psi_k_dsigmoid)
   );
   fixedPointSystem.GetLinearForm(Vars::psi)->AddDomainIntegrator(
      new DomainLFIntegrator(alph_f_lam_dsigmoid)
   );

   // Equation λ̃
   fixedPointSystem.GetDiagBlock(Vars::f_lam)->AddDomainIntegrator(
      new DiffusionIntegrator(eps_cf)
   );
   fixedPointSystem.GetDiagBlock(Vars::f_lam)->AddDomainIntegrator(
      new MassIntegrator(one_cf)
   );
   fixedPointSystem.GetLinearForm(Vars::f_lam)->AddDomainIntegrator(
      new DomainLFIntegrator(dsimp_squared_normDu)
   );



   socketstream sout_u, sout_rho, sout_f_rho;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout_u.open(vishost, visport);
      sout_rho.open(vishost, visport);
      sout_f_rho.open(vishost, visport);
      if (!sout_u)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout_u.precision(precision);
         sout_u << "solution\n" << mesh << u;
         sout_u << "window_title 'u'\n";
         sout_u << "keys jmmR**c\n";
         sout_u << flush;
         sout_rho.precision(precision);
         GridFunction rho(&fes_L2_Qk2);
         rho.ProjectCoefficient(rho_cf);
         sout_rho << "solution\n" << mesh << rho;
         sout_rho << "window_title 'ρ'\n";
         sout_rho << "autoscale off\n";
         sout_rho << "valuerange 0.0 1.0\n";
         sout_rho << "keys jmmR**c\n";
         sout_rho << flush;
         sout_f_rho << "solution\n" << mesh << f_rho;
         sout_f_rho << "window_title 'ρ̃'\n";
         sout_f_rho << "autoscale off\n";
         sout_f_rho << "valuerange 0.0 1.0\n";
         sout_f_rho << "keys jmmR**c\n";
         sout_f_rho << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }


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
      bool newton_converged = false;
      for (int j=0; j<maxit_newton; j++) // Newton Iteration
      {
         mfem::out << "\tNewton Iteration " << std::setw(5) << j + 1 << ": " <<
                   std::flush;
         // delta_sol = 0.0; // initialize newton difference
         // fixedPointSystem.Assemble(delta_sol); // Update system with current solution
         // fixedPointSystem.PCG(delta_sol); // Solve system
         Vector old_sol(sol);
         // fixedPointSystem.Assemble(sol);
         // fixedPointSystem.PCG(sol);
         fixedPointSystem.SolveDiag(sol, ordering, true);
         // Project solution
         // NOTE: Newton stopping criteria cannot see this update. Should I consider this update?
         const double current_volume_fraction = VolumeProjection(psi,
                                                                 target_volume) / volume;
         // newton successive difference
         clip_abs(psi, max_psi);
         const double diff_newton = std::sqrt(old_sol.DistanceSquaredTo(
                                                 sol) / old_sol.Size());
         mfem::out << std::scientific << diff_newton << std::endl;

         if (diff_newton < tol_newton)
         {
            newton_converged = true;
            break;
         }
      } // end of Newton iteration
      if (!newton_converged)
      {
         mfem::out << "Newton failed to converge" << std::endl;
      }
      if (visualization)
      {

         sout_u << "solution\n" << mesh << u << flush;
         GridFunction rho(&fes_L2_Qk2);
         rho.ProjectCoefficient(rho_cf);
         sout_rho << "solution\n" << mesh << rho << "valuerange 0.0 1.0\n" << flush;
         sout_f_rho << "solution\n" << mesh << f_rho << "valuerange 0.0 1.0\n" << flush;


         ostringstream filename;
         ofstream file;

         filename << "mesh" << std::setfill('0') << std::setw(6) << k << ".mesh";
         mesh.Save(filename.str().c_str());
         filename.str(std::string());

         filename << "u" << std::setfill('0') << std::setw(6) << k << ".gf";
         file.open(filename.str());
         GridFunction u_high(&fes_L2_Qk2);
         u_high.ProjectCoefficient(u_cf);
         u_high.Save(file);
         file.close();
         file.clear();
         filename.str(std::string());

         filename << "rho" << std::setfill('0') << std::setw(6) << k << ".gf";
         file.open(filename.str());
         rho.Save(file);
         file.close();
         file.clear();
         filename.str(std::string());

         GridFunction f_rho_high(&fes_L2_Qk2);
         f_rho_high.ProjectCoefficient(f_rho_cf);
         filename << "f_rho" << std::setfill('0') << std::setw(6) << k << ".gf";
         file.open(filename.str());
         f_rho_high.Save(file);
         file.close();
         file.clear();
         filename.str(std::string());
      }
      const double diff_penalty = zero_gf.ComputeL2Error(diff_rho) /
                                  alpha_k.constant / std::sqrt(volume);
      mfem::out << "||ρ - ρ_k|| = " << std::scientific << diff_penalty << std::endl
                << std::endl;
      if (diff_penalty < tol_penalty)
      {
         break;
      }
   } // end of penalty iteration

   return 0;
}