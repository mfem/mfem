//                                MFEM Example X
//
// Compile with: make ex9
//
// Sample runs:
//    exX
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of implicit
//               and explicit ODE time integrators, the definition of periodic
//               boundary conditions through periodic meshes, as well as the use
//               of GLVis for persistent visualization of a time-evolving
//               solution. The saving of time-dependent data files for external
//               visualization with VisIt (visit.llnl.gov) and ParaView
//               (paraview.org) is also illustrated.

#include "mfem.hpp"
#include "proximalGalerkin.hpp"

// Solution variables
class Vars { public: enum {u, f_rho, psi, f_lam, numVars}; };

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int problem = 0;
   const char *mesh_file = "../data/rect_with_top_fixed.mesh";
   int ref_levels = 2;
   int order = 3;
   const char *device_config = "cpu";
   bool visualization = true;
   double alpha0 = 1.0;
   double epsilon = 1e-03;
   double rho0 = 1e-6;
   int simp_exp = 3;


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
   FiniteElementSpace fes_H1_Qk0(&mesh, new H1_FECollection(order + 0, dim,
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
   psi_k = logit(volume_fraction);
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
   GridFunctionCoefficient psi_k_cf(&psi);

   GradientGridFunctionCoefficient Du(&u);
   GradientGridFunctionCoefficient Df_rho(&f_rho);
   GradientGridFunctionCoefficient Df_lam(&f_lam);

   InnerProductCoefficient squared_normDu(Du, Du);

   ProductCoefficient alph_f_lam(alpha_k, f_lam_cf);
   ProductCoefficient neg_dsimp(-1.0, dsimp_cf);
   ProductCoefficient dsimp_times2(2.0, dsimp_cf);
   ProductCoefficient neg_dsimp_normDu(neg_dsimp, squared_normDu);
   ProductCoefficient d2simp_normDu(d2simp_cf, squared_normDu);
   ProductCoefficient neg_dsigmoid(-1.0, dsigmoid_cf);

   ScalarVectorProductCoefficient neg_eps_Df_rho(-epsilon, Df_rho);
   ScalarVectorProductCoefficient neg_eps_Df_lam(-epsilon, Df_lam);
   ScalarVectorProductCoefficient dsimp_Du(dsimp_cf, Du);
   ScalarVectorProductCoefficient dsimp_Du_times2(dsimp_times2, Du);

   SumCoefficient diff_filter(rho_cf, f_rho_cf, 1.0, -1.0);
   SumCoefficient diff_psi_k(psi_k_cf, psi_cf, 1.0, -1.0);
   SumCoefficient diff_psi_grad(diff_psi_k, alph_f_lam, 1.0, -1.0);

   // 5. Define global system for newton iteration
   BlockLinearSystem globalSystem(fes, ess_bdr);
   globalSystem.own_blocks = true;
   for (int i=0; i<Vars::numVars; i++){
      globalSystem.SetDiagBlockMatrix(i, new BilinearForm(fes[i]));
      globalSystem.SetBlockVector(i, new LinearForm(fes[i]));
   }
   std::vector<std::vector<int>> offDiagBlocks{
      {Vars::u, Vars::f_rho},
      {Vars::f_rho, Vars::psi},
      {Vars::psi, Vars::f_lam},
      {Vars::f_lam, Vars::u},
      {Vars::f_lam, Vars::f_rho}
   };
   for(auto idx: offDiagBlocks)
   {
      globalSystem.SetBlockMatrix(idx[0], idx[1], new MixedBilinearForm(fes[idx[0]], fes[idx[1]]));
   }


   return 0;
}

