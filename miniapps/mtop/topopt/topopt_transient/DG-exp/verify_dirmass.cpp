//            Native (serial) correctness harness for the directional
//            accumulated-mass constraint  (directional_mass.hpp).
//
// The full constrained optimization runs through MMA in the parallel driver
// (elasticity_topopt_par.cpp), which needs the MPI/MMA toolchain.  This small
// serial program exercises the SAME DirectionalMass class against the serial
// MFEM install so the new physics can be validated with g++ alone:
//
//   (1) transport unit test  : beta=(1,0), rho~=1  =>  m(x,y) = x  (exact).
//   (2) adjoint consistency  : c_m . (T^{-1} S rho~)  ==  (S^T T^{-T} c_m) . rho~
//                              (forward integral == precomputed-weight form).
//   (3) FD gradient test     : d/drho [ w . filter(rho) ]  vs central FD.
//
// Build (native Windows, MSYS2 UCRT64 -- serial MFEM 4.9):
//   g++ -O3 -std=c++17 -D_USE_MATH_DEFINES -I./mfem verify_dirmass.cpp \
//       -o verify_dirmass.exe -L./mfem -lmfem -lws2_32 -static
//
// Run:   ./verify_dirmass.exe [-nx 80 -ny 40 -o 1 -bx 1 -by 0 -r 0.06]

#include "mfem.hpp"
#include "topopt.hpp"            // DensityFilter
#include "directional_mass.hpp"  // DirectionalMass
#include <iostream>
#include <iomanip>
#include <cstdlib>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   int nx = 80, ny = 40, order = 1;
   real_t Lx = 2.0, bx = 1.0, by = 0.0, rmin = 0.06;
   OptionsParser args(argc, argv);
   args.AddOption(&nx, "-nx", "--nx", "elements in x");
   args.AddOption(&ny, "-ny", "--ny", "elements in y");
   args.AddOption(&order, "-o", "--order", "density H1 order");
   args.AddOption(&Lx, "-lx", "--length-x", "domain length in x (height 1)");
   args.AddOption(&bx, "-bx", "--beta-x", "transport direction x");
   args.AddOption(&by, "-by", "--beta-y", "transport direction y");
   args.AddOption(&rmin, "-r", "--rmin", "filter radius (alpha = rmin^2/12)");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, false,
                                     Lx, 1.0);
   H1_FECollection fec(order, 2);
   FiniteElementSpace fes_rho(&mesh, &fec);
   const int n = fes_rho.GetVSize();

   Vector beta(2); beta(0) = bx; beta(1) = by;
   DirectionalMass dm(fes_rho, mesh, order, beta);
   DensityFilter filter(fes_rho, rmin * rmin / 12.0);

   cout << "\n[verify_dirmass] mesh " << nx << "x" << ny << ", order " << order
        << ", beta = (" << bx << ", " << by << "), density dofs " << n << "\n";

   // ---- (1) transport unit test: beta=(1,0), rho~=1  =>  m = x -------------
   bool unit_ok = true;
   if (bx == 1.0 && by == 0.0)
   {
      Vector rho_tilde(n);  rho_tilde = 1.0;          // constant density 1
      Vector m;  const int it = dm.Solve(rho_tilde, m);
      GridFunction mgf(&dm.TransportSpace());  mgf = m;
      FunctionCoefficient xcoef([](const Vector &p) { return p[0]; });
      const real_t err = mgf.ComputeL2Error(xcoef);
      // INT_Omega x dx over [0,Lx]x[0,1] = Lx^2/2.
      const real_t integ = dm.Integral(m), exact = 0.5 * Lx * Lx;
      unit_ok = (err < 1e-8) && (std::abs(integ - exact) < 1e-6 * exact);
      cout << "\n  (1) transport unit test  (rho~=1 => m=x)\n"
           << "      ||m - x||_L2 = " << scientific << setprecision(3) << err
           << "   INT m = " << integ << "  (exact " << exact << ")"
           << "   [GMRES " << it << "]   ["
           << (unit_ok ? "PASS" : "FAIL") << "]\n";
   }
   else
   {
      cout << "\n  (1) transport unit test skipped (needs beta=(1,0))\n";
   }

   // ---- (2) adjoint consistency: forward integral == weight form ----------
   srand(12345);
   Vector rho(n);
   for (int i = 0; i < n; i++) { rho[i] = real_t(rand()) / RAND_MAX; }
   Vector rho_tilde(n);  filter.Apply(rho, rho_tilde);

   Vector w;  const int ait = dm.AdjointWeights(w);     // w = S^T T^{-T} c_m
   Vector m;  const int fit = dm.Solve(rho_tilde, m);    // m = T^{-1} S rho~
   const real_t g_fwd = dm.Integral(m);                 // c_m . m
   const real_t g_wgt = dm.DirMass(w, rho_tilde);        // w . rho~
   const real_t rel2 = std::abs(g_fwd - g_wgt) /
                       std::max(real_t(1), std::abs(g_fwd));
   const bool adj_ok = rel2 < 1e-8;
   cout << "\n  (2) adjoint consistency  (forward INT m  vs  w.rho~)\n"
        << "      forward = " << setprecision(10) << g_fwd
        << "   weight = " << g_wgt << "   rel.diff = "
        << scientific << setprecision(3) << rel2
        << "   [GMRES fwd " << fit << ", adj " << ait << "]   ["
        << (adj_ok ? "PASS" : "FAIL") << "]\n";

   // ---- (3) FD gradient test:  g(rho) = w . filter(rho) -------------------
   // dg/drho = M F^{-1} w  (filter chain rule); compare to central FD.
   Vector dgdrho;  filter.Chain(w, dgdrho);
   Vector s(n);
   for (int i = 0; i < n; i++) { s[i] = 2.0 * (real_t(rand()) / RAND_MAX) - 1.0; }
   const real_t predicted = dgdrho * s;

   auto gval = [&](const Vector &r) -> real_t
   {
      Vector rt(n);  filter.Apply(r, rt);
      return dm.DirMass(w, rt);
   };

   cout << "\n  (3) gradient finite-difference test   (predicted g.s = "
        << scientific << setprecision(8) << predicted << ")\n"
        << "      delta        central FD          rel. error\n";
   real_t best = 1.0;
   for (real_t delta : {1e-3, 1e-4, 1e-5, 1e-6})
   {
      Vector rp(rho), rm(rho);
      rp.Add(+delta, s);  rm.Add(-delta, s);
      const real_t fd = (gval(rp) - gval(rm)) / (2.0 * delta);
      const real_t rel = std::abs(fd - predicted) / std::abs(predicted);
      best = std::min(best, rel);
      printf("      %.0e   %18.10e   %.3e\n", delta, fd, rel);
   }
   const bool grad_ok = best < 1e-6;
   cout << "      => best relative error " << scientific << setprecision(2)
        << best << "   [" << (grad_ok ? "PASS" : "FAIL") << "]\n";

   const bool all_ok = unit_ok && adj_ok && grad_ok;
   cout << "\n[verify_dirmass] " << (all_ok ? "ALL TESTS PASSED" : "TESTS FAILED")
        << "\n";
   return all_ok ? 0 : 1;
}
