//          Compliance-minimization topology optimization (MFEM, PARALLEL/MPI)
//
// Distributed-memory version of elasticity_topopt.cpp (the cantilever).  Same
// mathematics, same CLI; the serial->parallel changes are the scaling-roadmap
// substitutions documented in topopt.hpp and thermal_topopt_par.cpp.  The
// elasticity-specific parallel detail is in the header: ParEllipticOperator
// gives HypreBoomerAMG the rigid-body systems options for the vector space.
//
// ALL optimization vectors (rho, rho~, dc/drho, u) are LOCAL TRUE-DOF chunks.
// Runs on the serial MFEM install too via `mpirun -np 1`.
//
// Build (parallel MFEM 4.9):
//   mpicxx -O3 -std=c++17 -I$MFEM/include elasticity_topopt_par.cpp \
//       [MMA_MFEM.cpp -DTOPOPT_WITH_MMA] -o elasticity_topopt_par \
//       -L$MFEM/lib -lmfem -L$HYPRE/lib -lHYPRE -L$METIS/lib -lmetis -llapack
//
// Run:
//   mpirun -np 4 ./elasticity_topopt_par -p opt
//   mpirun -np 4 ./elasticity_topopt_par -p mms
//   mpirun -np 4 ./elasticity_topopt_par -p check
// ---------------------------------------------------------------------------

#include "mfem.hpp"
#include "topopt.hpp"
#include "directional_mass.hpp"   // ParDirectionalMass: beta.grad m = rho~ + adjoint
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <memory>

using namespace std;
using namespace mfem;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Lame parameters from (E, nu); plane stress uses the reduced lambda.
void LameFromENu(real_t E, real_t nu, bool plane_stress,
                 real_t &lambda, real_t &mu)
{
   lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
   mu     = E / (2.0 * (1.0 + nu));
   if (plane_stress) { lambda = 2.0 * lambda * mu / (lambda + 2.0 * mu); }
}

// ParGridFunction::ComputeL2Error already does the global MPI reduction, so it
// is used directly for the displacement L2 error.  The H1 seminorm has no
// vector ComputeGradError, so it is integrated locally and reduced once below.

// Global |u - Jex|_H1 for a vector ParGridFunction (MFEM's ComputeGradError is
// scalar-only): integrate ||grad u_h - Jex||_F^2 over LOCAL elements, reduce.
static real_t GlobalVectorGradError(MPI_Comm comm, const ParGridFunction &u,
                                    MatrixCoefficient &Jex)
{
   const FiniteElementSpace &fes = *u.FESpace();
   Mesh &m = *fes.GetMesh();
   const int dim = m.Dimension();
   DenseMatrix Jh(dim), J(dim);
   real_t err2 = 0.0;
   for (int e = 0; e < fes.GetNE(); e++)
   {
      const FiniteElement &fe = *fes.GetFE(e);
      ElementTransformation &Tr = *m.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(fe.GetGeomType(),
                                               2 * fe.GetOrder() + 3);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         u.GetVectorGradient(Tr, Jh);
         Jex.Eval(J, Tr, ip);
         Jh -= J;
         err2 += ip.weight * Tr.Weight() * Jh.FNorm2();
      }
   }
   real_t glob;  MPI_Allreduce(&err2, &glob, 1, MPITypeMap<real_t>::mpi_type,
                               MPI_SUM, comm);
   return std::sqrt(glob);
}

// ===========================================================================
//  Parallel cantilever compliance-minimization problem (clamped x=0, downward
//  traction patch on the middle of the right edge).  Scalar density space +
//  vector displacement space on one ParMesh; constant filter and load built
//  once.
// ===========================================================================
struct ParElastTopOpt
{
   ParMesh pmesh;
   H1_FECollection fec;
   ParFiniteElementSpace fes_rho;        // scalar: density
   ParFiniteElementSpace fes_u;          // vector: displacement

   real_t emin, e0, penal, alpha, volfrac;
   real_t lambda0, mu0;
   const IntegrationRule *ir;

   Array<int> clamp_bdr, load_bdr;
   Vector tvec;
   std::unique_ptr<VectorConstantCoefficient> traction_cf;
   std::unique_ptr<HypreParVector> load_tv;   // true-dof load (assembled once)

   std::unique_ptr<ParDensityFilter> filter;

   ParElastTopOpt(ParMesh &&pmesh_, int order, real_t Lx, real_t emin_,
                  real_t e0_, real_t penal_, real_t nu, bool plane_stress,
                  real_t rmin, real_t volfrac_, real_t load_half)
      : pmesh(std::move(pmesh_)),
        fec(order, 2),
        fes_rho(&pmesh, &fec),
        fes_u(&pmesh, &fec, 2),
        emin(emin_), e0(e0_), penal(penal_), volfrac(volfrac_)
   {
      alpha = rmin * rmin / 12.0;
      LameFromENu(1.0, nu, plane_stress, lambda0, mu0);
      Geometry::Type geom = fes_u.GetFE(0)->GetGeomType();
      ir = &IntRules.Get(geom, 2 * order + 4);

      SetupBoundaries(Lx, load_half);
      filter = std::make_unique<ParDensityFilter>(fes_rho, alpha);

      tvec.SetSize(2);  tvec(0) = 0.0;  tvec(1) = -1.0;
      traction_cf = std::make_unique<VectorConstantCoefficient>(tvec);
      ParLinearForm load(&fes_u);
      load.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*traction_cf),
                                 load_bdr);
      load.Assemble();
      load_tv.reset(load.ParallelAssemble());      // true-dof load vector
   }

   // Relabel boundary: 2 = clamp (x=0), 3 = load patch, 1 = traction-free.
   void SetupBoundaries(real_t Lx, real_t load_half)
   {
      const real_t yc = 0.5;
      for (int i = 0; i < pmesh.GetNBE(); i++)
      {
         Element *be = pmesh.GetBdrElement(i);
         Array<int> v;  be->GetVertices(v);
         real_t xm = 0.0, ym = 0.0;
         for (int k = 0; k < v.Size(); k++)
         {
            const real_t *c = pmesh.GetVertex(v[k]);
            xm += c[0];  ym += c[1];
         }
         xm /= v.Size();  ym /= v.Size();
         int attr = 1;
         if (xm < 1e-12) { attr = 2; }
         else if (xm > Lx - 1e-12 && std::abs(ym - yc) <= load_half) { attr = 3; }
         be->SetAttribute(attr);
      }
      pmesh.SetAttributes();
      int locmax = pmesh.bdr_attributes.Size() ? pmesh.bdr_attributes.Max() : 0;
      int glomax;  MPI_Allreduce(&locmax, &glomax, 1, MPI_INT, MPI_MAX,
                                 pmesh.GetComm());
      MFEM_VERIFY(glomax >= 3, "no load patch found");
      clamp_bdr.SetSize(glomax);  clamp_bdr = 0;  clamp_bdr[1] = 1;  // clamp
      load_bdr.SetSize(glomax);   load_bdr = 0;   load_bdr[2] = 1;   // load
   }

   // Solve K(rho~) u = load on true dofs; compliance c = load.u (global).
   real_t SolveState(ParGridFunction &rho_tilde_gf, Vector &u_tdof,
                     int *iters = nullptr, real_t reltol = 1e-10)
   {
      SIMPCoefficient Ecf(rho_tilde_gf, emin, e0, penal);
      ParEllipticOperator K(fes_u, &Ecf, nullptr, ir, &clamp_bdr, -1,
                            lambda0, mu0);
      K.SetRelTol(reltol);  K.SetMaxIter(50000);

      u_tdof.SetSize(fes_u.GetTrueVSize());  u_tdof = 0.0;
      const int sit = K.Solve(*load_tv, u_tdof);
      if (iters) { *iters = sit; }

      return InnerProduct(fes_u.GetComm(), *load_tv, u_tdof);   // c = load.u
   }

   void Sensitivity(ParGridFunction &rho_tilde_gf, ParGridFunction &u_gf,
                    Vector &dcdrho)
   {
      SIMPGradEnergyCoefficient wcf(rho_tilde_gf, u_gf, emin, e0, penal,
                                    lambda0, mu0);
      ParLinearForm g(&fes_rho);
      DomainLFIntegrator *lfi = new DomainLFIntegrator(wcf);
      lfi->SetIntRule(ir);
      g.AddDomainIntegrator(lfi);
      g.Assemble();
      std::unique_ptr<HypreParVector> G(g.ParallelAssemble());
      filter->Chain(*G, dcdrho);
   }

   real_t Compliance(const Vector &rho, real_t reltol = 1e-12)
   {
      Vector rt(fes_rho.GetTrueVSize()), u;
      filter->Apply(rho, rt);
      ParGridFunction rt_gf(&fes_rho);  rt_gf.SetFromTrueDofs(rt);
      return SolveState(rt_gf, u, nullptr, reltol);
   }
};

#ifdef TOPOPT_WITH_MMA
// ===========================================================================
//  Two-constraint MMA updater: compliance min subject to BOTH the volume
//  budget and the directional accumulated-mass budget.  Both constraints are
//  AFFINE in the design,  g_i(rho) = dg_i . rho - 1 <= 0,  with CONSTANT
//  gradients dg_i (volume weights v/Vstar and the filtered transport-adjoint
//  weights w/Mstar), so each Update() only forms two global inner products and
//  hands the exact (constant) gradients to MMA.
// ===========================================================================
class MMAUpdaterParDM
{
   MPI_Comm comm;
   mfem_mma::MMAOptimizerParallel mma;
   real_t move;
   Vector fival, xmin, xmax, xold;
   std::vector<Vector> dfi;             // { d g_vol/d rho , d g_dm/d rho }

public:
   MMAUpdaterParDM(MPI_Comm comm_, const Vector &rho,
                   const Vector &dgvol, const Vector &dgdm, real_t move_)
      : comm(comm_), mma(comm_, rho.Size(), 2, rho), move(move_),
        fival(2), xmin(rho.Size()), xmax(rho.Size()), dfi{dgvol, dgdm} {}

   real_t Update(Vector &rho, real_t compliance, const Vector &dcdrho)
   {
      const int n = rho.Size();
      fival(0) = InnerProduct(comm, dfi[0], rho) - 1.0;   // volume      (global)
      fival(1) = InnerProduct(comm, dfi[1], rho) - 1.0;   // directional (global)
      for (int i = 0; i < n; i++)
      {
         xmin[i] = std::max(real_t(0), rho[i] - move);
         xmax[i] = std::min(real_t(1), rho[i] + move);
      }
      xold = rho;
      mma.Update(rho, dcdrho, compliance, fival, dfi.data(), xmin, xmax);
      real_t change = 0.0;
      for (int i = 0; i < n; i++)
      { change = std::max(change, std::abs(rho[i] - xold[i])); }
      MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, comm);
      return change;
   }
};
#endif // TOPOPT_WITH_MMA

// ===========================================================================
//  MODE: topology optimization  -> frames_elast/  +  topopt_elast_history.csv
// ===========================================================================
int RunOpt(MPI_Comm comm, int rank, int nx, int ny, int order, real_t Lx,
           real_t emin, real_t e0, real_t penal, real_t nu, bool plane_stress,
           real_t rmin, real_t volfrac, real_t move, real_t tol, int max_it,
           real_t load_half, int save_every, const string &optname,
           bool use_dm, real_t bx, real_t by, real_t dmfrac)
{
   Mesh smesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, false,
                                      Lx, 1.0);
   ParElastTopOpt prob(ParMesh(comm, smesh), order, Lx, emin, e0, penal, nu,
                       plane_stress, rmin, volfrac, load_half);
   smesh.Clear();

   const int n = prob.fes_rho.GetTrueVSize();
   if (rank == 0)
      cout << "\n[opt] density dofs = " << prob.fes_rho.GlobalTrueVSize()
           << ",  displacement dofs = " << prob.fes_u.GlobalTrueVSize()
           << ",  alpha = " << prob.alpha
           << ",  |Omega| = " << prob.filter->DomainVolume()
           << ",  optimizer = " << optname
           << "\n      (lambda0, mu0) = (" << prob.lambda0 << ", " << prob.mu0
           << ")  [" << (plane_stress ? "plane stress" : "plane strain")
           << "]  (" << Mpi::WorldSize() << " ranks)\n\n";

   std::filesystem::create_directory("frames_elast");
   ExportNodalCoordsPar(prob.fes_rho, "frames_elast/coords.csv");
   ofstream hist;
   if (rank == 0) { hist.open("topopt_elast_history.csv");
                    hist << "iter,compliance,volume,change"
                         << (use_dm ? ",dirmass\n" : "\n"); }

   Vector rho(n);  rho = volfrac;
   Vector rho_tilde(n), dcdrho;
   ParGridFunction rho_tilde_gf(&prob.fes_rho), u_gf(&prob.fes_u);

   const Vector &vw = prob.filter->VolWeights();
   const real_t max_volume = volfrac * prob.filter->DomainVolume();

   // ----- Directional accumulated-mass constraint (optional) ---------------
   // g_dm(rho) = (INT_Omega m dx)/Mstar - 1 <= 0,  beta.grad m = rho~,  m=0 at
   // inflow.  The functional is affine in rho, so its gradient is the CONSTANT
   // vector  dgdm = (M F^{-1} S^T T^{-T} c_m)/Mstar  -- one adjoint transport
   // solve at setup; the budget Mstar is dmfrac x the initial directional mass.
   std::unique_ptr<ParDirectionalMass> dirmass;
   Vector w_tilde, dgdm;
   real_t M_ref = 0.0, Mstar = 0.0;
   if (use_dm)
   {
      Vector bvec(2);  bvec(0) = bx;  bvec(1) = by;
      dirmass = std::make_unique<ParDirectionalMass>(prob.fes_rho, prob.pmesh,
                                                     order, bvec);
      const int ait = dirmass->AdjointWeights(w_tilde);     // w = S^T T^{-T} c_m
      prob.filter->Apply(rho, rho_tilde);                   // rho~ of uniform rho
      M_ref = dirmass->DirMass(w_tilde, rho_tilde);         // reference dir. mass
      Mstar = dmfrac * M_ref;
      prob.filter->Chain(w_tilde, dgdm);                    // M F^{-1} w
      dgdm /= Mstar;                                        // d g_dm / d rho
      if (rank == 0)
         cout << "[opt] directional-mass constraint: beta = (" << bx << ", "
              << by << "),  M_ref = " << M_ref << ",  budget = " << dmfrac
              << " x M_ref  (adjoint transport GMRES " << ait << ")\n\n";
   }

#ifdef TOPOPT_WITH_MMA
   std::unique_ptr<MMAUpdaterPar>   mma;
   std::unique_ptr<MMAUpdaterParDM> dmma;
   if (use_dm)
   {
      Vector dgvol(vw);  dgvol /= max_volume;               // d g_vol / d rho
      dmma = std::make_unique<MMAUpdaterParDM>(comm, rho, dgvol, dgdm, move);
   }
   else if (optname == "mma")
   { mma = std::make_unique<MMAUpdaterPar>(comm, rho, vw, max_volume, move); }
#else
   MFEM_VERIFY(optname != "mma" && !use_dm,
               "rebuild with -DTOPOPT_WITH_MMA + MMA_MFEM.cpp (MMA needed for "
               "-opt mma and for the directional-mass constraint)");
#endif

   real_t change = 1.0, c = 0.0;
   int it = 0, frame = 0;
   for (; it < max_it && change > tol; it++)
   {
      prob.filter->Apply(rho, rho_tilde);
      rho_tilde_gf.SetFromTrueDofs(rho_tilde);
      Vector u_tdof;
      int sit = 0;
      c = prob.SolveState(rho_tilde_gf, u_tdof, &sit);
      u_gf.SetFromTrueDofs(u_tdof);
      prob.Sensitivity(rho_tilde_gf, u_gf, dcdrho);
#ifdef TOPOPT_WITH_MMA
      if (dmma)     { change = dmma->Update(rho, c, dcdrho); }
      else if (mma) { change = mma->Update(rho, c, dcdrho); }
      else
#endif
      { change = OCUpdate(comm, rho, dcdrho, vw, max_volume, move); }

      const real_t vol = prob.filter->Volume(rho) / prob.filter->DomainVolume();
      // Directional mass as a fraction of the reference (M_ref); affine in rho,
      // so InnerProduct(dgdm, rho)*Mstar = w.rho~ = current directional mass.
      const real_t dm_frac =
         use_dm ? InnerProduct(comm, dgdm, rho) * Mstar / M_ref : 0.0;
      if (rank == 0)
      {
         if (use_dm)
            printf("  it %3d   c = %.6e   vol = %.4f   dm = %.4f   "
                   "change = %.4f   (state CG %d)\n",
                   it + 1, c, vol, dm_frac, change, sit);
         else
            printf("  it %3d   c = %.6e   vol = %.4f   change = %.4f   "
                   "(state CG %d)\n", it + 1, c, vol, change, sit);
         hist << it + 1 << "," << c << "," << vol << "," << change;
         if (use_dm) { hist << "," << dm_frac; }
         hist << "\n";
      }
      if (it % save_every == 0)
      {
         char fn[80];  snprintf(fn, sizeof(fn), "frames_elast/rho_%04d.csv", frame++);
         SaveFieldPar(rho_tilde, fn, rank);
      }
   }
   prob.filter->Apply(rho, rho_tilde);
   char fn[80];  snprintf(fn, sizeof(fn), "frames_elast/rho_%04d.csv", frame);
   SaveFieldPar(rho_tilde, fn, rank);
   SaveFieldPar(rho_tilde, "frames_elast/rho_final.csv", rank);

   if (rank == 0)
      cout << "\n[opt] finished after " << it << " iterations,  final compliance = "
           << scientific << setprecision(6) << c << "\n"
           << "      wrote " << frame + 1 << " frame(s) x " << Mpi::WorldSize()
           << " ranks + topopt_elast_history.csv\n";
   return 0;
}

// ===========================================================================
//  MODE: manufactured-solution convergence study (parallel).
//   u1 = u2 = sin(pi x) sin(pi y),  u = 0 on dOmega, constant (lambda, mu).
// ===========================================================================
void RunMMSOrder(MPI_Comm comm, int rank, ofstream &csv, int order,
                 int nlevels, int n0, real_t nu, bool plane_stress)
{
   real_t lam, mu;
   LameFromENu(1.0, nu, plane_stress, lam, mu);

   VectorFunctionCoefficient uex_cf(2, [](const Vector &p, Vector &u)
   { const real_t S = sin(M_PI * p[0]) * sin(M_PI * p[1]);  u(0) = S;  u(1) = S; });
   VectorFunctionCoefficient f_cf(2, [lam, mu](const Vector &p, Vector &f)
   {
      const real_t Sx = sin(M_PI * p[0]), Cx = cos(M_PI * p[0]);
      const real_t Sy = sin(M_PI * p[1]), Cy = cos(M_PI * p[1]);
      const real_t v = M_PI * M_PI
                       * ((lam + 3.0 * mu) * Sx * Sy - (lam + mu) * Cx * Cy);
      f(0) = v;  f(1) = v;
   });
   MatrixFunctionCoefficient gradu_cf(2, [](const Vector &p, DenseMatrix &J)
   {
      const real_t Sx = sin(M_PI * p[0]), Cx = cos(M_PI * p[0]);
      const real_t Sy = sin(M_PI * p[1]), Cy = cos(M_PI * p[1]);
      J(0, 0) = M_PI * Cx * Sy;  J(0, 1) = M_PI * Sx * Cy;
      J(1, 0) = M_PI * Cx * Sy;  J(1, 1) = M_PI * Sx * Cy;
   });

   if (rank == 0)
   {
      cout << "\n  order " << order << "\n";
      cout << "   h          dofs     ||e||_L2      rate    |e|_H1       rate\n";
      cout << "  ------------------------------------------------------------------\n";
   }

   real_t prevL2 = 0, prevH1 = 0;
   for (int l = 0; l < nlevels; l++)
   {
      const int nn = n0 << l;
      Mesh smesh = Mesh::MakeCartesian2D(nn, nn, Element::QUADRILATERAL, false,
                                         1.0, 1.0);
      ParMesh pmesh(comm, smesh);  smesh.Clear();
      H1_FECollection fec(order, 2);
      ParFiniteElementSpace fes(&pmesh, &fec, 2);

      Array<int> ess_bdr(pmesh.bdr_attributes.Max());  ess_bdr = 1;
      ConstantCoefficient one(1.0);
      ParEllipticOperator A(fes, &one, nullptr, nullptr, &ess_bdr, -1, lam, mu);
      A.SetMaxIter(40000);

      ParLinearForm b(&fes);
      b.AddDomainIntegrator(new VectorDomainLFIntegrator(f_cf));
      b.Assemble();
      std::unique_ptr<HypreParVector> B(b.ParallelAssemble());

      ParGridFunction u(&fes);  u = 0.0;
      Vector U(fes.GetTrueVSize());  U = 0.0;
      A.Solve(*B, U);
      u.SetFromTrueDofs(U);

      const real_t L2 = u.ComputeL2Error(uex_cf);      // already MPI-reduced
      const real_t H1 = GlobalVectorGradError(comm, u, gradu_cf);
      const real_t h  = 1.0 / nn;
      const real_t rL2 = (l > 0) ? log(prevL2 / L2) / log(2.0) : 0.0;
      const real_t rH1 = (l > 0) ? log(prevH1 / H1) / log(2.0) : 0.0;
      if (rank == 0)
      {
         printf("  %8.5f  %7lld  %10.4e  %6.3f  %10.4e  %6.3f\n",
                h, (long long)fes.GlobalTrueVSize(), L2, rL2, H1, rH1);
         csv << order << "," << l << "," << h << "," << fes.GlobalTrueVSize()
             << "," << L2 << "," << H1 << "\n";
      }
      prevL2 = L2;  prevH1 = H1;
   }
}

int RunMMS(MPI_Comm comm, int rank, int nlevels, int n0, real_t nu,
           bool plane_stress)
{
   ofstream csv;
   if (rank == 0) { csv.open("mms_elast_convergence.csv");
                    csv << "order,level,h,dofs,l2,h1\n"; }
   if (rank == 0)
      cout << "\n[mms] elasticity solver convergence (u1 = u2 = sin pi x sin pi y)\n";
   for (int order = 1; order <= 3; order++)
   { RunMMSOrder(comm, rank, csv, order, nlevels, n0, nu, plane_stress); }
   if (rank == 0) cout << "\n[mms] wrote mms_elast_convergence.csv\n";
   return 0;
}

// ===========================================================================
//  MODE: correctness checks (parallel): FD gradient + filter volume.
// ===========================================================================
int RunCheck(MPI_Comm comm, int rank, int nx, int ny, int order, real_t Lx,
             real_t emin, real_t e0, real_t penal, real_t nu, bool plane_stress,
             real_t rmin, real_t volfrac, real_t load_half,
             bool use_dm, real_t bx, real_t by, real_t dmfrac)
{
   Mesh smesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, false,
                                      Lx, 1.0);
   ParElastTopOpt prob(ParMesh(comm, smesh), order, Lx, emin, e0, penal, nu,
                       plane_stress, rmin, volfrac, load_half);
   smesh.Clear();
   const int n = prob.fes_rho.GetTrueVSize();
   if (rank == 0)
      cout << "\n[check] mesh " << nx << "x" << ny << ", order " << order
           << ", density dofs " << prob.fes_rho.GlobalTrueVSize()
           << ", displacement dofs " << prob.fes_u.GlobalTrueVSize()
           << "  (" << Mpi::WorldSize() << " ranks)\n";

   Vector rho(n), s(n);
   srand(1 + rank);
   for (int i = 0; i < n; i++)
   {
      rho[i] = 0.5;
      s[i]   = 2.0 * (real_t(rand()) / RAND_MAX) - 1.0;
   }

   Vector rt(n), u, g;
   prob.filter->Apply(rho, rt);
   ParGridFunction rt_gf(&prob.fes_rho), u_gf(&prob.fes_u);
   rt_gf.SetFromTrueDofs(rt);
   prob.SolveState(rt_gf, u, nullptr, 1e-13);
   u_gf.SetFromTrueDofs(u);
   prob.Sensitivity(rt_gf, u_gf, g);
   const real_t predicted = InnerProduct(comm, g, s);

   if (rank == 0)
   {
      cout << "\n  (1) gradient finite-difference test   (predicted g.s = "
           << scientific << setprecision(8) << predicted << ")\n";
      cout << "      delta        central FD          rel. error\n";
      cout << "      ---------------------------------------------------\n";
   }
   real_t best = 1.0;
   for (real_t delta : {1e-4, 1e-5, 1e-6, 1e-7})
   {
      Vector rp(rho), rm(rho);
      rp.Add(+delta, s);
      rm.Add(-delta, s);
      const real_t cp = prob.Compliance(rp);
      const real_t cm = prob.Compliance(rm);
      const real_t fd = (cp - cm) / (2.0 * delta);
      const real_t rel = std::abs(fd - predicted) / std::abs(predicted);
      best = std::min(best, rel);
      if (rank == 0) { printf("      %.0e   %18.10e   %.3e\n", delta, fd, rel); }
   }
   const bool grad_ok = best < 1e-4;
   if (rank == 0)
      cout << "      => best relative error " << scientific << setprecision(2)
           << best << "   [" << (grad_ok ? "PASS" : "FAIL") << "]\n";

   prob.filter->Apply(rho, rt);
   const real_t v_rho = prob.filter->Volume(rho);
   const real_t v_rt  = prob.filter->Volume(rt);
   const real_t vrel  = std::abs(v_rt - v_rho) / std::abs(v_rho);
   const bool vol_ok  = vrel < 1e-8;
   if (rank == 0)
   {
      cout << "\n  (2) filter volume preservation\n";
      printf("      INT rho = %.10e   INT rho~ = %.10e   rel. diff = %.2e   [%s]\n",
             v_rho, v_rt, vrel, vol_ok ? "PASS" : "FAIL");
   }

   // ----- (3) directional accumulated-mass constraint (optional) -----------
   bool dm_ok = true;
   if (use_dm)
   {
      Vector bvec(2);  bvec(0) = bx;  bvec(1) = by;
      ParDirectionalMass dirmass(prob.fes_rho, prob.pmesh, order, bvec);
      Vector w;  dirmass.AdjointWeights(w);                  // S^T T^{-T} c_m

      // adjoint consistency:  c_m . (T^{-1} S rho~)  ==  w . rho~
      Vector m;  dirmass.Solve(rt, m);
      const real_t g_fwd = dirmass.Integral(m);
      const real_t g_wgt = dirmass.DirMass(w, rt);
      const real_t rel_adj = std::abs(g_fwd - g_wgt) /
                             std::max(real_t(1), std::abs(g_fwd));

      // FD test of  g_dm(rho) = (w . filter(rho)) / Mstar - 1.
      const real_t M_ref = dirmass.DirMass(w, rt);
      const real_t Mstar = dmfrac * M_ref;
      Vector dgdm;  prob.filter->Chain(w, dgdm);  dgdm /= Mstar;
      const real_t pred_dm = InnerProduct(comm, dgdm, s);
      auto gdm = [&](const Vector &r) -> real_t
      {
         Vector tt(prob.fes_rho.GetTrueVSize());
         prob.filter->Apply(r, tt);
         return dirmass.DirMass(w, tt) / Mstar - 1.0;
      };
      real_t best_dm = 1.0;
      Vector fds(4);  int k = 0;
      for (real_t delta : {1e-4, 1e-5, 1e-6, 1e-7})
      {
         Vector rp(rho), rm(rho);
         rp.Add(+delta, s);  rm.Add(-delta, s);
         const real_t fd = (gdm(rp) - gdm(rm)) / (2.0 * delta);
         fds(k++) = fd;
         best_dm = std::min(best_dm, std::abs(fd - pred_dm) /
                                     std::max(real_t(1e-30), std::abs(pred_dm)));
      }
      dm_ok = (rel_adj < 1e-8) && (best_dm < 1e-4);
      if (rank == 0)
      {
         cout << "\n  (3) directional-mass constraint (beta = (" << bx << ", "
              << by << "))\n";
         printf("      adjoint consistency: forward %.10e  weight %.10e  "
                "rel %.2e\n", g_fwd, g_wgt, rel_adj);
         printf("      gradient FD: predicted g.s = %.8e   best rel. error "
                "%.2e   [%s]\n", pred_dm, best_dm, dm_ok ? "PASS" : "FAIL");
      }
   }

   if (rank == 0)
      cout << "\n[check] " << ((grad_ok && vol_ok && dm_ok) ? "ALL TESTS PASSED"
                                                            : "TESTS FAILED")
           << "\n";
   return (grad_ok && vol_ok && dm_ok) ? 0 : 1;
}

// ===========================================================================
int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   const int rank = Mpi::WorldRank();
   MPI_Comm comm = MPI_COMM_WORLD;

   const char *problem = "opt";
#ifdef TOPOPT_WITH_MMA
   const char *optimizer = "mma";    // default when MMA is compiled in
#else
   const char *optimizer = "oc";     // built without MMA: OC only
#endif
   int order = 1, nx = 120, ny = 60, max_it = 100, save_every = 1;
   int mms_levels = 5, mms_n0 = 4, check_n = 12;
   real_t Lx = 2.0, volfrac = 0.5, penal = 3.0, rmin = 0.06, move = 0.2, tol = 1e-2;
   real_t emin = 1e-3, e0 = 1.0, nu = 0.3, load_half = 0.05;
   bool plane_stress = true;
   bool use_dm = false;
   real_t bx = 1.0, by = 0.0, dmfrac = 0.5;

   OptionsParser args(argc, argv);
   args.AddOption(&problem, "-p", "--problem", "opt | mms | check");
   args.AddOption(&optimizer, "-opt", "--optimizer",
                  "mma | oc (design update; default mma where built with MMA)");
   args.AddOption(&order, "-o", "--order", "Lagrange (H1) polynomial order");
   args.AddOption(&nx, "-nx", "--nx", "elements in x (opt mode)");
   args.AddOption(&ny, "-ny", "--ny", "elements in y (opt mode)");
   args.AddOption(&Lx, "-lx", "--length-x", "cantilever length (height is 1)");
   args.AddOption(&volfrac, "-vf", "--volume-fraction", "material volume fraction");
   args.AddOption(&penal, "-pen", "--penal", "SIMP penalization power p");
   args.AddOption(&rmin, "-r", "--rmin", "filter radius (alpha = rmin^2/12)");
   args.AddOption(&move, "-mv", "--move", "OC move limit");
   args.AddOption(&tol, "-tol", "--tol", "OC stopping tolerance on max design change");
   args.AddOption(&max_it, "-mi", "--max-it", "max OC iterations");
   args.AddOption(&emin, "-emin", "--emin", "void Young modulus (SIMP lower bound)");
   args.AddOption(&e0, "-e0", "--e0", "solid Young modulus");
   args.AddOption(&nu, "-nu", "--nu", "Poisson ratio");
   args.AddOption(&plane_stress, "-pstress", "--plane-stress",
                  "-pstrain", "--plane-strain", "2D constitutive reduction");
   args.AddOption(&load_half, "-lh", "--load-half", "half-length of the load patch");
   args.AddOption(&save_every, "-se", "--save-every", "save a density frame every N its");
   args.AddOption(&use_dm, "-dm", "--directional-mass", "-no-dm",
                  "--no-directional-mass",
                  "add the directional accumulated-mass constraint (needs MMA)");
   args.AddOption(&bx, "-bx", "--beta-x", "directional-mass transport direction x");
   args.AddOption(&by, "-by", "--beta-y", "directional-mass transport direction y");
   args.AddOption(&dmfrac, "-dmf", "--dirmass-frac",
                  "directional-mass budget as a fraction of the initial "
                  "(uniform-design) directional mass");
   args.AddOption(&mms_levels, "-ml", "--mms-levels", "mms: number of refinements");
   args.AddOption(&mms_n0, "-mn", "--mms-n0", "mms: coarsest elements per side");
   args.AddOption(&check_n, "-cn", "--check-n", "check: elements in y (x gets 2x)");
   args.Parse();
   if (!args.Good()) { if (rank == 0) { args.PrintUsage(cout); } return 1; }
   if (rank == 0) { args.PrintOptions(cout); }

   const string prob(problem);
   int ret = 1;
   if (prob == "opt")
      ret = RunOpt(comm, rank, nx, ny, order, Lx, emin, e0, penal, nu,
                   plane_stress, rmin, volfrac, move, tol, max_it, load_half,
                   save_every, optimizer, use_dm, bx, by, dmfrac);
   else if (prob == "mms")
      ret = RunMMS(comm, rank, mms_levels, mms_n0, nu, plane_stress);
   else if (prob == "check")
      ret = RunCheck(comm, rank, 2 * check_n, check_n, order, Lx, emin, e0,
                     penal, nu, plane_stress, rmin, volfrac, load_half,
                     use_dm, bx, by, dmfrac);
   else if (rank == 0)
      cout << "unknown problem '" << prob << "' (use opt | mms | check)\n";

   return ret;
}
