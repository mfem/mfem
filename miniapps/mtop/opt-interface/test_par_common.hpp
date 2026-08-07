/**
 * test_par_common.hpp  —  Shared scaffolding for the parallel solver tests
 * (MMA / SQ / CCSA).  Distributed compliance proxy
 *     min Σ_global 1/x_j  s.t.  mean_global(x) {<=,=} Vfrac,  0.001<=x<=1
 * with objective/constraint reduced globally and gradients/bounds local.
 * Exercises: inequality AND equality constraint paths, and the zero-rank case
 * (a rank owning n_local=0 dofs).  Optimum: uniform x = Vfrac on ranks with dofs.
 */
#pragma once

#include <mfem.hpp>

#ifdef MFEM_USE_MPI
#include <cmath>
#include <cstdio>
#include <memory>

namespace optpar
{
using namespace mfem;

// Distributed mean constraint  m(x) = mean_global(x) - Vfrac.  Mult() reduces
// globally; the gradient is the local chunk (1/n_global) applied via VJP.
class ParMeanConstraint : public Operator
{
public:
   ParMeanConstraint(MPI_Comm comm, int n_local, int n_global, real_t Vfrac)
      : Operator(1, n_local), comm_(comm), nglob_(n_global), vf_(Vfrac),
        grad_(n_local, n_global) {}
   void Mult(const Vector &x, Vector &y) const override
   {
      double loc = 0.0;
      for (int j = 0; j < width; ++j) { loc += double(x(j)); }
      double glob = 0.0;
      MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
      y.SetSize(1);
      y(0) = real_t(glob / nglob_) - vf_;
   }
   Operator &GetGradient(const Vector &) const override { return grad_; }
private:
   class Grad : public Operator
   {
   public:
      Grad(int n_local, int n_global) : Operator(1, n_local), nglob_(n_global) {}
      void MultTranspose(const Vector &dy, Vector &dx) const override
      { dx.SetSize(width); dx = real_t(dy(0)) / nglob_; }
      void Mult(const Vector &dx, Vector &y) const override
      { real_t s = 0; for (int j = 0; j < width; ++j) { s += dx(j); } y.SetSize(1); y(0) = s / nglob_; }
   private:
      int nglob_;
   };
   MPI_Comm comm_;
   int nglob_;
   real_t vf_;
   mutable Grad grad_;
};

// Distributed compliance problem. The mean is registered as an equality (C) or
// an inequality (D) constraint; objective is globally reduced.
class ParComplianceProblem : public OptimizationProblem
{
public:
   ParComplianceProblem(MPI_Comm comm, int n_local, Operator *meanop,
                        bool equality, const Vector &xlo, const Vector &xhi)
      : OptimizationProblem(n_local, equality ? meanop : nullptr,
                            equality ? nullptr : meanop),
        comm_(comm), own_(meanop), xlo_(xlo), xhi_(xhi)
   {
      if (equality)
      { ce_.SetSize(1); ce_ = 0.0; SetEqualityConstraint(ce_); }
      else
      { dlo_.SetSize(1); dlo_ = -infinity(); dhi_.SetSize(1); dhi_ = 0.0;
        SetInequalityConstraint(dlo_, dhi_); }
      SetSolutionBounds(xlo_, xhi_);
   }
   real_t CalcObjective(const Vector &x) const override
   {
      double loc = 0.0;
      for (int j = 0; j < input_size; ++j) { loc += 1.0 / double(x(j)); }
      double glob = 0.0;
      MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
      return real_t(glob);
   }
   void CalcObjectiveGrad(const Vector &x, Vector &g) const override
   {
      g.SetSize(input_size);
      for (int j = 0; j < input_size; ++j) { g(j) = real_t(-1.0) / (x(j) * x(j)); }
   }
private:
   MPI_Comm comm_;
   std::unique_ptr<Operator> own_;
   Vector ce_, dlo_, dhi_, xlo_, xhi_;
};

/// Run one distributed compliance solve with @tparam Solver and check the global
/// optimum (uniform x=Vfrac). @p n_local may be 0 on some ranks (zero-rank case).
/// Returns 0 on pass, 1 on fail. Prints one line on rank 0.
template <class Solver>
int RunParallelCompliance(MPI_Comm comm, bool equality, int n_local,
                          real_t Vfrac, const char *tag)
{
   int rank = 0, nranks = 1;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nranks);
   int n_global = 0;
   MPI_Allreduce(&n_local, &n_global, 1, MPI_INT, MPI_SUM, comm);

   Vector xlo(n_local), xhi(n_local); xlo = real_t(0.001); xhi = real_t(1.0);
   ParComplianceProblem prob(comm, n_local,
                             new ParMeanConstraint(comm, n_local, n_global, Vfrac),
                             equality, xlo, xhi);
   Solver solver(comm);
   solver.SetOptimizationProblem(prob);
   solver.SetRelTol(0.0); solver.SetAbsTol(1e-4); solver.SetMaxIter(500);

   Vector x0(n_local), x(n_local);
   x0 = real_t(0.5);
   solver.Mult(x0, x);

   double loc_sum = 0.0, loc_maxerr = 0.0;
   for (int j = 0; j < n_local; ++j)
   {
      loc_sum += double(x(j));
      loc_maxerr = std::max(loc_maxerr, std::abs(double(x(j)) - double(Vfrac)));
   }
   double gsum = 0.0, gmax = 0.0;
   MPI_Allreduce(&loc_sum, &gsum, 1, MPI_DOUBLE, MPI_SUM, comm);
   MPI_Allreduce(&loc_maxerr, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm);
   const double xmean = gsum / n_global;

   const bool ok = solver.GetConverged() &&
                   std::abs(xmean - double(Vfrac)) < 0.01 && gmax < 0.05;
   if (rank == 0)
   {
      printf("  [%s] %-26s conv=%d iters=%d xmean=%.5f(%.2f) maxerr=%.2e\n",
             ok ? "PASS" : "FAIL", tag, solver.GetConverged(),
             solver.GetNumIterations(), xmean, double(Vfrac), gmax);
   }
   return ok ? 0 : 1;
}

/// Run the standard scenario set (LE, EQ, and — with >1 rank — zero-rank LE)
/// for @tparam Solver. Returns the global failure count.
template <class Solver>
int RunAllScenarios(MPI_Comm comm, const char *label)
{
   int rank = 0, nranks = 1;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nranks);
   if (rank == 0) { printf("=== %s (nranks=%d) ===\n", label, nranks); }

   const int base = 32;
   int fail = 0;
   fail += RunParallelCompliance<Solver>(comm, /*eq=*/false, base, real_t(0.4), "inequality");
   fail += RunParallelCompliance<Solver>(comm, /*eq=*/true,  base, real_t(0.4), "equality");
   if (nranks > 1)
   {
      // Last rank owns 0 dofs -> exercises the zero-rank code path.
      const int nloc = (rank == nranks - 1) ? 0 : base;
      fail += RunParallelCompliance<Solver>(comm, /*eq=*/false, nloc, real_t(0.4), "zero-rank ineq");
   }

   int gfail = 0;
   MPI_Allreduce(&fail, &gfail, 1, MPI_INT, MPI_SUM, comm);
   if (rank == 0)
   {
      printf("%s\n", gfail == 0 ? "  ALL PARALLEL SCENARIOS PASSED."
                                : "  SOME PARALLEL SCENARIOS FAILED.");
   }
   return gfail;
}

} // namespace optpar
#endif // MFEM_USE_MPI
