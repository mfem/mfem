/**
 * test_mma_scaling.cpp
 *
 * Regression tests for the scale-aware p/q positivity floor used by MMA.
 * The tests exercise the two invariances required by finite-element design
 * vectors:
 *
 *   1. Rescaling a function and its gradient must not change an MMA step.
 *   2. Splitting one cell into equal children (and splitting its integral
 *      derivatives equally) must not change the child design update.
 *
 * Exact-zero gradients are also checked because BuildCoeffs must retain a
 * strictly positive numerical floor to avoid 0/0 in the unconstrained
 * closed-form update.
 */

#include "MMA_MFEM.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using mfem::Vector;
using mfem_mma::MMAOptimizer;

namespace
{

int rank_id = 0;
int nranks = 1;
int failures = 0;

void Check(bool ok, const char *name, double error = 0.0)
{
   if (!ok) { failures++; }
   if (rank_id == 0)
   {
      std::printf("  [%s] %s", ok ? "PASS" : "FAIL", name);
      if (error != 0.0) { std::printf(" (error %.3e)", error); }
      std::printf("\n");
   }
}

double MaxDiff(const Vector &a, const Vector &b)
{
   double error = 0.0;
   for (int i = 0; i < a.Size(); i++)
   {
      error = std::max(error, std::abs(double(a(i)) - double(b(i))));
   }
   return error;
}

Vector SerialScaleStep(double scale)
{
   const int n = 6;
   Vector x(n), xmin(n), xmax(n), gradient(n);
   const double initial[n] = {0.23, 0.37, 0.46, 0.54, 0.68, 0.79};
   const double base_gradient[n] = {-4.0, -1.0, -0.15, 0.2, 1.5, 5.0};
   for (int i = 0; i < n; i++)
   {
      x(i) = initial[i];
      gradient(i) = scale * base_gradient[i];
   }
   xmin = 0.0;
   xmax = 1.0;

   MMAOptimizer optimizer(n, 0, x);
   optimizer.Update(x, gradient, scale * 2.75, xmin, xmax);
   return x;
}

void TestSerialFunctionScaling()
{
   if (rank_id != 0) { return; }
   Vector reference = SerialScaleStep(1.0);
   Vector tiny = SerialScaleStep(1e-12);
   Vector huge = SerialScaleStep(1e12);
   const double error = std::max(MaxDiff(reference, tiny),
                                 MaxDiff(reference, huge));
   Check(error < 2e-12, "serial function-scale invariance", error);
}

Vector SerialRefinementStep(int children, bool constrained)
{
   const int n_parent = 5;
   const int n = children * n_parent;
   const double initial[n_parent] = {0.31, 0.42, 0.50, 0.61, 0.73};
   const double objective_gradient[n_parent] =
      {-3.0e-6, -0.8e-6, 0.25e-6, 1.4e-6, 4.0e-6};
   const double constraint_gradient[n_parent] =
      {0.7e-6, 1.3e-6, 0.9e-6, 1.8e-6, 1.1e-6};

   Vector x(n), xmin(n), xmax(n), df0(n);
   Vector dfi[1];
   if (constrained) { dfi[0].SetSize(n); }
   for (int j = 0; j < n; j++)
   {
      const int parent = j / children;
      x(j) = initial[parent];
      df0(j) = objective_gradient[parent] / children;
      if (constrained)
      {
         dfi[0](j) = constraint_gradient[parent] / children;
      }
   }
   xmin = 0.0;
   xmax = 1.0;

   MMAOptimizer optimizer(n, constrained ? 1 : 0, x);
   if (constrained)
   {
      Vector fival(1);
      fival(0) = -0.015;
      optimizer.Update(x, df0, 0.25, fival, dfi, xmin, xmax);
   }
   else
   {
      optimizer.Update(x, df0, 0.25, xmin, xmax);
   }
   return x;
}

void TestSerialMeshRefinement()
{
   if (rank_id != 0) { return; }
   for (int constrained = 0; constrained <= 1; constrained++)
   {
      Vector coarse = SerialRefinementStep(1, constrained != 0);
      Vector refined = SerialRefinementStep(2, constrained != 0);
      double error = 0.0;
      for (int j = 0; j < refined.Size(); j++)
      {
         error = std::max(error,
                          std::abs(double(refined(j)) -
                                   double(coarse(j / 2))));
      }
      Check(error < (constrained ? 2e-7 : 2e-12),
            constrained ? "serial constrained refinement invariance"
                        : "serial unconstrained refinement invariance",
            error);
   }
}

void TestSerialCommonCellWeightCancellation()
{
   if (rank_id != 0) { return; }

   // A P0 finite-element derivative is a sensitivity density times the cell
   // volume. If objective and volume-constraint densities are spatially
   // constant, unequal tetrahedron volumes must not create unequal updates.
   // The old absolute p/q floor dominated these O(1e-6) derivatives and
   // produced precisely that mesh-volume imprint.
   const int n = 6;
   const double weights[n] =
      {0.25e-6, 0.5e-6, 1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6};
   Vector x(n), df0(n), xmin(n), xmax(n), fival(1);
   Vector dfi[1];
   dfi[0].SetSize(n);
   x = 0.5;
   xmin = 0.0;
   xmax = 1.0;
   for (int j = 0; j < n; j++)
   {
      df0(j) = -2.0 * weights[j];
      dfi[0](j) = weights[j];
   }
   fival(0) = 0.0;

   MMAOptimizer optimizer(n, 1, x);
   optimizer.Update(x, df0, 0.1, fival, dfi, xmin, xmax);

   double min_update = double(x(0)) - 0.5;
   double max_update = min_update;
   for (int j = 1; j < n; j++)
   {
      const double update = double(x(j)) - 0.5;
      min_update = std::min(min_update, update);
      max_update = std::max(max_update, update);
   }
   const double spread = max_update - min_update;
   Check(spread < 2e-7, "serial common cell-weight cancellation", spread);
}

void TestSerialZeroGradient()
{
   if (rank_id != 0) { return; }
   const int n = 4;
   Vector x(n), initial(n), gradient(n), xmin(n), xmax(n);
   x(0) = 0.17;
   x(1) = 0.39;
   x(2) = 0.63;
   x(3) = 0.84;
   initial = x;
   gradient = 0.0;
   xmin = 0.0;
   xmax = 1.0;
   MMAOptimizer optimizer(n, 0, x);
   optimizer.Update(x, gradient, 0.0, xmin, xmax);

   bool finite = true;
   for (int i = 0; i < n; i++) { finite = finite && std::isfinite(double(x(i))); }
   const double error = MaxDiff(x, initial);
   Check(finite && error < 2e-12, "serial exact-zero gradient", error);
}

#ifdef MFEM_USE_MPI

using mfem_mma::MMAOptimizerParallel;

void Distribution(int n_global, int &n_local, int &offset)
{
   const int base = n_global / nranks;
   const int remainder = n_global % nranks;
   n_local = base + (rank_id < remainder ? 1 : 0);
   offset = rank_id * base + std::min(rank_id, remainder);
}

double GlobalMax(double local)
{
   double global = 0.0;
   MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return global;
}

Vector ParallelScaleStep(double scale)
{
   const int n_global = 13;
   int n_local, offset;
   Distribution(n_global, n_local, offset);
   Vector x(n_local), xmin(n_local), xmax(n_local), gradient(n_local);
   for (int j = 0; j < n_local; j++)
   {
      const int g = offset + j;
      x(j) = 0.2 + 0.6 * (g + 1.0) / (n_global + 1.0);
      const double sign = (g % 2 == 0) ? -1.0 : 1.0;
      gradient(j) = scale * sign * (0.2 + g);
   }
   xmin = 0.0;
   xmax = 1.0;
   MMAOptimizerParallel optimizer(MPI_COMM_WORLD, n_local, 0, x);
   optimizer.Update(x, gradient, scale, xmin, xmax);
   return x;
}

void TestParallelFunctionScaling()
{
   Vector reference = ParallelScaleStep(1.0);
   Vector tiny = ParallelScaleStep(1e-12);
   Vector huge = ParallelScaleStep(1e12);
   const double local_error = std::max(MaxDiff(reference, tiny),
                                       MaxDiff(reference, huge));
   const double error = GlobalMax(local_error);
   Check(error < 2e-12, "parallel function-scale invariance", error);
}

void FillParentData(int parent, double &x, double &df0, double &dfc)
{
   static const double initial[6] = {0.29, 0.38, 0.47, 0.55, 0.66, 0.77};
   static const double objective_gradient[6] =
      {-2.7e-6, -1.1e-6, -0.2e-6, 0.4e-6, 1.6e-6, 3.8e-6};
   static const double constraint_gradient[6] =
      {0.8e-6, 1.5e-6, 0.6e-6, 1.9e-6, 1.2e-6, 0.9e-6};
   x = initial[parent];
   df0 = objective_gradient[parent];
   dfc = constraint_gradient[parent];
}

void TestParallelRefinementAgainstSerial()
{
   const int n_parent = 6;

   // Every rank can form this tiny serial reference independently.  Comparing
   // the distributed refined result against it checks both the global scale
   // reduction and the serial/parallel BuildCoeffs paths.
   Vector x_reference(n_parent), df0_reference(n_parent);
   Vector dfc_reference[1];
   dfc_reference[0].SetSize(n_parent);
   for (int p = 0; p < n_parent; p++)
   {
      double x, df0, dfc;
      FillParentData(p, x, df0, dfc);
      x_reference(p) = x;
      df0_reference(p) = df0;
      dfc_reference[0](p) = dfc;
   }
   Vector xmin_reference(n_parent), xmax_reference(n_parent), fival(1);
   xmin_reference = 0.0;
   xmax_reference = 1.0;
   fival(0) = -0.02;
   MMAOptimizer serial(n_parent, 1, x_reference);
   serial.Update(x_reference, df0_reference, 0.1, fival, dfc_reference,
                 xmin_reference, xmax_reference);

   const int n_refined = 2 * n_parent;
   int n_local, offset;
   Distribution(n_refined, n_local, offset);
   Vector x(n_local), xmin(n_local), xmax(n_local), df0(n_local);
   Vector dfi[1];
   dfi[0].SetSize(n_local);
   for (int j = 0; j < n_local; j++)
   {
      const int parent = (offset + j) / 2;
      double xp, gp, cp;
      FillParentData(parent, xp, gp, cp);
      x(j) = xp;
      df0(j) = 0.5 * gp;
      dfi[0](j) = 0.5 * cp;
   }
   xmin = 0.0;
   xmax = 1.0;
   MMAOptimizerParallel parallel(MPI_COMM_WORLD, n_local, 1, x);
   parallel.Update(x, df0, 0.1, fival, dfi, xmin, xmax);

   double local_error = 0.0;
   for (int j = 0; j < n_local; j++)
   {
      const int parent = (offset + j) / 2;
      local_error = std::max(local_error,
                             std::abs(double(x(j)) -
                                      double(x_reference(parent))));
   }
   const double error = GlobalMax(local_error);
   Check(error < 2e-7, "parallel constrained refinement invariance", error);
}

void TestParallelZeroGradientAndZeroDofRanks()
{
   // With four ranks, two ranks intentionally own no variables.
   const int n_global = 2;
   int n_local, offset;
   Distribution(n_global, n_local, offset);
   Vector x(n_local), initial(n_local), gradient(n_local),
          xmin(n_local), xmax(n_local);
   for (int j = 0; j < n_local; j++) { x(j) = 0.3 + 0.4 * (offset + j); }
   initial = x;
   gradient = 0.0;
   xmin = 0.0;
   xmax = 1.0;
   MMAOptimizerParallel optimizer(MPI_COMM_WORLD, n_local, 0, x);
   optimizer.Update(x, gradient, 0.0, xmin, xmax);

   double local_error = 0.0;
   bool local_finite = true;
   for (int j = 0; j < n_local; j++)
   {
      local_error = std::max(local_error,
                             std::abs(double(x(j)) - double(initial(j))));
      local_finite = local_finite && std::isfinite(double(x(j)));
   }
   int finite_int = local_finite ? 1 : 0;
   int finite_global = 0;
   MPI_Allreduce(&finite_int, &finite_global, 1, MPI_INT, MPI_MIN,
                 MPI_COMM_WORLD);
   const double error = GlobalMax(local_error);
   Check(finite_global == 1 && error < 2e-12,
         "parallel exact-zero gradient (including zero-DOF ranks)", error);
}

#endif // MFEM_USE_MPI

} // namespace

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
   MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#else
   (void)argc;
   (void)argv;
#endif

   if (rank_id == 0)
   {
      std::printf("=== MMA scaling/refinement regression tests (%d rank(s)) ===\n",
                  nranks);
   }

   TestSerialFunctionScaling();
   TestSerialMeshRefinement();
   TestSerialCommonCellWeightCancellation();
   TestSerialZeroGradient();

#ifdef MFEM_USE_MPI
   MPI_Barrier(MPI_COMM_WORLD);
   TestParallelFunctionScaling();
   TestParallelRefinementAgainstSerial();
   TestParallelZeroGradientAndZeroDofRanks();
#endif

#ifdef MFEM_USE_MPI
   int global_failures = 0;
   MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_MAX,
                 MPI_COMM_WORLD);
   failures = global_failures;
#endif

   if (rank_id == 0)
   {
      std::printf("=== %s ===\n", failures == 0 ? "ALL PASSED" : "FAILED");
   }

#ifdef MFEM_USE_MPI
   MPI_Finalize();
#endif
   return failures == 0 ? 0 : 1;
}
