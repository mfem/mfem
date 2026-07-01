//                                MFEM Example 43
//
// Compile with: make ex43
//
// Sample runs:  ex43 -m ../data/beam-hex.mesh -o 3 -r 2 -d hip -reps 100
//               ex43 -m ../data/beam-hex.mesh -o 3 -r 3 -d cuda -reps 50
//
// Description:  This example isolates the element mass-matrix
//               inverse paths used by element matrix kernels. It
//               assembles upper-packed L2 element mass matrices for the
//               eq-iter-cholesky inverse path and lower-packed matrices for
//               the MAGMA packed Cholesky solve path, then times repeated
//               mass inverse applications. 

#include "mfem.hpp"
#ifdef MFEM_USE_MAGMA
#include "linalg/batched/magma.hpp"
#endif
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace mfem;
using namespace std;

namespace
{

#ifdef MFEM_USE_MAGMA
void PrintMagmaFasterCondition(const double eq_fixed_ms,
                               const double eq_apply_ms,
                               const double magma_fixed_ms,
                               const double magma_apply_ms)
{
   cout << "MAGMA faster condition (assembly+setup+N applies): ";

   if (magma_fixed_ms <= eq_fixed_ms && magma_apply_ms <= eq_apply_ms)
   {
      if (magma_fixed_ms == eq_fixed_ms && magma_apply_ms == eq_apply_ms)
      {
         cout << "modeled totals are equal for all positive integer N.\n";
      }
      else
      {
         cout << "faster for every positive integer N.\n";
      }
      return;
   }

   if (magma_fixed_ms >= eq_fixed_ms && magma_apply_ms >= eq_apply_ms)
   {
      cout << "not faster for any positive integer N.\n";
      return;
   }

   if (magma_apply_ms > eq_apply_ms)
   {
      const double crossover =
         (eq_fixed_ms - magma_fixed_ms)/(magma_apply_ms - eq_apply_ms);
      if (crossover <= 1.0)
      {
         cout << "only for N < " << crossover
              << ", so not for any positive integer N.\n";
      }
      else
      {
         const double last_n =
            floor(nextafter(crossover, -numeric_limits<double>::infinity()));
         cout << "faster for N < " << crossover
              << " applies (positive integer N <= " << (long long)last_n
              << "); eq-iter is faster above that.\n";
      }
   }
   else
   {
      const double crossover =
         (magma_fixed_ms - eq_fixed_ms)/(eq_apply_ms - magma_apply_ms);
      const double first_n = floor(crossover) + 1.0;
      cout << "faster for N > " << crossover
           << " applies (positive integer N >= " << (long long)first_n
           << "); eq-iter is faster below that.\n";
   }
}

#ifdef MFEM_USE_SINGLE
#define MFEM_EX43_MAGMA_PREFIX(stub) magma_s##stub
#define MFEM_EX43_MAGMA_SET_POINTER magma_sset_pointer
#elif defined(MFEM_USE_DOUBLE)
#define MFEM_EX43_MAGMA_PREFIX(stub) magma_d##stub
#define MFEM_EX43_MAGMA_SET_POINTER magma_dset_pointer
#endif

real_t **SetMagmaPackedPointerArray(Array<real_t *> &ptrs, real_t *data,
                                    const int stride,
                                    const int batch_size,
                                    const magma_queue_t queue)
{
   if (ptrs.Size() != batch_size)
   {
      if (ptrs.Size() != 0) { magma_queue_sync(queue); }
      ptrs.SetSize(batch_size, Device::GetDeviceMemoryType());
   }

   real_t **d_ptrs = ptrs.Write();
   MFEM_EX43_MAGMA_SET_POINTER(d_ptrs, data, 1, 0, 0, stride,
                               batch_size, queue);
   return d_ptrs;
}

void ComputeMagmaPackedCholeskyLower(
   const TriPackMatrix<TriangularPart::LOWER> &packed_lower,
   TriPackMatrix<TriangularPart::LOWER> &lower_factor,
   Array<real_t *> &factor_ptrs)
{
   const int n = packed_lower.GetNumRows();
   const int batch_size = packed_lower.GetNumMatrices();
   const int packed_size = packed_lower.GetPackedSize();

   lower_factor.SetSize(n, batch_size);
   lower_factor.UseDevice(true);

   if (batch_size == 0) { return; }

   lower_factor.Data() = packed_lower.Data();

   magma_queue_t queue = Magma::Queue();
   real_t *factor_data = lower_factor.Data().ReadWrite();
   real_t **d_factor_ptrs =
      SetMagmaPackedPointerArray(factor_ptrs, factor_data, packed_size,
                                 batch_size, queue);

   Array<magma_int_t> info_array(batch_size, Device::GetDeviceMemoryType());
   magma_int_t *d_info = info_array.Write();
   magma_memset(d_info, 0, batch_size*sizeof(magma_int_t));

   const magma_int_t status =
      (n <= 8) ?
      MFEM_EX43_MAGMA_PREFIX(pptrf_batched_small)(
         MagmaLower, n, d_factor_ptrs, d_info, batch_size, queue) :
      MFEM_EX43_MAGMA_PREFIX(pptf2_batched_small)(
         MagmaLower, n, d_factor_ptrs, d_info, batch_size, queue);
   MFEM_VERIFY(status == MAGMA_SUCCESS,
               "MAGMA packed Cholesky factorization failed.");

   magma_queue_sync(queue);

   const magma_int_t *info = info_array.HostRead();
   for (int e = 0; e < batch_size; ++e)
   {
      MFEM_VERIFY(info[e] == 0,
                  "MAGMA packed Cholesky factorization failed on matrix "
                  << e << '.');
   }
}

void SolveMagmaPackedCholeskyLowerInPlace(
   const TriPackMatrix<TriangularPart::LOWER> &lower_factor,
   const Array<real_t *> &factor_ptrs,
   Array<real_t *> &rhs_ptrs,
   Vector &rhs_sol)
{
   const int n = lower_factor.GetNumRows();
   const int batch_size = lower_factor.GetNumMatrices();

   if (batch_size == 0)
   {
      rhs_sol.SetSize(0);
      return;
   }

   MFEM_VERIFY(rhs_sol.Size() == batch_size*n,
               "Right-hand side has the wrong size.");
   MFEM_VERIFY(factor_ptrs.Size() == batch_size,
               "Factor pointer array has the wrong size.");

   magma_queue_t queue = Magma::Queue();
   real_t *rhs_data = rhs_sol.ReadWrite();
   real_t **d_factor_ptrs = const_cast<real_t **>(factor_ptrs.Read());
   real_t **d_rhs_ptrs =
      SetMagmaPackedPointerArray(rhs_ptrs, rhs_data, n, batch_size, queue);

   const magma_int_t status =
      MFEM_EX43_MAGMA_PREFIX(pptrs_batched_small)(
         n, 1, d_factor_ptrs, d_rhs_ptrs, n, batch_size, queue);
   MFEM_VERIFY(status == MAGMA_SUCCESS,
               "MAGMA packed Cholesky solve failed.");
}

#undef MFEM_EX43_MAGMA_SET_POINTER
#undef MFEM_EX43_MAGMA_PREFIX
#endif

void FillRHS(Vector &rhs)
{
   rhs.UseDevice(true);
   real_t *x = rhs.HostWrite();
   for (int i = 0; i < rhs.Size(); ++i)
   {
      x[i] = 1.0 + real_t((13*i + 7) % 29)/real_t(29);
   }
}

void ComputeLowerPackedResidual(
   const TriPackMatrix<TriangularPart::LOWER> &lower,
   const Vector &x,
   const Vector &rhs,
   double &l2_residual,
   double &relative_l2_residual,
   real_t &max_residual,
   real_t &relative_max_residual)
{
   const int n = lower.GetNumRows();
   const int batch_size = lower.GetNumMatrices();
   const int packed_size = lower.GetPackedSize();
   MFEM_VERIFY(x.Size() == batch_size*n, "Solution vector has the wrong size.");
   MFEM_VERIFY(rhs.Size() == batch_size*n, "Right-hand side has the wrong size.");

   const real_t *A = lower.Data().HostRead();
   const real_t *X = x.HostRead();
   const real_t *B = rhs.HostRead();

   long double l2_sum = 0.0;
   long double rhs_l2_sum = 0.0;
   real_t max_abs = 0.0;
   real_t rhs_max_abs = 0.0;

   for (int e = 0; e < batch_size; ++e)
   {
      const real_t *Ae = A + e*packed_size;
      const real_t *Xe = X + e*n;
      const real_t *Be = B + e*n;
      for (int i = 0; i < n; ++i)
      {
         long double ax = 0.0;
         for (int j = 0; j < n; ++j)
         {
            const real_t aij =
               (i >= j) ?
               Ae[TriPackMatrix<TriangularPart::LOWER>::LowerIndex(i, j, n)] :
               Ae[TriPackMatrix<TriangularPart::LOWER>::LowerIndex(j, i, n)];
            ax += (long double)aij * (long double)Xe[j];
         }
         const long double residual = ax - (long double)Be[i];
         l2_sum += residual*residual;
         rhs_l2_sum += (long double)Be[i]*(long double)Be[i];
         max_abs = max(max_abs, (real_t)fabs((double)residual));
         rhs_max_abs = max(rhs_max_abs, fabs(Be[i]));
      }
   }

   l2_residual = sqrt((double)l2_sum);
   const double rhs_l2_norm = sqrt((double)rhs_l2_sum);
   relative_l2_residual =
      (rhs_l2_norm > 0.0) ? l2_residual/rhs_l2_norm : l2_residual;
   max_residual = max_abs;
   relative_max_residual =
      (rhs_max_abs > 0.0) ? max_residual/rhs_max_abs : max_residual;
}

void ApplyUpperInverseInPlace(
   const TriPackMatrix<TriangularPart::UPPER> &upper_inverse,
   Vector &x,
   Vector &work)
{
   const int n = upper_inverse.GetNumRows();
   const int batch_size = upper_inverse.GetNumMatrices();
   const int packed_size = upper_inverse.GetPackedSize();
   MFEM_VERIFY(x.Size() == batch_size*n, "Input vector has the wrong size.");

   work.SetSize(batch_size*n);
   work.UseDevice(true);

   const real_t *U = upper_inverse.Data().Read();
   const real_t *X = x.Read();
   real_t *T = work.Write();

   mfem::forall(batch_size*n, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx % n;
      const int e = idx/n;
      const real_t *Ue = U + e*packed_size;
      const real_t *Xe = X + e*n;
      real_t sum = 0.0;
      for (int j = 0; j <= i; ++j)
      {
         sum += Ue[TriPackMatrix<TriangularPart::UPPER>::UpperIndex(j, i, n)]*
                Xe[j];
      }
      T[idx] = sum;
   });

   const real_t *T_in = work.Read();
   real_t *Y = x.Write();
   mfem::forall(batch_size*n, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx % n;
      const int e = idx/n;
      const real_t *Ue = U + e*packed_size;
      const real_t *Te = T_in + e*n;
      real_t sum = 0.0;
      for (int j = i; j < n; ++j)
      {
         sum += Ue[TriPackMatrix<TriangularPart::UPPER>::UpperIndex(i, j, n)]*
                Te[j];
      }
      Y[idx] = sum;
   });
}

double TimeUpperInverseApply(
   const TriPackMatrix<TriangularPart::UPPER> &inverse,
   const Vector &rhs,
   const int reps,
   Vector &x,
   Vector &work)
{
   StopWatch sw;

   // Dry run to remove first-use kernel and workspace allocation costs.
   x = rhs;
   ApplyUpperInverseInPlace(inverse, x, work);
   MFEM_DEVICE_SYNC;
   sw.Start();
   for (int r = 0; r < reps; ++r)
   {
      x = rhs;
      ApplyUpperInverseInPlace(inverse, x, work);
   }
   MFEM_DEVICE_SYNC;
   sw.Stop();
   return 1000.0*sw.RealTime()/reps;
}

#ifdef MFEM_USE_MAGMA
double TimeMagmaSolve(
   const TriPackMatrix<TriangularPart::LOWER> &lower_factor,
   const Array<real_t *> &factor_ptrs,
   const Vector &rhs,
   const int reps,
   Vector &x)
{
   StopWatch sw;
   Array<real_t *> rhs_ptrs;

   // Dry run to remove first-use MAGMA and RHS pointer-array setup costs.
   x = rhs;
   SolveMagmaPackedCholeskyLowerInPlace(lower_factor, factor_ptrs,
                                        rhs_ptrs, x);
   MFEM_DEVICE_SYNC;
   sw.Start();
   for (int r = 0; r < reps; ++r)
   {
      x = rhs;
      SolveMagmaPackedCholeskyLowerInPlace(lower_factor, factor_ptrs,
                                           rhs_ptrs, x);
   }
   MFEM_DEVICE_SYNC;
   sw.Stop();
   return 1000.0*sw.RealTime()/reps;
}
#endif

} // namespace

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/beam-hex.mesh";
   int order = 3;
   int ref_levels = 1;
   int reps = 100;
   int setup_reps = 10;
   const char *device_config = "cpu";
   bool use_magma = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of uniform refinements.");
   args.AddOption(&reps, "-reps", "--apply-repetitions",
                  "Number of mass inverse applications to time.");
   args.AddOption(&setup_reps, "-sreps", "--setup-repetitions",
                  "Number of setup repetitions to time.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&use_magma, "-magma", "--magma-solve",
                  "-no-magma", "--no-magma-solve",
                  "Time the MAGMA packed Cholesky solve when available.");
   args.ParseCheck();

   MFEM_VERIFY(reps > 0, "The number of apply repetitions must be positive.");
   MFEM_VERIFY(setup_reps > 0,
               "The number of setup repetitions must be positive.");

   Device device(device_config);
   device.Print();

#ifndef MFEM_USE_MAGMA
   MFEM_VERIFY(!use_magma, "MFEM was built without MAGMA support.");
#endif

   Mesh mesh(mesh_file, 1, 1);
   for (int l = 0; l < ref_levels; ++l)
   {
      mesh.UniformRefinement();
   }

   const int dim = mesh.Dimension();
   L2_FECollection fec(order, dim, BasisType::Positive);
   FiniteElementSpace fespace(&mesh, &fec);

   MFEM_VERIFY(UsesTensorBasis(fespace),
               "This example requires a tensor-product finite element space.");

   const int ne = mesh.GetNE();
   const int elem_dofs = fespace.GetTypicalFE()->GetDof();

   MassIntegrator mass;
   StopWatch sw;

   // Dry run each reported assembly path before timing steady-state work.
   TriPackMatrix<TriangularPart::UPPER> upper_ea;
   mass.AssembleEATriangular(fespace, upper_ea, false);
   MFEM_DEVICE_SYNC;

   TriPackMatrix<TriangularPart::LOWER> lower_ea;
   mass.AssembleEATriangular(fespace, lower_ea, false);
   MFEM_DEVICE_SYNC;

   sw.Clear();
   sw.Start();
   mass.AssembleEATriangular(fespace, upper_ea, false);
   MFEM_DEVICE_SYNC;
   sw.Stop();
   const double upper_assemble_ms = 1000.0*sw.RealTime();

   sw.Clear();
   sw.Start();
   mass.AssembleEATriangular(fespace, lower_ea, false);
   MFEM_DEVICE_SYNC;
   sw.Stop();
   const double lower_assemble_ms = 1000.0*sw.RealTime();

   TriPackMatrix<TriangularPart::UPPER> upper_inverse;
   // Dry run setup before timing steady-state setup work.
   tripack::ComputeJacobiScaledCholeskyUpperInverse(upper_ea, upper_inverse);
   MFEM_DEVICE_SYNC;

   sw.Clear();
   sw.Start();
   for (int r = 0; r < setup_reps; ++r)
   {
      tripack::ComputeJacobiScaledCholeskyUpperInverse(upper_ea, upper_inverse);
      MFEM_DEVICE_SYNC;
   }
   sw.Stop();
   const double upper_inverse_setup_ms = 1000.0*sw.RealTime()/setup_reps;

#ifdef MFEM_USE_MAGMA
   TriPackMatrix<TriangularPart::LOWER> magma_factor;
   Array<real_t *> magma_factor_ptrs;
   double magma_factor_ms = 0.0;
   if (use_magma)
   {
      // Dry run setup before timing steady-state setup work.
      ComputeMagmaPackedCholeskyLower(lower_ea, magma_factor,
                                      magma_factor_ptrs);
      MFEM_DEVICE_SYNC;

      sw.Clear();
      sw.Start();
      for (int r = 0; r < setup_reps; ++r)
      {
         ComputeMagmaPackedCholeskyLower(lower_ea, magma_factor,
                                         magma_factor_ptrs);
         MFEM_DEVICE_SYNC;
      }
      sw.Stop();
      magma_factor_ms = 1000.0*sw.RealTime()/setup_reps;
   }
#endif

   Vector rhs(ne*elem_dofs);
   FillRHS(rhs);

   Vector upper_inverse_x(rhs.Size()), work;
   upper_inverse_x.UseDevice(true);

   const double upper_inverse_apply_ms =
      TimeUpperInverseApply(upper_inverse, rhs, reps, upper_inverse_x, work);

   double upper_inverse_res_l2 = 0.0, upper_inverse_rel_res_l2 = 0.0;
   real_t upper_inverse_res_max = 0.0, upper_inverse_rel_res_max = 0.0;
   ComputeLowerPackedResidual(lower_ea, upper_inverse_x, rhs,
                              upper_inverse_res_l2, upper_inverse_rel_res_l2,
                              upper_inverse_res_max, upper_inverse_rel_res_max);

#ifdef MFEM_USE_MAGMA
   double magma_solve_ms = 0.0;
   double magma_res_l2 = 0.0, magma_rel_res_l2 = 0.0;
   real_t magma_res_max = 0.0, magma_rel_res_max = 0.0;
   Vector magma_x;
   if (use_magma)
   {
      magma_x.SetSize(rhs.Size());
      magma_x.UseDevice(true);
      magma_solve_ms =
         TimeMagmaSolve(magma_factor, magma_factor_ptrs, rhs, reps, magma_x);
      ComputeLowerPackedResidual(lower_ea, magma_x, rhs,
                                 magma_res_l2, magma_rel_res_l2,
                                 magma_res_max, magma_rel_res_max);
   }
#endif

   cout << fixed << setprecision(6);
   cout << "Mass matrix inverse microbenchmark" << '\n';
   cout << "Mesh: " << mesh_file << '\n';
   cout << "Dimension: " << dim << '\n';
   cout << "Elements: " << ne << '\n';
   cout << "Element dofs: " << elem_dofs << '\n';
   cout << "Scalar element unknowns: " << ne*elem_dofs << '\n';
   cout << "Apply repetitions: " << reps << '\n';
   cout << "Setup repetitions: " << setup_reps << '\n';
   cout << '\n';

   cout << "Assembly eq-iter-cholesky upper packed EA (ms): "
        << upper_assemble_ms << '\n';
   cout << "Assembly MAGMA lower packed EA (ms): " << lower_assemble_ms << '\n';
   cout << "Setup eq-iter-cholesky upper packed inverse (ms): "
        << upper_inverse_setup_ms << '\n';
#ifdef MFEM_USE_MAGMA
   if (use_magma)
   {
      cout << "Setup MAGMA lower packed Cholesky factor (ms): "
           << magma_factor_ms << '\n';
   }
#endif
   cout << '\n';

   cout << "Apply eq-iter-cholesky upper packed inverse (ms/apply): "
        << upper_inverse_apply_ms << '\n';
#ifdef MFEM_USE_MAGMA
   if (use_magma)
   {
      cout << "Apply MAGMA lower packed Cholesky solve (ms/apply): "
           << magma_solve_ms << '\n';
      cout << "MAGMA solve / eq-iter upper packed inverse apply: "
           << magma_solve_ms/upper_inverse_apply_ms << '\n';

      const double eq_fixed_ms = upper_assemble_ms + upper_inverse_setup_ms;
      const double magma_fixed_ms = lower_assemble_ms + magma_factor_ms;
      const double eq_total_ms =
         eq_fixed_ms + reps*upper_inverse_apply_ms;
      const double magma_total_ms = magma_fixed_ms + reps*magma_solve_ms;
      cout << "Total eq-iter upper packed inverse for current repetitions "
           << "(assembly+setup+applies, ms): " << eq_total_ms << '\n';
      cout << "Total MAGMA lower packed Cholesky solve for current repetitions "
           << "(assembly+setup+applies, ms): " << magma_total_ms << '\n';
      cout << "Faster approach for current repetitions: "
           << ((magma_total_ms < eq_total_ms) ? "MAGMA" :
               ((eq_total_ms < magma_total_ms) ? "eq-iter" : "tie"))
           << '\n';
      PrintMagmaFasterCondition(eq_fixed_ms, upper_inverse_apply_ms,
                                magma_fixed_ms, magma_solve_ms);
   }
#endif
   cout << '\n';

   cout << scientific << setprecision(12);
   cout << "Residual, eq-iter upper packed inverse, max: "
        << upper_inverse_res_max << " (relative "
        << upper_inverse_rel_res_max << "), L2: "
        << upper_inverse_res_l2 << " (relative "
        << upper_inverse_rel_res_l2 << ")\n";
#ifdef MFEM_USE_MAGMA
   if (use_magma)
   {
      cout << "Residual, MAGMA lower packed Cholesky solve, max: "
           << magma_res_max << " (relative "
           << magma_rel_res_max << "), L2: "
           << magma_res_l2 << " (relative "
           << magma_rel_res_l2 << ")\n";
   }
#endif

   return 0;
}
